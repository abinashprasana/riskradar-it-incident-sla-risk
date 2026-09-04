"""A small GRU over event prefixes, and the tensors to feed it.

Phase 3b. The order ablation (`scripts/run_order_ablation.py`) already measured
that event order carries no reliable signal over the multiset of activities on
either log. This model exists to confirm that at the model level rather than
the feature level: if a recurrent encoder given the raw ordered stream cannot
beat a tuned gradient-boosted model on aggregate features, the null result
holds for architectures, not just for hand-built encodings.

Two sizing decisions follow from the data rather than from convention:

* **Embeddings are tiny.** UCI's activity vocabulary is six symbols. A d_model
  of 64 would give a six-token alphabet more capacity than it can use; the
  activity embedding is 4 dimensions.
* **One model spans every prefix length**, using a padding mask, rather than
  one model per k. Index encoding failed partly because bucket k=8 had ~1,500
  training cases against a widening feature space. Masking lets every case
  contribute at every length it reaches, which is the specific fix for that.

Static case attributes join the pooled representation once at the end rather
than being repeated at every timestep, and `priority` and `reopen_count` are
treated as static because they are constant within 98.4% and 98.8% of cases
respectively -- repeating them per event would be parameters spent on nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Levels rarer than this collapse into a single OTHER bucket. caller_id has
# 5,022 levels over 23k cases; embedding the tail would be memorisation.
MIN_LEVEL_COUNT = 30


@dataclass
class Vocab:
    """String levels to integer ids, with 0 reserved for pad/unknown."""

    mapping: dict = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.mapping) + 2  # +pad(0) +other(1)

    @classmethod
    def fit(cls, values: pd.Series, min_count: int = MIN_LEVEL_COUNT) -> "Vocab":
        counts = values.dropna().astype(str).value_counts()
        keep = counts[counts >= min_count].index
        return cls({v: i + 2 for i, v in enumerate(keep)})

    def encode(self, values: pd.Series) -> np.ndarray:
        return values.astype(str).map(self.mapping).fillna(1).to_numpy(dtype=np.int64)

    def embed_dim(self) -> int:
        # Small by design; see module docstring.
        return int(min(16, max(2, round(self.size ** 0.5))))


@dataclass
class SeqSpec:
    """Which columns become the time axis and which stay static."""

    event_cat: tuple[str, ...]
    event_num: tuple[str, ...]
    static_cat: tuple[str, ...]
    static_num: tuple[str, ...]


def spec_for(events: pd.DataFrame) -> SeqSpec:
    """Read the available channels off a canonical event frame.

    `priority` and `reopen_count` are deliberately absent from the event axis:
    both are constant within virtually every case, so they carry no sequence
    information and belong with the statics.
    """
    ev_cat = [c for c in ("activity", "resource") if c in events.columns]
    ev_cat += [
        c for c in events.columns
        if c.startswith("dyn_cat__") and "priority" not in c
    ]
    ev_num = [
        c for c in events.columns
        if c.startswith("dyn_num__") and "reopen" not in c
    ]
    st_cat = [c for c in events.columns if c.startswith("static__")]
    return SeqSpec(tuple(ev_cat), tuple(ev_num), tuple(st_cat), ())


class PrefixTensors:
    """Builds padded (case, k_max, d) tensors once, then slices prefixes from them.

    A prefix of length k is the first k timesteps with a mask; there is no need
    to materialise a separate tensor per (case, k), which would multiply memory
    by k_max for no gain.
    """

    def __init__(self, events: pd.DataFrame, spec: SeqSpec, k_max: int):
        self.spec = spec
        self.k_max = k_max
        df = events[events["event_idx"] < k_max].sort_values(["case_id", "event_idx"])

        self.cat_vocabs = {c: Vocab.fit(df[c]) for c in spec.event_cat}
        firsts = df.groupby("case_id").first()
        self.static_vocabs = {c: Vocab.fit(firsts[c]) for c in spec.static_cat}

        self.case_ids = pd.Index(df["case_id"].unique())
        pos = pd.Series(np.arange(len(self.case_ids)), index=self.case_ids)
        row = pos.reindex(df["case_id"]).to_numpy()
        col = df["event_idx"].to_numpy()
        n = len(self.case_ids)

        self.cat = {
            c: self._scatter(row, col, self.cat_vocabs[c].encode(df[c]), n, np.int64)
            for c in spec.event_cat
        }

        gap = df.groupby("case_id")["event_ts"].diff().dt.total_seconds().fillna(0) / 3600.0
        elapsed = (df["event_ts"] - df["_case_start_ts"]).dt.total_seconds().clip(lower=0) / 3600.0
        num_cols = {
            **{c: df[c].astype(float).fillna(0.0).to_numpy() for c in spec.event_num},
            "gap_h": np.log1p(gap.clip(lower=0).to_numpy()),
            "elapsed_h": np.log1p(elapsed.to_numpy()),
        }
        self.num_names = list(num_cols)
        self.num = np.stack(
            [self._scatter(row, col, v, n, np.float32) for v in num_cols.values()], axis=-1
        )
        # Standardise on observed steps only; padding must not shift the mean.
        self.mask = self._scatter(row, col, np.ones(len(df)), n, np.float32)
        obs = self.mask.astype(bool)
        for j in range(self.num.shape[-1]):
            v = self.num[..., j][obs]
            mu, sd = float(v.mean()), float(v.std() or 1.0)
            self.num[..., j] = (self.num[..., j] - mu) / sd
        self.num[~obs] = 0.0

        self.static = {
            c: self.static_vocabs[c].encode(firsts.reindex(self.case_ids)[c])
            for c in spec.static_cat
        }
        self.row_of = pos

    @staticmethod
    def _scatter(row, col, values, n_rows, dtype):
        out = np.zeros((n_rows, int(col.max()) + 1), dtype=dtype)
        out[row, col] = values
        return out

    def batch(self, case_ids: np.ndarray, ks: np.ndarray) -> dict[str, torch.Tensor]:
        r = self.row_of.reindex(case_ids).to_numpy()
        steps = np.arange(self.k_max)[None, :]
        keep = (steps < ks[:, None]).astype(np.float32)
        mask = self.mask[r][:, : self.k_max] * keep
        return {
            "cat": {c: torch.from_numpy(v[r][:, : self.k_max]) for c, v in self.cat.items()},
            "num": torch.from_numpy(self.num[r][:, : self.k_max, :]),
            "mask": torch.from_numpy(mask),
            "static": {c: torch.from_numpy(v[r]) for c, v in self.static.items()},
        }


class PrefixGRU(nn.Module):
    """Embeddings -> GRU -> masked pool + last valid state -> static -> logit."""

    def __init__(self, tensors: PrefixTensors, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.event_emb = nn.ModuleDict(
            {c: nn.Embedding(v.size, v.embed_dim(), padding_idx=0)
             for c, v in tensors.cat_vocabs.items()}
        )
        self.static_emb = nn.ModuleDict(
            {c: nn.Embedding(v.size, v.embed_dim(), padding_idx=0)
             for c, v in tensors.static_vocabs.items()}
        )
        in_dim = sum(e.embedding_dim for e in self.event_emb.values()) + tensors.num.shape[-1]
        self.gru = nn.GRU(in_dim, hidden, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        head_in = hidden * 2 + sum(e.embedding_dim for e in self.static_emb.values())
        self.head = nn.Sequential(
            nn.Linear(head_in, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, b: dict) -> torch.Tensor:
        parts = [self.event_emb[c](v) for c, v in b["cat"].items()] + [b["num"]]
        h, _ = self.gru(torch.cat(parts, dim=-1))

        m = b["mask"].unsqueeze(-1)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)

        # Last *valid* step, not the last padded one.
        last_idx = (b["mask"].sum(1).long() - 1).clamp(min=0)
        last = h[torch.arange(h.size(0)), last_idx]

        feats = [self.drop(pooled), self.drop(last)]
        feats += [self.static_emb[c](v) for c, v in b["static"].items()]
        return self.head(torch.cat(feats, dim=-1)).squeeze(-1)
