/**
 * Types for the exported study data.
 *
 * Every field here originates in `artifacts/`, written by
 * `scripts/export_web_data.py`. Nothing on the site is hand-entered, so a
 * pipeline re-run propagates automatically and the page cannot drift from what
 * the experiments actually reported.
 */

export type Cohort = "fixed" | "variable";

export type PerK = {
  cohort: Cohort;
  k: number;
  n_k: number;
  base_rate_k: number;
  auc: number;
  /** 1000-resample bootstrap interval on `auc`. Absent on older exports. */
  auc_lo?: number;
  auc_hi?: number;
  ap: number;
  brier: number;
  ece?: number;
  bias?: number;
  precision: number;
  recall: number;
  f1: number;
};

export type ArmId =
  | "uci_headline"
  | "uci_randomsplit"
  | "uci_agg_logreg"
  | "uci_index"
  | "bpi_headline"
  | "bpi_agg_logreg"
  | "bpi_index";

export type PerKData = Record<ArmId, PerK[]>;

export type Summary = {
  leaky: { auc: number; accuracy: number; defects: string[] };
  uci: {
    selected_model: string;
    k_max: number;
    case_level_k1_auc: number;
    fixed_k1_auc: number;
    fixed_kmax_auc: number;
    prefix_weighted_auc: number;
    n_cases_total: number;
    n_cases_dense_window: number;
    n_cases_dropped_all_terminal: number;
    breach_rate_dropped_cases: number;
    breach_rate_kept_cases: number;
    split: {
      n_train_cases: number;
      n_calib_cases: number;
      n_test_cases: number;
      test_boundary: string;
      calib_mode: string;
      n_dropped_straddling: { modelling_period: number };
    };
    split_mode: string;
    base_rate_by_fold: Record<string, number>;
  };
  uci_randomsplit: { fixed_k1_auc: number; fixed_kmax_auc: number };
  bpi: {
    selected_model: string;
    k_max: number;
    fixed_k1_auc: number;
    fixed_kmax_auc: number;
    label: { deadline_hours: number; source: string; n_cases_censored_dropped: number };
    split_mode: string;
    already_past_deadline: number;
  };
  encoding_control: {
    uci_agg_logreg_kmax: number;
    uci_index_kmax: number;
    bpi_agg_logreg_kmax: number;
    bpi_index_kmax: number;
  };
  backfill: { column: string; populated_at_k1: number; equals_final: number }[];
  drift: { month: string; n: number; breach_rate: number }[];
};

export type RiskBand = "low" | "medium" | "high";

export type ExplorerSample = {
  k_max: number;
  cases: Record<string, { y: 0 | 1; p: number[]; band: RiskBand[] }>;
};

/** Fixed series slots. Assigned by entity, never by rank, never cycled. */
export const SERIES = {
  fixedCohort: "var(--series-1)",
  variableCohort: "var(--series-2)",
  randomSplit: "var(--series-3)",
  bpi: "var(--series-4)",
  indexEncoding: "var(--series-5)",
  leaky: "var(--series-6)",
} as const;

/**
 * Secondary encoding, required because tritan separation on this palette is
 * 6.4 -- inside the 6-8 floor band, which is legal only when colour is not the
 * sole carrier of identity. Each series therefore also owns a dash pattern.
 */
export const SERIES_DASH = {
  fixedCohort: undefined,
  variableCohort: "6 4",
  randomSplit: "2 3",
  bpi: undefined,
  indexEncoding: "8 3 2 3",
  leaky: "4 4",
} as const;

export const BAND_COLOR: Record<RiskBand, string> = {
  low: "var(--status-good)",
  medium: "var(--status-warn)",
  high: "var(--status-critical)",
};

export const fmt = {
  auc: (v: number) => v.toFixed(3),
  pct: (v: number) => `${(v * 100).toFixed(1)}%`,
  int: (v: number) => v.toLocaleString("en-GB"),
};
