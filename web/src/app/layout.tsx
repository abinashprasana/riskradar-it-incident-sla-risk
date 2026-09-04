import type { Metadata, Viewport } from "next";
import { Geist, Instrument_Serif, JetBrains_Mono } from "next/font/google";
import "./globals.css";

// Self-hosted at build time by next/font -- no render-blocking request to
// Google, no layout shift.
const geist = Geist({ variable: "--font-geist-sans", subsets: ["latin"], display: "swap" });

const instrument = Instrument_Serif({
  variable: "--font-instrument",
  subsets: ["latin"],
  weight: "400",
  display: "swap",
});

const jetbrains = JetBrains_Mono({
  variable: "--font-jetbrains",
  subsets: ["latin"],
  display: "swap",
});

const DESCRIPTION =
  "KAIROS scores a live IT incident queue by SLA-breach risk using only what is " +
  "known at the moment you look, and measures how early that call can be trusted. " +
  "Two ITSM event logs, prefix-based evaluation, 478 held-out incidents.";

export const metadata: Metadata = {
  metadataBase: new URL("https://kairos-study.vercel.app"),
  title: {
    default: "KAIROS — Incident SLA Breach Risk, Scored Early",
    template: "%s · KAIROS",
  },
  description: DESCRIPTION,
  keywords: [
    "predictive process monitoring",
    "SLA breach prediction",
    "prefix encoding",
    "earliness curve",
    "ITSM",
    "process mining",
  ],
  authors: [{ name: "Abinash Prasana Selvanathan" }],
  openGraph: {
    title: "KAIROS — Incident SLA Breach Risk, Scored Early",
    description: DESCRIPTION,
    type: "website",
    locale: "en_GB",
  },
  twitter: {
    card: "summary_large_image",
    title: "KAIROS — Incident SLA Breach Risk, Scored Early",
    description: DESCRIPTION,
  },
  robots: { index: true, follow: true },
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f7f9fb" },
    { media: "(prefers-color-scheme: dark)", color: "#0b1016" },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body
        className={`${geist.variable} ${instrument.variable} ${jetbrains.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
