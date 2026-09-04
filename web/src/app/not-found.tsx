import Link from "next/link";
import { KairosMark } from "@/components/brand/KairosMark";

export const metadata = { title: "Page not found" };

export default function NotFound() {
  return (
    <main className="flex min-h-screen items-center justify-center px-6">
      <div className="max-w-md text-center">
        <div className="flex justify-center">
          <KairosMark size={40} />
        </div>
        <h1 className="display mt-6 text-4xl">Nothing here</h1>
        <p className="mt-4 leading-relaxed" style={{ color: "var(--text-2)" }}>
          That address does not point at anything. The study lives on a single page.
        </p>
        <Link
          href="/"
          className="mt-8 inline-block rounded-lg px-5 py-3 text-sm font-medium"
          style={{ background: "var(--accent)", color: "var(--accent-ink)" }}
        >
          Back to KAIROS
        </Link>
      </div>
    </main>
  );
}
