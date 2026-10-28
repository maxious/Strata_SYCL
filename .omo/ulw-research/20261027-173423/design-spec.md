# Design spec - joint_matrix trellis GEMV report

STATUS: draft

Reader: the Strata SYCL maintainer deciding what to port for dsv41 decode.
Shape: one column, verdict card first, measured table, source list in brief order, implementation plan as an ordered list, limits last.
Voice: plain words, short sentences, numbers only from the fact ledger.
Typography: system sans, tabular numerals in tables, one accent (green) for the verdict, amber for the quoted root cause.
Surfaces: light and dark via prefers-color-scheme; no external assets.

Sections required: Verdict, What decides support, Measured kernel timings, The seven sources, Implementation plan, Limits.
