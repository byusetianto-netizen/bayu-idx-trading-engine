# V2.1A-3 — IDX Publication Evidence & PIT Validation

Purpose:
- Add explicit IDX publication timestamps for the three validation issuers:
  AADI, AALI, ABBA.
- Keep `period_end` separate from `publication_date`.
- Validate the core PIT rule:
  `publication_date <= analysis_date`.

Evidence:
- IDX disclosure page screenshot supplied by the user on 2026-10-08.
- AADI TW1 2026: 30 Apr 2026 16:51 WIB.
- AALI TW1 2026: 28 Apr 2026 16:30 WIB.
- ABBA TW1 2026: 28 Apr 2026 17:14 WIB.

Files:
- `data/fundamental/publication_evidence.csv`
- `scripts/apply_idx_publication_evidence.py`
- `scripts/validate_pit_publication_evidence.py`

Important:
- This patch does NOT change `actual_engine.py`.
- This patch does NOT create an investment score.
- This patch does NOT infer publication date from report date.
- Evidence is limited to the three issuers shown in the supplied IDX screenshot.
- Broader automation still needs a reliable historical IDX disclosure acquisition route.
