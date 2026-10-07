# V1.3E-2 — Robust Validation Fix

This patch fixes the pandas boolean-index alignment warnings observed in V1.3E-1.

It also adds bootstrap confidence intervals for the **difference** between:
- FULL 903
- RESEARCH 297

Conservative "robust positive" criterion:
1. Research has higher win rate
2. Research has higher average return
3. 95% bootstrap CI for the mean-return difference is entirely above zero

This is still a diagnostic of universe selection, NOT a proof of strategy profitability and NOT PIT-safe backtest evidence.

No actual_engine.py changes.
