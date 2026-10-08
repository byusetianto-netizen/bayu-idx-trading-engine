# IDX Engine Fix v2

Fixes the GitHub Actions failure caused by `actual_engine.py` expecting the old IHSG columns `Price` and `Close` after `data_update.py` had normalized the file to `date` and `close`.

Also fixes the updater manifest so `new_last_date` is calculated from the updated file, not the pre-update in-memory dataframe.

Replace:
- `actual_engine.py`
- `data_update.py`

in the repository root, commit, and push.

Then rerun:
GitHub -> Actions -> Daily IDX Data Update -> Run workflow.
