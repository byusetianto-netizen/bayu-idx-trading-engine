# IDX Weekly Trading Engine — Streamlit

## Run locally
```bash
pip install -r requirements_streamlit.txt
streamlit run streamlit_app.py
```

The app is research/paper-only. Current bundled data ends 2026-07-01; run `data_update.py` in an internet-enabled environment to refresh.

## Deploy
Push this folder to GitHub, then create a Streamlit Community Cloud app pointing to `streamlit_app.py`.
