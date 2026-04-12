# EventZilla ML Dashboard

## Deployment Options

### Option 1 — Streamlit Cloud (RECOMMENDED — Free & Online)
1. Push to GitHub (see below)
2. Go to https://share.streamlit.io
3. Connect your GitHub repo
4. Set main file: `app.py`
5. Deploy!

### Option 2 — Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Option 3 — Google Colab
```python
!pip install streamlit pyngrok -q
!wget -q https://your-repo/app.py
from pyngrok import ngrok
!streamlit run app.py &
public_url = ngrok.connect(8501)
print(public_url)
```

## Upload your CSV files
Place these files in the same folder as app.py (or /content/ for Colab):
- RESERVATION.csv
- EVALUATION.csv
- COMPLAINT.csv
- MARKETING_SPEND.csv
- VISITORS.csv
- benchmark_prix_marche_tunisie_latest.csv

## Pages
- **Overview** — KPIs + Revenue trend + Model badges
- **Cancellation Risk** — Real-time prediction + Feature impact
- **Revenue Forecast** — 6-month forecast + Confidence intervals
- **Customer Segmentation** — RFM analysis + Cluster profiling
- **Market Analysis** — Price positioning + Regression results
- **Model Performance** — Full A-F validation report
