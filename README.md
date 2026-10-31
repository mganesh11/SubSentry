# SubSentry

**Track recurring payments. Spot hidden subscriptions. Stay in control.**

## About
SubSentry is a prototype that helps users identify recurring payments, understand subscription spending, and spot possible price increases.

## Features
- Upload transaction data using CSV.
- Detect potential recurring payments.
- Highlight possible price increases.
- Estimate annual subscription spending.
- Provide cancellation guidance.

## Technology
- Python
- Streamlit
- Pandas
- Plotly

## Run locally

Install dependencies:

```bash
py -m pip install -r requirements.txt
```

Start the application:

```bash
py -m streamlit run app.py
```

## Demo data
The included `sample_transactions.csv` file contains sample transaction data for testing.

## Limitations
This is a prototype. It does not connect to real bank accounts or automatically cancel subscriptions.
