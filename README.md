# 365 Coffee Café — Customer Retention Analysis

A Streamlit dashboard for exploring café sales and visit patterns and evaluating a Random Forest model that predicts whether a customer returns within 14 days.

## Files

- `app.py` — Streamlit application.
- `requirements.txt` — Python dependencies.
- `365 Coffee Cafe Dataset.xlsx` — source workbook required by the app.

The Excel workbook must contain a sheet named `Dataset`. A `Data Dictionary` sheet is optional. The app offers two choices: load the default workbook automatically from the project folder, or upload another workbook from the dashboard.

## Run locally

1. Install Python.
2. Open a terminal in this folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

## Deploy to Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, and `365 Coffee Cafe Dataset.xlsx` to your GitHub repository if you want the default automatic-loading option available. `README.md` is recommended.
2. Open https://share.streamlit.io/ and connect the repository.
3. Set the main file path to `app.py`.
4. Deploy.

## Data privacy

If the repository is public, the Excel workbook will also be publicly accessible. Only publish the dataset if you have permission and it contains no confidential or personal information.
