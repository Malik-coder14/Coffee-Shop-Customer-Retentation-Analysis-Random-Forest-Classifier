"""
365 Coffee Café — Customer Retention Analysis
Streamlit app converted from the supplied Jupyter notebook.

Run:
    pip install -r requirements.txt
    streamlit run app.py

The app automatically loads 365 Coffee Cafe Dataset.xlsx from the same folder.
Expected sheets: "Dataset" and optionally "Data Dictionary".
"""

from pathlib import Path
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

st.set_page_config(page_title="365 Coffee Café Retention", page_icon="☕", layout="wide")
st.title("☕ 365 Coffee Café — Customer Retention Analysis")
st.caption("Predicting whether a customer returns within 14 days")

st.markdown(
    """
    This dashboard follows the supplied notebook: it explores sales and visit patterns,
    converts transaction lines into one row per basket/visit, trains a Random Forest
    classifier, and reports test-set performance. Results are descriptive/predictive;
    they do not prove that a factor causes customer retention.
    """
)

REQUIRED_COLUMNS = [
    "transaction_date", "transaction_time", "hour_of_day", "weekday",
    "store_location", "customer_id", "customer_age", "customer_gender",
    "loyalty_tier", "visit_number", "days_since_previous_visit", "order_channel",
    "payment_method", "promo_code", "discount_pct", "wait_time_min",
    "service_rating", "returned_within_14_days", "basket_id", "transaction_id",
    "transaction_qty", "revenue", "discount_amount", "net_revenue", "unit_price",
    "product_id", "product_category", "product_type"
]
INVARIANT_COLS = [
    "transaction_date", "transaction_time", "hour_of_day", "weekday", "store_location",
    "customer_id", "customer_age", "customer_gender", "loyalty_tier", "visit_number",
    "days_since_previous_visit", "order_channel", "payment_method", "promo_code",
    "discount_pct", "wait_time_min", "service_rating", "returned_within_14_days"
]
NUMERIC_FEATURES = [
    "customer_age", "visit_number", "days_since_previous_visit", "is_first_recorded_visit",
    "hour_of_day", "discount_pct", "wait_time_min", "service_rating", "item_line_count",
    "total_quantity", "gross_revenue", "total_discount_amount", "net_revenue",
    "average_unit_price", "unique_products", "unique_categories", "unique_product_types"
]
CATEGORICAL_FEATURES = [
    "weekday", "store_location", "customer_gender", "loyalty_tier", "order_channel",
    "payment_method", "promo_code", "main_product_category", "visit_month"
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def make_visit_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate visit-level fields and aggregate transaction lines to basket-level rows."""
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError("The Dataset sheet is missing required columns: " + ", ".join(missing))

    data = raw.copy()
    data["transaction_date"] = pd.to_datetime(data["transaction_date"], errors="coerce")

    consistency = data.groupby("basket_id")[INVARIANT_COLS].nunique(dropna=False)
    inconsistent = int((consistency.max(axis=1) != 1).sum())
    if inconsistent:
        raise ValueError(
            f"{inconsistent} basket(s) have inconsistent visit-level fields. "
            "Please inspect the source workbook before modelling; the notebook checks this rather than silently choosing values."
        )

    visit = data.groupby("basket_id", sort=False).agg(
        transaction_date=("transaction_date", "first"),
        transaction_time=("transaction_time", "first"),
        hour_of_day=("hour_of_day", "first"),
        weekday=("weekday", "first"),
        store_location=("store_location", "first"),
        customer_id=("customer_id", "first"),
        customer_age=("customer_age", "first"),
        customer_gender=("customer_gender", "first"),
        loyalty_tier=("loyalty_tier", "first"),
        visit_number=("visit_number", "first"),
        days_since_previous_visit=("days_since_previous_visit", "first"),
        order_channel=("order_channel", "first"),
        payment_method=("payment_method", "first"),
        promo_code=("promo_code", "first"),
        discount_pct=("discount_pct", "first"),
        wait_time_min=("wait_time_min", "first"),
        service_rating=("service_rating", "first"),
        returned_within_14_days=("returned_within_14_days", "first"),
        item_line_count=("transaction_id", "count"),
        total_quantity=("transaction_qty", "sum"),
        gross_revenue=("revenue", "sum"),
        total_discount_amount=("discount_amount", "sum"),
        net_revenue=("net_revenue", "sum"),
        average_unit_price=("unit_price", "mean"),
        unique_products=("product_id", "nunique"),
        unique_categories=("product_category", "nunique"),
        unique_product_types=("product_type", "nunique"),
        main_product_category=("product_category", "first"),
    ).reset_index()

    visit["is_first_recorded_visit"] = (visit["days_since_previous_visit"] == -1).astype(int)
    visit["visit_month"] = visit["transaction_date"].dt.month_name()
    target_map = {"No": 0, "Yes": 1}
    visit["target"] = visit["returned_within_14_days"].map(target_map)
    if visit["target"].isna().any():
        bad_labels = visit.loc[visit["target"].isna(), "returned_within_14_days"].value_counts(dropna=False)
        raise ValueError(
            "The target column must contain only 'Yes' and 'No'. Unexpected labels: "
            + str(bad_labels.to_dict())
        )
    visit["target"] = visit["target"].astype(int)
    return visit


def train_retention_model(visit: pd.DataFrame):
    X = visit[FEATURES].copy()
    y = visit["target"].copy()
    if y.nunique() < 2 or y.value_counts().min() < 2:
        raise ValueError("At least two examples of each target class are needed for a stratified train/test split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), CATEGORICAL_FEATURES)
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=20, random_state=42, n_jobs=1, min_samples_leaf=5, max_depth=8
        ))
    ])

    # Preserve the notebook's capped training sample for reasonable runtime.
    X_fit = X_train.sample(n=min(3000, len(X_train)), random_state=42)
    y_fit = y_train.loc[X_fit.index]
    model.fit(X_fit, y_fit)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    scores = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
        "Test score": [
            accuracy_score(y_test, pred),
            precision_score(y_test, pred, zero_division=0),
            recall_score(y_test, pred, zero_division=0),
            f1_score(y_test, pred, zero_division=0),
            roc_auc_score(y_test, proba),
        ],
    })
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    baseline_accuracy = accuracy_score(y_test, baseline.predict(X_test))

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    encoded_importance = model.named_steps["classifier"].feature_importances_
    importance_rows = []
    for name, value in zip(feature_names, encoded_importance):
        short = name.split("__", 1)[-1]
        original = next((f for f in FEATURES if short == f or short.startswith(f + "_")), short)
        importance_rows.append((original, value))
    importance = (
        pd.DataFrame(importance_rows, columns=["feature", "importance"])
        .groupby("feature", as_index=False)["importance"].sum()
        .sort_values("importance", ascending=False)
    )
    return model, X_test, y_test, pred, proba, scores, baseline_accuracy, importance, len(X_train), len(X_fit)


# Offer both data-loading options:
# 1) Use the project's default Excel workbook from the same folder as app.py.
# 2) Upload a different Excel workbook from the user's computer.
DEFAULT_DATA_FILE = Path(__file__).resolve().parent / "365 Coffee Cafe Dataset.xlsx"

st.subheader("Choose your data source")
data_source = st.radio(
    "How would you like to load data?",
    options=["Use project dataset automatically", "Upload my own Excel file"],
    horizontal=True,
)

raw = None
dictionary = None
source_name = None

if data_source == "Use project dataset automatically":
    if not DEFAULT_DATA_FILE.exists():
        st.error(
            "The default dataset was not found. Place **365 Coffee Cafe Dataset.xlsx** "
            "in the same folder as app.py and include it in your GitHub repository, "
            "or choose **Upload my own Excel file** above."
        )
        st.stop()
    try:
        raw = pd.read_excel(DEFAULT_DATA_FILE, sheet_name="Dataset")
        source_name = DEFAULT_DATA_FILE.name
        try:
            dictionary = pd.read_excel(DEFAULT_DATA_FILE, sheet_name="Data Dictionary")
        except Exception:
            dictionary = None
    except Exception as exc:
        st.error(f"Could not read the 'Dataset' sheet from {DEFAULT_DATA_FILE.name}: {exc}")
        st.stop()
else:
    uploaded = st.file_uploader(
        "Upload an Excel workbook",
        type=["xlsx", "xls"],
        help="The workbook must contain a sheet named 'Dataset'. A 'Data Dictionary' sheet is optional.",
    )
    if uploaded is None:
        st.info("Upload your Excel workbook to continue.")
        st.stop()
    try:
        raw = pd.read_excel(uploaded, sheet_name="Dataset")
        source_name = uploaded.name
        try:
            dictionary = pd.read_excel(uploaded, sheet_name="Data Dictionary")
        except Exception:
            dictionary = None
    except Exception as exc:
        st.error(f"Could not read the 'Dataset' sheet from the uploaded workbook: {exc}")
        st.stop()

st.success(f"Workbook loaded successfully: {source_name}")
with st.expander("Data Dictionary (optional)", expanded=False):
    if dictionary is not None:
        st.dataframe(dictionary, use_container_width=True)
    else:
        st.caption("No readable 'Data Dictionary' sheet was found.")

try:
    visit = make_visit_table(raw)
except Exception as exc:
    st.error(str(exc))
    st.stop()

# Sidebar filters for descriptive analysis
st.sidebar.header("Dashboard controls")
branches = sorted(raw["store_location"].dropna().astype(str).unique().tolist())
selected_branches = st.sidebar.multiselect("Branches", branches, default=branches)
filtered_raw = raw[raw["store_location"].astype(str).isin(selected_branches)].copy()
filtered_visit = visit[visit["store_location"].astype(str).isin(selected_branches)].copy()

if filtered_visit.empty:
    st.warning("Select at least one branch to display results.")
    st.stop()

overview, sales_tab, visits_tab, model_tab, segments_tab, data_tab = st.tabs([
    "Overview", "Sales analysis", "Visit patterns", "Retention model", "Segments & importance", "Data quality"
])

with overview:
    st.subheader("Business overview")
    total_visits = filtered_visit["basket_id"].nunique()
    return_rate = filtered_visit["target"].mean() * 100
    revenue = filtered_raw["net_revenue"].sum()
    avg_wait = filtered_visit["wait_time_min"].mean()
    avg_rating = filtered_visit["service_rating"].mean()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Visits / baskets", f"{total_visits:,}")
    c2.metric("Return within 14 days", f"{return_rate:.1f}%")
    c3.metric("Net revenue", f"{revenue:,.2f}")
    c4.metric("Average wait", f"{avg_wait:.2f} min" if pd.notna(avg_wait) else "N/A")
    c5.metric("Average service rating", f"{avg_rating:.2f}" if pd.notna(avg_rating) else "N/A")
    st.markdown("**Important:** branch revenue is not profit; cost and margin data are not supplied.")
    st.write("Date range:", str(filtered_raw["transaction_date"].min())[:10], "to", str(filtered_raw["transaction_date"].max())[:10])

with sales_tab:
    st.subheader("Revenue by branch")
    branch_summary = filtered_raw.groupby("store_location").agg(
        net_revenue=("net_revenue", "sum"),
        transaction_lines=("transaction_id", "count")
    ).sort_values("net_revenue", ascending=False)
    st.dataframe(branch_summary.style.format({"net_revenue": "{:,.2f}"}), use_container_width=True)
    st.bar_chart(branch_summary["net_revenue"])
    st.subheader("Sales by product category")
    category_summary = filtered_raw.groupby("product_category").agg(
        net_revenue=("net_revenue", "sum"),
        quantity=("transaction_qty", "sum")
    ).sort_values("net_revenue", ascending=False)
    st.dataframe(category_summary.style.format({"net_revenue": "{:,.2f}", "quantity": "{:,.0f}"}), use_container_width=True)
    st.bar_chart(category_summary["net_revenue"])
    st.caption("Revenue contribution does not establish profit or product margin.")

with visits_tab:
    st.subheader("Visits by hour")
    by_hour = filtered_visit.groupby("hour_of_day").agg(
        visits=("target", "size"), return_rate=("target", "mean")
    ).sort_index()
    left, right = st.columns(2)
    with left:
        st.bar_chart(by_hour["visits"])
        st.dataframe(by_hour.assign(return_rate_pct=by_hour["return_rate"] * 100), use_container_width=True)
    st.subheader("Visits by weekday")
    by_weekday = filtered_visit.groupby("weekday").agg(
        visits=("target", "size"), return_rate=("target", "mean")
    )
    with right:
        st.bar_chart(by_weekday["visits"])
    st.dataframe(by_weekday.assign(return_rate_pct=by_weekday["return_rate"] * 100), use_container_width=True)
    st.caption("These patterns can inform staffing hypotheses, but roster data would be needed to test staffing effects.")

with model_tab:
    st.subheader("Random Forest evaluation")
    st.write("The model uses an 80/20 stratified split and fits on up to 3,000 training visits, matching the notebook.")
    if st.button("Train / evaluate model", type="primary"):
        with st.spinner("Training the model..."):
            try:
                result = train_retention_model(visit)
                model, X_test, y_test, pred, proba, scores, baseline_accuracy, importance, n_train, n_fit = result
                st.session_state["model_result"] = result
                st.session_state["trained_model"] = model
                st.session_state["visit_data"] = visit
            except Exception as exc:
                st.error(f"Model could not be trained: {exc}")

    if "model_result" in st.session_state:
        result = st.session_state["model_result"]
        model, X_test, y_test, pred, proba, scores, baseline_accuracy, importance, n_train, n_fit = result
        m1, m2, m3 = st.columns(3)
        m1.metric("Training partition", f"{n_train:,}")
        m2.metric("Model-fit sample", f"{n_fit:,}")
        m3.metric("Majority-class baseline accuracy", f"{baseline_accuracy:.3f}")
        st.dataframe(scores.style.format({"Test score": "{:.3f}"}), use_container_width=True)
        st.metric("Accuracy improvement over baseline", f"{scores.loc[scores['Metric']=='Accuracy', 'Test score'].iloc[0] - baseline_accuracy:+.3f}")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(y_test, pred),
            display_labels=["No return", "Return"]
        ).plot(values_format=",d", colorbar=False, ax=ax)
        ax.set_title("Test-set confusion matrix")
        st.pyplot(fig)
        plt.close(fig)
        report = classification_report(y_test, pred, target_names=["No return", "Return"], digits=3, output_dict=True, zero_division=0)
        st.subheader("Classification report")
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
        model_bytes = io.BytesIO()
        joblib.dump({
            "model": model, "features": FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES
        }, model_bytes)
        st.download_button(
            "Download trained model (.joblib)",
            data=model_bytes.getvalue(),
            file_name="365_Coffee_Cafe_Retention_Model.joblib",
            mime="application/octet-stream"
        )
    else:
        st.info("Click **Train / evaluate model** to fit the model and view metrics.")

with segments_tab:
    st.subheader("Return rate by branch")
    branch_results = filtered_visit.groupby("store_location").agg(
        visits=("basket_id", "count"), return_rate=("target", "mean"),
        avg_wait_min=("wait_time_min", "mean"), avg_rating=("service_rating", "mean")
    ).reset_index().sort_values("return_rate", ascending=False)
    st.dataframe(branch_results.style.format({
        "return_rate": "{:.1%}", "avg_wait_min": "{:.2f}", "avg_rating": "{:.2f}"
    }), use_container_width=True)

    st.subheader("Return rate by order channel")
    channel_results = filtered_visit.groupby("order_channel").agg(
        visits=("basket_id", "count"), return_rate=("target", "mean"),
        avg_wait_min=("wait_time_min", "mean"), avg_rating=("service_rating", "mean")
    ).reset_index().sort_values("return_rate", ascending=False)
    st.dataframe(channel_results.style.format({
        "return_rate": "{:.1%}", "avg_wait_min": "{:.2f}", "avg_rating": "{:.2f}"
    }), use_container_width=True)

    st.subheader("Return rate by loyalty tier")
    loyalty_results = filtered_visit.groupby("loyalty_tier").agg(
        visits=("basket_id", "count"), return_rate=("target", "mean")
    ).reset_index().sort_values("return_rate", ascending=False)
    st.dataframe(loyalty_results.style.format({"return_rate": "{:.1%}"}), use_container_width=True)

    st.subheader("Feature importance")
    if "model_result" not in st.session_state:
        st.info("Train the model in the Retention model tab to calculate feature importance.")
    else:
        importance = st.session_state["model_result"][7].head(12).sort_values("importance")
        st.bar_chart(importance.set_index("feature")["importance"])
        st.dataframe(importance, use_container_width=True)
        st.caption("Random Forest feature importance is an association/ranking measure, not evidence of causation.")

with data_tab:
    st.subheader("Source data quality")
    c1, c2, c3 = st.columns(3)
    c1.metric("Transaction lines", f"{len(raw):,}")
    c2.metric("Unique baskets", f"{raw['basket_id'].nunique():,}")
    c3.metric("Exact duplicate rows", f"{int(raw.duplicated().sum()):,}")
    st.write("Missing values by column")
    st.dataframe(raw.isna().sum().rename("missing_count").to_frame(), use_container_width=True)
    st.write("Duplicate transaction_id values:", int(raw["transaction_id"].duplicated().sum()))
    st.write("Baskets with inconsistent return target:", int((raw.groupby("basket_id")["returned_within_14_days"].nunique(dropna=False) != 1).sum()))
    st.subheader("Visit-level data preview")
    st.dataframe(filtered_visit.head(100), use_container_width=True)
    st.download_button(
        "Download visit-level dataset (.csv)",
        data=filtered_visit.to_csv(index=False).encode("utf-8"),
        file_name="365_Coffee_Cafe_Visit_Level_Dataset.csv",
        mime="text/csv"
    )

st.divider()
st.caption(
    "Limitations from the notebook: observed return patterns are not causal; branch revenue is not profit "
    "without cost data; staffing effects require roster data; and model performance should be monitored "
    "before using predictions for customer-level decisions."
)
