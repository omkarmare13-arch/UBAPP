import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


st.set_page_config(
    page_title="Personal Loan Propensity Dashboard",
    layout="wide"
)


# ---------------- Theming / Styling ----------------
def inject_global_css():
    st.markdown(
        """<style>
        /* Make background a bit darker and cards cleaner */
        .main {
            background: radial-gradient(circle at top left, #1f2933, #020617);
            color: #e5e7eb;
        }
        [data-testid="stMetric"] {
            background-color: #020617;
            padding: 12px 16px;
            border-radius: 0.75rem;
            border: 1px solid rgba(148,163,184,0.4);
        }
        .kpi-card {
            padding: 1rem 1.25rem;
            border-radius: 0.75rem;
            border: 1px solid rgba(148,163,184,0.4);
            background: rgba(15,23,42,0.8);
        }
        .section-title {
            font-size: 1.15rem;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }
        .section-subtitle {
            font-size: 0.85rem;
            color: #9ca3af;
            margin-bottom: 0.75rem;
        }
        </style>""", unsafe_allow_html=True
    )


# ---------------- Utility functions ----------------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().replace(" ", "") for c in df.columns]
    return df


def pick_column(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_base_data():
    """Load UniversalBank.csv if available; otherwise show a warning and return None."""
    try:
        df = pd.read_csv("UniversalBank.csv")
        df = standardize_columns(df)
        return df
    except FileNotFoundError:
        st.warning(
            "Base dataset 'UniversalBank.csv' was not found in the app folder. "
            "Please make sure the file is in the same directory as app.py in your GitHub repo."
        )
        return None


def prepare_features(df: pd.DataFrame):
    df = standardize_columns(df)

    col_map = {}
    col_map["ID"] = pick_column(df, ["ID"])
    col_map["PersonalLoan"] = pick_column(df, ["PersonalLoan"])
    col_map["Age"] = pick_column(df, ["Age"])
    col_map["Experience"] = pick_column(df, ["Experience"])
    col_map["Income"] = pick_column(df, ["Income"])
    col_map["Zipcode"] = pick_column(df, ["ZIPCode", "Zipcode", "ZIP"])
    col_map["Family"] = pick_column(df, ["Family"])
    col_map["CCAvg"] = pick_column(df, ["CCAvg"])
    col_map["Education"] = pick_column(df, ["Education"])
    col_map["Mortgage"] = pick_column(df, ["Mortgage"])
    col_map["Securities"] = pick_column(df, ["SecuritiesAccount", "Securities"])
    col_map["CDAccount"] = pick_column(df, ["CDAccount"])
    col_map["Online"] = pick_column(df, ["Online"])
    col_map["CreditCard"] = pick_column(df, ["CreditCard"])

    target_col = col_map["PersonalLoan"]
    if target_col is None:
        raise ValueError("Could not find 'PersonalLoan' column in data.")

    drop_cols = [c for c in [col_map["ID"]] if c is not None]

    feature_cols = [
        col_map["Age"],
        col_map["Experience"],
        col_map["Income"],
        col_map["Zipcode"],
        col_map["Family"],
        col_map["CCAvg"],
        col_map["Education"],
        col_map["Mortgage"],
        col_map["Securities"],
        col_map["CDAccount"],
        col_map["Online"],
        col_map["CreditCard"],
    ]
    feature_cols = [c for c in feature_cols if c is not None]

    model_df = df.drop(columns=drop_cols, errors="ignore")
    needed = [c for c in feature_cols + [target_col] if c is not None]
    model_df = model_df[needed].dropna()

    X = model_df[feature_cols]
    y = model_df[target_col].astype(int)

    return X, y, feature_cols, col_map, target_col, model_df


def train_and_evaluate_models(X, y, test_size=0.30, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=42, max_depth=6),
        "RandomForest": RandomForestClassifier(
            random_state=42, n_estimators=150, max_depth=None, n_jobs=1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=42, n_estimators=100
        ),
    }

    metric_rows = []
    roc_curves = {}
    confusion_matrices = {}
    feature_importances = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        y_tr = model.predict(X_train)
        y_te = model.predict(X_test)
        proba_tr = model.predict_proba(X_train)[:, 1]
        proba_te = model.predict_proba(X_test)[:, 1]

        row = {
            "Algorithm": name,
            "Training Accuracy": accuracy_score(y_train, y_tr),
            "Testing Accuracy": accuracy_score(y_test, y_te),
            "Precision": precision_score(y_test, y_te, zero_division=0),
            "Recall": recall_score(y_test, y_te, zero_division=0),
            "F1-Score": f1_score(y_test, y_te, zero_division=0),
            "AUC (Train)": roc_auc_score(y_train, proba_tr),
            "AUC (Test)": roc_auc_score(y_test, proba_te),
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_auc = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1
        )
        row["CV5 AUC Mean (Train)"] = cv_auc.mean()
        row["CV5 AUC Std (Train)"] = cv_auc.std()

        metric_rows.append(row)

        fpr, tpr, _ = roc_curve(y_test, proba_te)
        roc_curves[name] = (fpr, tpr, roc_auc_score(y_test, proba_te))

        confusion_matrices[name] = {
            "train": confusion_matrix(y_train, y_tr),
            "test": confusion_matrix(y_test, y_te),
        }

        feature_importances[name] = model.feature_importances_

    metrics_df = pd.DataFrame(metric_rows).set_index("Algorithm").round(4)

    return (
        metrics_df,
        models,
        roc_curves,
        confusion_matrices,
        feature_importances,
        (X_train, y_train, X_test, y_test),
    )


# ---------------- Plotting helpers ----------------
def plot_roc_curves(roc_curves, selected=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    if selected is None:
        selected = list(roc_curves.keys())
    for name in selected:
        fpr, tpr, auc_val = roc_curves[name]
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (Test Set)")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_confusion_matrices(conf_matrices, focus=None):
    algorithms = list(conf_matrices.keys())
    if focus is not None and focus in algorithms:
        algorithms = [focus]

    splits = ["train", "test"]
    fig, axes = plt.subplots(
        nrows=len(algorithms), ncols=len(splits), figsize=(10, 4 * len(algorithms))
    )

    if len(algorithms) == 1:
        axes = np.array([axes])

    for i, algo in enumerate(algorithms):
        for j, split in enumerate(splits):
            cm = conf_matrices[algo][split]
            ax = axes[i, j]
            ax.imshow(cm, cmap="Blues")
            ax.set_title(f"{algo} - {split.capitalize()}")
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["0", "1"])
            ax.set_yticklabels(["0", "1"])

            thresh = cm.max() / 2.0
            for r in range(cm.shape[0]):
                for c in range(cm.shape[1]):
                    ax.text(
                        c,
                        r,
                        int(cm[r, c]),
                        ha="center",
                        va="center",
                        color="white" if cm[r, c] > thresh else "black",
                    )
    fig.tight_layout()
    return fig


def plot_feature_importances(feature_importances, feature_names):
    algos = list(feature_importances.keys())
    fig, axes = plt.subplots(
        nrows=1, ncols=len(algos), figsize=(5 * len(algos), 4), sharey=True
    )

    if len(algos) == 1:
        axes = [axes]

    for ax, algo in zip(axes, algos):
        importances = feature_importances[algo]
        idx = np.argsort(importances)[::-1]
        ordered_features = [feature_names[i] for i in idx]
        ordered_importances = importances[idx]
        ax.bar(range(len(ordered_features)), ordered_importances)
        ax.set_xticks(range(len(ordered_features)))
        ax.set_xticklabels(ordered_features, rotation=45, ha="right")
        ax.set_title(algo)
        ax.set_ylabel("Importance")
    fig.suptitle("Feature Importances by Algorithm", y=1.02)
    fig.tight_layout()
    return fig


# ---------------- EDA charts ----------------
def chart_conversion_by_income_education(df, col_map):
    df = df.copy()
    income_col = col_map["Income"]
    edu_col = col_map["Education"]
    loan_col = col_map["PersonalLoan"]
    df["IncomeQuintile"] = pd.qcut(
        df[income_col], 5,
        labels=["Q1 Lowest", "Q2", "Q3", "Q4", "Q5 Highest"]
    )
    conv = (
        df.groupby(["IncomeQuintile", edu_col])[loan_col]
        .mean()
        .reset_index()
        .rename(columns={loan_col: "ConversionRate"})
    )
    pivot = conv.pivot(index="IncomeQuintile", columns=edu_col, values="ConversionRate")
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Loan Acceptance Rate")
    ax.set_title("Conversion by Income Quintile and Education")
    ax.legend(title="Education (1=UG,2=Grad,3=Adv)", loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def chart_conversion_by_ccavg_cd(df, col_map):
    df = df.copy()
    cc_col = col_map["CCAvg"]
    cd_col = col_map["CDAccount"]
    loan_col = col_map["PersonalLoan"]
    df["CCBucket"] = pd.qcut(
        df[cc_col], 4,
        labels=["Low", "Medium", "High", "Very High"]
    )
    conv = (
        df.groupby(["CCBucket", cd_col])[loan_col]
        .mean()
        .reset_index()
        .rename(columns={loan_col: "ConversionRate"})
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    for cd_val in sorted(conv[cd_col].unique()):
        subset = conv[conv[cd_col] == cd_val]
        ax.plot(
            subset["CCBucket"],
            subset["ConversionRate"],
            marker="o",
            label=f"CD Account={cd_val}",
        )
    ax.set_xlabel("Credit Card Spend Bucket")
    ax.set_ylabel("Loan Acceptance Rate")
    ax.set_title("Conversion by Credit Card Spend & CD Account")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def chart_heatmap_age_family(df, col_map):
    df = df.copy()
    age_col = col_map["Age"]
    fam_col = col_map["Family"]
    loan_col = col_map["PersonalLoan"]
    df["AgeBand"] = pd.cut(
        df[age_col],
        bins=[20, 30, 40, 50, 60, 70],
        labels=["20-29", "30-39", "40-49", "50-59", "60-69"],
    )
    conv = (
        df.groupby(["AgeBand", fam_col])[loan_col]
        .mean()
        .reset_index()
        .rename(columns={loan_col: "ConversionRate"})
    )
    pivot = conv.pivot(index="AgeBand", columns=fam_col, values="ConversionRate")
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Family Size")
    ax.set_ylabel("Age Band")
    ax.set_title("Conversion Heatmap: Age Band vs Family Size")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax, label="Loan Acceptance Rate")
    fig.tight_layout()
    return fig


def chart_online_creditcard_segments(df, col_map):
    df = df.copy()
    online_col = col_map["Online"]
    cc_col = col_map["CreditCard"]
    loan_col = col_map["PersonalLoan"]
    df["Segment"] = "Offline / No Card"
    df.loc[(df[online_col] == 1) & (df[cc_col] == 0), "Segment"] = "Online / No Card"
    df.loc[(df[online_col] == 0) & (df[cc_col] == 1), "Segment"] = "Offline / Card"
    df.loc[(df[online_col] == 1) & (df[cc_col] == 1), "Segment"] = "Online / Card"

    agg = (
        df.groupby("Segment")[loan_col]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "ConversionRate", "count": "Customers"})
    )
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()

    ax1.bar(agg["Segment"], agg["ConversionRate"], alpha=0.7, label="Conversion Rate")
    ax2.plot(
        agg["Segment"],
        agg["Customers"],
        marker="o",
        color="tab:red",
        label="Customer Count",
    )
    ax1.set_ylabel("Loan Acceptance Rate")
    ax2.set_ylabel("Number of Customers")
    ax1.set_title("Digital & Card Segments: Conversion & Volume")
    ax1.set_xticklabels(agg["Segment"], rotation=20, ha="right")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    return fig


def chart_feature_corr_with_target(model_df, col_map, feature_cols):
    loan_col = col_map["PersonalLoan"]
    numeric_df = model_df[feature_cols + [loan_col]].select_dtypes(include=[np.number])
    corr = numeric_df.corr()[loan_col].drop(loan_col)
    corr = corr.reindex(feature_cols)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(corr)), corr.values)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.index, rotation=45, ha="right")
    ax.set_ylabel("Correlation with Loan Acceptance")
    ax.set_title("Feature Correlation with Personal Loan")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------- MAIN APP LAYOUT ----------------
inject_global_css()

st.markdown(
    "<h1 style='margin-bottom:0.25rem;'>📊 Personal Loan Propensity & Customer Insight Dashboard</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#9ca3af; margin-bottom:1.5rem;'>Designed for Heads of Marketing to discover high-conversion customer segments and compare ML models for personal loan campaigns.</p>",
    unsafe_allow_html=True,
)

base_df = load_base_data()

# Initialise placeholders in case dataset is missing
X = y = None
feature_cols = []
col_map = {}
target_col = None
model_df = None

if base_df is not None:
    X, y, feature_cols, col_map, target_col, model_df = prepare_features(base_df)

# KPI row
if model_df is not None and target_col is not None:
    total_customers = len(model_df)
    loan_rate = model_df[target_col].mean() if total_customers > 0 else 0
    income_col = col_map.get("Income")
    avg_income = model_df[income_col].mean() if income_col is not None else 0
    cc_col = col_map.get("CCAvg")
    avg_cc = model_df[cc_col].mean() if cc_col is not None else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Customers", f"{total_customers:,}")
    with k2:
        st.metric("Loan Acceptance Rate", f"{loan_rate*100:.1f}%")
    with k3:
        st.metric("Average Income ($000)", f"{avg_income:.1f}")
    with k4:
        st.metric("Avg CC Spend ($000)", f"{avg_cc:.1f}")
else:
    st.info("Upload or place 'UniversalBank.csv' next to app.py to enable full insights and modeling.")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(
    [
        "1️⃣ Customer Insights",
        "2️⃣ Model Performance (DT / RF / GBT)",
        "3️⃣ Score New Customer File",
    ]
)

# ---- TAB 1: Customer Insights ----
with tab1:
    st.markdown("<div class='section-title'>Customer Insights for Better Marketing Actions</div>", unsafe_allow_html=True)
    if model_df is None or not feature_cols:
        st.error("Base dataset is missing. Please ensure 'UniversalBank.csv' is in the same folder as app.py in your GitHub repo.")
    else:
        st.markdown(
            "<div class='section-subtitle'>Explore how income, education, digital behaviour and product holding impact personal loan uptake.</div>",
            unsafe_allow_html=True,
        )

        insight_choice = st.radio(
            "Choose insight focus",
            [
                "Income & Education",
                "Credit Card Spend & CD Account",
                "Age & Family Heatmap",
                "Digital & Card Segments",
                "Feature–Target Correlation",
                "Show All",
            ],
            horizontal=True,
        )

        col_a, col_b = st.columns(2)

        if insight_choice in ["Income & Education", "Show All"]:
            with col_a:
                st.markdown("#### 1. Conversion by Income Quintile & Education")
                fig1 = chart_conversion_by_income_education(model_df, col_map)
                st.pyplot(fig1)

        if insight_choice in ["Credit Card Spend & CD Account", "Show All"]:
            with col_b:
                st.markdown("#### 2. Conversion by Credit Card Spend & CD Account")
                fig2 = chart_conversion_by_ccavg_cd(model_df, col_map)
                st.pyplot(fig2)

        col_c, col_d = st.columns(2)

        if insight_choice in ["Age & Family Heatmap", "Show All"]:
            with col_c:
                st.markdown("#### 3. Conversion Heatmap: Age Band vs Family Size")
                fig3 = chart_heatmap_age_family(model_df, col_map)
                st.pyplot(fig3)

        if insight_choice in ["Digital & Card Segments", "Show All"]:
            with col_d:
                st.markdown("#### 4. Digital & Card Segments: Conversion & Volume")
                fig4 = chart_online_creditcard_segments(model_df, col_map)
                st.pyplot(fig4)

        if insight_choice in ["Feature–Target Correlation", "Show All"]:
            st.markdown("#### 5. Correlation of Features with Personal Loan")
            fig5 = chart_feature_corr_with_target(model_df, col_map, feature_cols)
            st.pyplot(fig5)

        with st.expander("💡 How can I use these insights for campaigns?"):
            st.markdown(
                """
- **High income + advanced education** segments with high conversion can be targeted with premium loan offers.
- **High CCAvg + CDAccount holders** often show strong cross-sell potential – ideal for relationship-based campaigns.
- **Digital + Card-heavy segments** (Online & Credit Card users) are perfect for **email + in-app** journeys.
- Use the **Age × Family heatmap** to craft life-stage messaging (e.g., education loans, renovation loans).
"""
            )


# ---- TAB 2: Model Performance ----
with tab2:
    st.markdown("<div class='section-title'>Apply All Three Algorithms & Compare Performance</div>", unsafe_allow_html=True)
    if X is None or y is None:
        st.error("Base dataset is missing. Please ensure 'UniversalBank.csv' is in the same folder as app.py in your GitHub repo.")
    else:
        st.markdown(
            "<div class='section-subtitle'>Tune the train/test split and compare Decision Tree, Random Forest and Gradient Boosting using accuracy, AUC and confusion matrices.</div>",
            unsafe_allow_html=True,
        )

        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            test_size = st.slider("Test size (%)", min_value=20, max_value=40, value=30, step=5) / 100.0
        with col_cfg2:
            random_state = st.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)

        if st.button("🚀 Run / Re-run Models"):
            (
                metrics_df,
                models,
                roc_curves,
                conf_matrices,
                feature_importances,
                split_data,
            ) = train_and_evaluate_models(X, y, test_size=test_size, random_state=random_state)

            st.markdown("### 1. Performance Summary Table")
            st.dataframe(metrics_df.style.format("{:.4f}"))

            # Small KPI strip for best model by AUC(Test)
            best_algo = metrics_df["AUC (Test)"].idxmax()
            best_auc = metrics_df.loc[best_algo, "AUC (Test)"]
            best_recall = metrics_df.loc[best_algo, "Recall"]
            k1, k2, k3 = st.columns(3)
            with k1:
                st.metric("Champion Model", best_algo)
            with k2:
                st.metric("Champion AUC (Test)", f"{best_auc:.3f}")
            with k3:
                st.metric("Champion Recall", f"{best_recall:.3f}")

            st.markdown("### 2. ROC Curve (Test Set)")
            algo_select = st.multiselect(
                "Select algorithms to display on ROC curve",
                list(roc_curves.keys()),
                default=list(roc_curves.keys())
            )
            if algo_select:
                fig_roc = plot_roc_curves(roc_curves, selected=algo_select)
                st.pyplot(fig_roc)
            else:
                st.info("Select at least one algorithm to view the ROC curve.")

            st.markdown("### 3. Confusion Matrices (Train & Test)")
            focus_algo = st.selectbox(
                "Focus on specific algorithm (optional)",
                ["All"] + list(conf_matrices.keys()),
                index=0
            )
            focus_param = None if focus_algo == "All" else focus_algo
            fig_cm = plot_confusion_matrices(conf_matrices, focus=focus_param)
            st.pyplot(fig_cm)

            st.markdown("### 4. Feature Importances")
            fig_fi = plot_feature_importances(feature_importances, feature_cols)
            st.pyplot(fig_fi)

            with st.expander("📌 Interpretation Tips"):
                st.markdown(
                    """
- **Use AUC(Test)** to choose the most robust model overall.
- **Recall** is critical if you care about catching as many potential loan takers as possible.
- Compare **feature importances** across models – stable top features (Income, CCAvg, CDAccount, etc.) are strong drivers for campaign design.
"""
                )
        else:
            st.warning("Set your configuration and click **Run / Re-run Models** to generate metrics and charts.")


# ---- TAB 3: New Data Scoring ----
with tab3:
    st.markdown("<div class='section-title'>Upload New Customer File & Predict Personal Loan Propensity</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Score fresh customer lists using the best-performing model and download a ready-to-use file for campaigns.</div>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Upload new customer CSV", type=["csv"])

    if uploaded_file is not None:
        new_df_raw = pd.read_csv(uploaded_file)
        new_df = standardize_columns(new_df_raw)
        st.markdown("##### Preview of uploaded data")
        st.dataframe(new_df_raw.head())

        if X is None or y is None:
            st.error("Base dataset is missing, so models cannot be trained. Ensure 'UniversalBank.csv' is in the same folder as app.py.")
        else:
            (
                metrics_df,
                models,
                roc_curves,
                conf_matrices,
                feature_importances,
                split_data,
            ) = train_and_evaluate_models(X, y)

            best_algo = metrics_df["AUC (Test)"].idxmax()
            best_model = models[best_algo]
            st.success(f"Best model based on AUC(Test): **{best_algo}**")

            missing_cols = [c for c in feature_cols if c not in new_df.columns]
            if missing_cols:
                st.error(
                    "Uploaded file is missing the following required columns (after cleaning): "
                    + ", ".join(missing_cols)
                )
            else:
                X_new = new_df[feature_cols]
                proba = best_model.predict_proba(X_new)[:, 1]
                # Allow user to adjust decision threshold interactively
                threshold = st.slider("Decision threshold for classifying as 'Will take loan'", 0.1, 0.9, 0.5, 0.05)
                pred = (proba >= threshold).astype(int)

                scored = new_df_raw.copy()
                scored["PredictedPersonalLoan"] = pred
                scored["LoanProbability"] = proba

                st.markdown("### Sample of Scored Customers")
                st.dataframe(scored.head())

                high_prop = (proba >= threshold).mean() * 100.0
                st.metric("Share of customers above threshold", f"{high_prop:.1f}%")

                csv_bytes = scored.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Scored File (CSV)",
                    data=csv_bytes,
                    file_name="scored_customers_with_personal_loan_prediction.csv",
                    mime="text/csv",
                )

                with st.expander("How to use this file in campaigns"):
                    st.markdown(
                        """
- Filter customers with **LoanProbability above your chosen threshold** for high-intent targeting.
- Combine propensity with **Income, CCAvg and product holdings** to create differentiated offer tiers.
- Export to your CRM or marketing automation tool to trigger **email/SMS/app** journeys.
"""
                    )
    else:
        st.warning("Please upload a CSV file to score new customers.")
