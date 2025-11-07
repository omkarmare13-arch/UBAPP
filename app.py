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
    df = pd.read_csv("UniversalBank.csv")
    df = standardize_columns(df)
    return df


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
        raise ValueError("Could not find 'Personal Loan' column in data.")

    # drop ID if present
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


def train_and_evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
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
def plot_roc_curves(roc_curves):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (fpr, tpr, auc_val) in roc_curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (Test Set)")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_confusion_matrices(conf_matrices):
    algorithms = list(conf_matrices.keys())
    splits = ["train", "test"]

    fig, axes = plt.subplots(
        nrows=len(algorithms), ncols=len(splits), figsize=(10, 10)
    )

    if len(algorithms) == 1:
        axes = np.array([[axes[0], axes[1]]])

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
st.title("Personal Loan Propensity & Customer Insight Dashboard")

st.markdown('''
As **Head of Marketing**, use this app to:
- Understand which customer segments are most likely to accept a *personal loan*.
- Compare machine learning models (Decision Tree, Random Forest, Gradient Boosting).
- Score new customer lists and download predictions for campaigns.
''')

base_df = load_base_data()
X, y, feature_cols, col_map, target_col, model_df = prepare_features(base_df)

tab1, tab2, tab3 = st.tabs(
    [
        "1️⃣ Customer Insights",
        "2️⃣ Model Performance (DT / RF / GBT)",
        "3️⃣ Score New Customer File",
    ]
)

# ---- TAB 1: Customer Insights ----
with tab1:
    st.subheader("Customer Insights for Better Marketing Actions")
    st.markdown('''
Below charts combine multiple variables so that you can design **sharper targeting rules**,
e.g. *income × education*, *digital usage × cards*, etc.
''')

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 1. Conversion by Income Quintile & Education")
        fig1 = chart_conversion_by_income_education(model_df, col_map)
        st.pyplot(fig1)

    with col_b:
        st.markdown("#### 2. Conversion by Credit Card Spend & CD Account")
        fig2 = chart_conversion_by_ccavg_cd(model_df, col_map)
        st.pyplot(fig2)

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("#### 3. Conversion Heatmap: Age Band vs Family Size")
        fig3 = chart_heatmap_age_family(model_df, col_map)
        st.pyplot(fig3)

    with col_d:
        st.markdown("#### 4. Digital & Card Segments: Conversion & Volume")
        fig4 = chart_online_creditcard_segments(model_df, col_map)
        st.pyplot(fig4)

    st.markdown("#### 5. Correlation of Features with Personal Loan")
    fig5 = chart_feature_corr_with_target(model_df, col_map, feature_cols)
    st.pyplot(fig5)

    st.info(
        "Focus campaigns on segments with high conversion rate but medium customer count "
        "(e.g. high-income + advanced education, high CCAvg + CDAccount) to get quick wins."
    )


# ---- TAB 2: Model Performance ----
with tab2:
    st.subheader("Apply All Three Algorithms & Compare Performance")
    st.markdown('''
Click the button below to train **Decision Tree**, **Random Forest** and
**Gradient Boosting** on the Universal Bank data, using 70/30 train-test split and
**5-fold cross validation** on the training set.
''')

    if st.button("🚀 Run / Re-run Models"):
        (
            metrics_df,
            models,
            roc_curves,
            conf_matrices,
            feature_importances,
            split_data,
        ) = train_and_evaluate_models(X, y)

        st.markdown("### 1. Performance Summary Table")
        st.dataframe(metrics_df.style.format("{:.4f}"))

        st.markdown("### 2. ROC Curve (Test Set, All Models)")
        fig_roc = plot_roc_curves(roc_curves)
        st.pyplot(fig_roc)

        st.markdown("### 3. Confusion Matrices (Train & Test)")
        fig_cm = plot_confusion_matrices(conf_matrices)
        st.pyplot(fig_cm)

        st.markdown("### 4. Feature Importances")
        fig_fi = plot_feature_importances(feature_importances, feature_cols)
        st.pyplot(fig_fi)

        st.info(
            "Use the **AUC (Test)** and **Recall** columns to select the best model "
            "for maximizing loan conversions."
        )
    else:
        st.warning("Click **Run / Re-run Models** to generate metrics and charts.")


# ---- TAB 3: New Data Scoring ----
with tab3:
    st.subheader("Upload New Customer File & Predict Personal Loan Propensity")
    st.markdown('''
Upload a **CSV file** with the same structure as the original UniversalBank data
(ID, Age, Experience, Income, Family, CCAvg, Education, Mortgage, Securities, CDAccount,
Online, CreditCard, etc.).  

The app will train all three models again on the original data and then use the
**best model (highest Test AUC)** to score the uploaded file.
''')

    uploaded_file = st.file_uploader("Upload new customer CSV", type=["csv"])

    if uploaded_file is not None:
        new_df_raw = pd.read_csv(uploaded_file)
        new_df = standardize_columns(new_df_raw)
        st.markdown("Preview of uploaded data:")
        st.dataframe(new_df_raw.head())

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
            pred = (proba >= 0.5).astype(int)

            scored = new_df_raw.copy()
            scored["PredictedPersonalLoan"] = pred
            scored["LoanProbability"] = proba

            st.markdown("### Sample of Scored Customers")
            st.dataframe(scored.head())

            csv_bytes = scored.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Scored File (CSV)",
                data=csv_bytes,
                file_name="scored_customers_with_personal_loan_prediction.csv",
                mime="text/csv",
            )

            st.info(
                "Filter customers with **LoanProbability >= 0.7** to build a "
                "high-propensity campaign list."
            )
    else:
        st.warning("Please upload a CSV file to score new customers.")
