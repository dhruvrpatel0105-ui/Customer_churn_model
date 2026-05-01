import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_curve, auc
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; color: #e0e0e0; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #1a1f2e 0%, #141824 100%);
        border-right: 1px solid #2d3748;
        color: white;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 12px 20px;
        border-radius: 10px;
        color: white;
        font-size: 1.25rem;
        font-weight: 700;
        margin: 20px 0 15px 0;
        letter-spacing: 0.5px;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2a3a 0%, #16213e 100%);
        border: 1px solid #2d4a6e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 8px 0;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #63b3ed;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #a0aec0;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Prediction result boxes */
    .predict-churn {
        background: linear-gradient(135deg, #742a2a, #9b2c2c);
        border: 2px solid #fc8181;
        border-radius: 14px;
        padding: 28px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        color: #fff5f5;
        margin: 12px 0;
    }
    .predict-no-churn {
        background: linear-gradient(135deg, #1a4731, #276749);
        border: 2px solid #68d391;
        border-radius: 14px;
        padding: 28px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        color: #f0fff4;
        margin: 12px 0;
    }

    /* Info card */
    .info-card {
        background: #1a2233;
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 10px 0;
        color: #cbd5e0;
        font-size: 0.9rem;
    }

    /* Divider */
    hr { border-color: #2d3748; }

    /* Plot backgrounds */
    .stPlotlyChart, .element-container { background-color: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  DATA & MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Customertravel.csv")
    return df

@st.cache_resource
def load_model_and_data():
    df = load_data()
    df_proc = df.copy()

    # Preprocessing (matches notebook exactly)
    df_proc['Frequent_Flyer'] = df_proc['Frequent_Flyer'].replace('No Record', pd.NA)
    df_proc['Frequent_Flyer'] = df_proc['Frequent_Flyer'].fillna(df_proc['Frequent_Flyer'].mode()[0])
    df_proc['Frequent_Flyer'] = df_proc['Frequent_Flyer'].map({'Yes': 1, 'No': 0})
    df_proc['Account_Synced_ToSocial_Media'] = df_proc['Account_Synced_ToSocial_Media'].map({'Yes': 1, 'No': 0})
    df_proc['Booked_Hotel'] = df_proc['Booked_Hotel'].map({'Yes': 1, 'No': 0})
    df_proc['Annual_Income_Class'] = df_proc['Annual_Income_Class'].map(
        {'Low Income': 0, 'Middle Income': 1, 'High Income': 2}
    )

    X = df_proc.drop('Target', axis=1)
    Y = df_proc['Target']
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, train_size=0.80, test_size=0.20, random_state=42
    )

    try:
        model = joblib.load("model.pkl")
    except Exception:
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, Y_train)

    Y_pred  = model.predict(X_test)
    Y_prob  = model.predict_proba(X_test)[:, 1]
    acc     = accuracy_score(Y_test, Y_pred)
    cm      = confusion_matrix(Y_test, Y_pred)
    cr      = classification_report(Y_test, Y_pred, output_dict=True)
    fpr, tpr, _ = roc_curve(Y_test, Y_prob)
    roc_auc = auc(fpr, tpr)
    fi      = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    return model, df, X, Y, acc, cm, cr, fpr, tpr, roc_auc, fi


model, df_raw, X, Y, acc, cm, cr, fpr, tpr, roc_auc, fi = load_model_and_data()


# ─────────────────────────────────────────────
#  SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✈️ Churn Prediction")
    st.markdown("---")
    page = st.radio(
        "Navigate to",
        ["🏠 Home & Overview",
         "📊 Data Exploration",
         "🤖 Model Evaluation",
         "🔮 Predict Churn"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(
        "<div class='info-card'>📁 <b>Dataset:</b> Customertravel.csv<br>"
        "🧠 <b>Algorithm:</b> Random Forest<br>"
        "🎯 <b>Task:</b> Binary Classification</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='info-card'>📏 <b>Records:</b> {df_raw.shape[0]}<br>"
        f"🧩 <b>Features:</b> {df_raw.shape[1] - 1}<br>"
        f"✅ <b>Accuracy:</b> {acc*100:.2f}%</div>",
        unsafe_allow_html=True
    )


# ═══════════════════════════════════════════
#  PAGE 1 — HOME & OVERVIEW
# ═══════════════════════════════════════════
if page == "🏠 Home & Overview":
    st.markdown("<h1 style='text-align:center;color:#667eea;'>✈️ Customer Churn Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#a0aec0;font-size:1.05rem;'>End-to-End Machine Learning Solution using Random Forest</p>", unsafe_allow_html=True)
    st.markdown("---")

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{df_raw.shape[0]}</div><div class='metric-label'>Total Customers</div></div>", unsafe_allow_html=True)
    with col2:
        churn_pct = df_raw['Target'].mean() * 100
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{churn_pct:.1f}%</div><div class='metric-label'>Churn Rate</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{acc*100:.2f}%</div><div class='metric-label'>Model Accuracy</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{roc_auc:.3f}</div><div class='metric-label'>ROC AUC Score</div></div>", unsafe_allow_html=True)

    st.markdown("---")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("<div class='section-header'>📌 What is Customer Churn?</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-card'>
        Customer churn (also known as <b>customer attrition</b>) refers to the rate at which customers
        stop doing business with a company — cancelling subscriptions, switching to competitors,
        or simply stopping to use a service.<br><br>
        In the travel/insurance industry, churn means customers who <b>do not purchase travel insurance</b>
        despite being eligible, resulting in lost revenue and engagement.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-header'>💡 Why Predict Churn?</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-card'>
        ✔ <b>Proactive intervention:</b> Identify at-risk customers before they leave<br>
        ✔ <b>Cost savings:</b> Retaining customers is ~5× cheaper than acquiring new ones<br>
        ✔ <b>Personalisation:</b> Tailor offers to likely churners<br>
        ✔ <b>Revenue protection:</b> Prioritise retention campaigns efficiently
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("<div class='section-header'>🌲 Why Random Forest?</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-card'>
        Random Forest is an <b>ensemble learning</b> method that builds multiple decision trees
        and merges their results for more accurate, stable predictions.<br><br>
        ✔ Handles both numerical & categorical features naturally<br>
        ✔ Robust to overfitting compared to single decision trees<br>
        ✔ Provides built-in <b>feature importance</b> scores<br>
        ✔ Works well on imbalanced datasets<br>
        ✔ Minimal hyperparameter tuning required
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-header'>🗂️ Dataset Snapshot</div>", unsafe_allow_html=True)
        st.dataframe(df_raw.head(8), use_container_width=True)


# ═══════════════════════════════════════════
#  PAGE 2 — DATA EXPLORATION
# ═══════════════════════════════════════════
elif page == "📊 Data Exploration":
    st.markdown("<div class='section-header'>📊 Data Exploration & Visualisations</div>", unsafe_allow_html=True)

    # Summary stats
    with st.expander("📋 Summary Statistics", expanded=False):
        st.dataframe(df_raw.describe().T.style.background_gradient(cmap="Blues"), use_container_width=True)

    with st.expander("🔍 Missing Values & Data Types", expanded=False):
        info_df = pd.DataFrame({
            "Data Type": df_raw.dtypes,
            "Non-Null Count": df_raw.notnull().sum(),
            "Missing Values": df_raw.isnull().sum(),
            "Unique Values": df_raw.nunique()
        })
        st.dataframe(info_df, use_container_width=True)

    st.markdown("---")

    # Row 1 — Churn distribution + Frequent Flyer
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🎯 Churn (Target) Distribution**")
        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor='#1a1f2e')
        ax.set_facecolor('#1a1f2e')
        counts = df_raw['Target'].value_counts()
        colors = ['#48bb78', '#fc8181']
        bars = ax.bar(['No Churn (0)', 'Churned (1)'], counts.values, color=colors, edgecolor='none', width=0.5)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(val), ha='center', va='bottom', color='white', fontsize=11, fontweight='bold')
        ax.set_title("Churn Distribution", color='white', fontsize=12, fontweight='bold')
        ax.set_ylabel("Count", color='#a0aec0')
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.yaxis.label.set_color('#a0aec0')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**✈️ Frequent Flyer vs Churn**")
        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor='#1a1f2e')
        ax.set_facecolor('#1a1f2e')
        ct = pd.crosstab(df_raw['Frequent_Flyer'], df_raw['Target'])
        ct.columns = ['No Churn', 'Churned']
        ct.plot(kind='bar', ax=ax, color=['#4299e1', '#f6ad55'], edgecolor='none', width=0.6)
        ax.set_title("Frequent Flyer vs Churn", color='white', fontsize=12, fontweight='bold')
        ax.set_xlabel("Frequent Flyer", color='#a0aec0')
        ax.set_ylabel("Count", color='#a0aec0')
        ax.tick_params(colors='white', axis='both')
        ax.legend(facecolor='#1a1f2e', labelcolor='white', fontsize=9)
        plt.xticks(rotation=0)
        for spine in ax.spines.values(): spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Row 2 — Age scatter + Income distribution
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**🎂 Age vs Churn**")
        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor='#1a1f2e')
        ax.set_facecolor('#1a1f2e')
        no_churn = df_raw[df_raw['Target'] == 0]
        churned  = df_raw[df_raw['Target'] == 1]
        ax.scatter(no_churn['Age'], no_churn['Target'], alpha=0.5, color='#68d391', s=18, label='No Churn')
        ax.scatter(churned['Age'], churned['Target'], alpha=0.6, color='#fc8181', s=18, label='Churned')
        ax.set_title("Age vs Churn Status", color='white', fontsize=12, fontweight='bold')
        ax.set_xlabel("Age", color='#a0aec0')
        ax.set_ylabel("Churn (0 / 1)", color='#a0aec0')
        ax.tick_params(colors='white')
        ax.legend(facecolor='#1a1f2e', labelcolor='white', fontsize=9)
        for spine in ax.spines.values(): spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col4:
        st.markdown("**💰 Annual Income Class Distribution**")
        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor='#1a1f2e')
        ax.set_facecolor('#1a1f2e')
        income_counts = df_raw['Annual_Income_Class'].value_counts()
        ax.bar(income_counts.index, income_counts.values,
               color=['#667eea', '#764ba2', '#f093fb'], edgecolor='none', width=0.5)
        ax.set_title("Annual Income Class Distribution", color='white', fontsize=12, fontweight='bold')
        ax.set_xlabel("Income Class", color='#a0aec0')
        ax.set_ylabel("Frequency", color='#a0aec0')
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Row 3 — Services opted + Hotel Booked
    col5, col6 = st.columns(2)
    with col5:
        st.markdown("**🛎️ Services Opted vs Churn**")
        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor='#1a1f2e')
        ax.set_facecolor('#1a1f2e')
        ct2 = pd.crosstab(df_raw['Services_Opted'], df_raw['Target'])
        ct2.columns = ['No Churn', 'Churned']
        ct2.plot(kind='bar', ax=ax, color=['#4299e1', '#ed64a6'], edgecolor='none', width=0.6)
        ax.set_title("Services Opted vs Churn", color='white', fontsize=12, fontweight='bold')
        ax.set_xlabel("Services Opted", color='#a0aec0')
        ax.set_ylabel("Count", color='#a0aec0')
        ax.tick_params(colors='white')
        ax.legend(facecolor='#1a1f2e', labelcolor='white', fontsize=9)
        plt.xticks(rotation=0)
        for spine in ax.spines.values(): spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col6:
        st.markdown("**🏨 Hotel Booked vs Churn**")
        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor='#1a1f2e')
        ax.set_facecolor('#1a1f2e')
        ct3 = pd.crosstab(df_raw['Booked_Hotel'], df_raw['Target'])
        ct3.columns = ['No Churn', 'Churned']
        ct3.plot(kind='bar', ax=ax, color=['#38b2ac', '#f6ad55'], edgecolor='none', width=0.5)
        ax.set_title("Hotel Booked vs Churn", color='white', fontsize=12, fontweight='bold')
        ax.set_xlabel("Booked Hotel", color='#a0aec0')
        ax.set_ylabel("Count", color='#a0aec0')
        ax.tick_params(colors='white')
        ax.legend(facecolor='#1a1f2e', labelcolor='white', fontsize=9)
        plt.xticks(rotation=0)
        for spine in ax.spines.values(): spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Correlation heatmap on processed data
    st.markdown("**🔥 Feature Correlation Heatmap**")
    df_corr = df_raw.copy()
    df_corr['Frequent_Flyer'] = df_corr['Frequent_Flyer'].replace('No Record', pd.NA)
    df_corr['Frequent_Flyer'] = df_corr['Frequent_Flyer'].fillna(df_corr['Frequent_Flyer'].mode()[0])
    df_corr['Frequent_Flyer'] = df_corr['Frequent_Flyer'].map({'Yes': 1, 'No': 0})
    df_corr['Account_Synced_ToSocial_Media'] = df_corr['Account_Synced_ToSocial_Media'].map({'Yes': 1, 'No': 0})
    df_corr['Booked_Hotel'] = df_corr['Booked_Hotel'].map({'Yes': 1, 'No': 0})
    df_corr['Annual_Income_Class'] = df_corr['Annual_Income_Class'].map({'Low Income': 0, 'Middle Income': 1, 'High Income': 2})
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='#1a1f2e')
    ax.set_facecolor('#1a1f2e')
    sns.heatmap(df_corr.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                linewidths=0.5, linecolor='#2d3748', annot_kws={"size": 9})
    ax.set_title("Correlation Matrix", color='white', fontsize=12, fontweight='bold')
    plt.xticks(color='white', fontsize=8); plt.yticks(color='white', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════
#  PAGE 3 — MODEL EVALUATION
# ═══════════════════════════════════════════
elif page == "🤖 Model Evaluation":
    st.markdown("<div class='section-header'>🤖 Random Forest — Model Evaluation</div>", unsafe_allow_html=True)

    # Accuracy + classification report
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{acc*100:.2f}%</div><div class='metric-label'>Overall Accuracy</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{roc_auc:.4f}</div><div class='metric-label'>AUC Score</div></div>", unsafe_allow_html=True)
        precision_1 = cr['1']['precision']
        recall_1    = cr['1']['recall']
        f1_1        = cr['1']['f1-score']
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{precision_1:.2f}</div><div class='metric-label'>Precision (Churn)</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{recall_1:.2f}</div><div class='metric-label'>Recall (Churn)</div></div>", unsafe_allow_html=True)

    with col2:
        st.markdown("**📄 Classification Report**")
        cr_df = pd.DataFrame(cr).transpose().round(3)
        cr_df = cr_df.drop(['accuracy'], errors='ignore')
        st.dataframe(cr_df.style.background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']),
                     use_container_width=True)

    st.markdown("---")

    # Confusion Matrix + ROC Curve
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**🧩 Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(4.5, 3.5), facecolor='#1a1f2e')
        ax.set_facecolor('#1a1f2e')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No Churn', 'Churned'],
                    yticklabels=['No Churn', 'Churned'],
                    linewidths=0.5, linecolor='#2d3748',
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_title("Confusion Matrix", color='white', fontsize=12, fontweight='bold')
        ax.set_xlabel("Predicted", color='#a0aec0')
        ax.set_ylabel("Actual", color='#a0aec0')
        ax.tick_params(colors='white')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Breakdown
        tn, fp, fn, tp = cm.ravel()
        st.markdown(f"""
        <div class='info-card'>
        ✅ <b>True Negatives (TN):</b> {tn} — Correctly predicted No Churn<br>
        ✅ <b>True Positives (TP):</b> {tp} — Correctly predicted Churn<br>
        ❌ <b>False Positives (FP):</b> {fp} — Predicted Churn, but No Churn<br>
        ❌ <b>False Negatives (FN):</b> {fn} — Predicted No Churn, but Churned
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown("**📈 ROC Curve**")
        fig, ax = plt.subplots(figsize=(4.5, 3.5), facecolor='#1a1f2e')
        ax.set_facecolor('#1a1f2e')
        ax.plot(fpr, tpr, color='#667eea', lw=2.5, label=f'AUC = {roc_auc:.3f}')
        ax.plot([0, 1], [0, 1], linestyle='--', color='#718096', lw=1.5, label='Random Classifier')
        ax.fill_between(fpr, tpr, alpha=0.15, color='#667eea')
        ax.set_title("ROC Curve", color='white', fontsize=12, fontweight='bold')
        ax.set_xlabel("False Positive Rate", color='#a0aec0')
        ax.set_ylabel("True Positive Rate", color='#a0aec0')
        ax.tick_params(colors='white')
        ax.legend(facecolor='#1a1f2e', labelcolor='white', fontsize=10)
        for spine in ax.spines.values(): spine.set_color('#2d3748')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # Feature Importance
    st.markdown("<div class='section-header'>🏆 Feature Importance Analysis</div>", unsafe_allow_html=True)
    col5, col6 = st.columns([3, 2])
    with col5:
        fig, ax = plt.subplots(figsize=(7, 3.5), facecolor='#1a1f2e')
        ax.set_facecolor('#1a1f2e')
        palette = ['#667eea', '#764ba2', '#f093fb', '#4299e1', '#48bb78', '#f6ad55']
        bars = ax.barh(fi.index[::-1], fi.values[::-1], color=palette[::-1], edgecolor='none', height=0.5)
        for bar, val in zip(bars, fi.values[::-1]):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', ha='left', color='white', fontsize=9)
        ax.set_title("Feature Importance (Random Forest)", color='white', fontsize=12, fontweight='bold')
        ax.set_xlabel("Importance Score", color='#a0aec0')
        ax.tick_params(colors='white')
        ax.set_xlim(0, fi.max() * 1.2)
        for spine in ax.spines.values(): spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col6:
        st.markdown("**Feature Importance Scores**")
        fi_df = pd.DataFrame({'Feature': fi.index, 'Importance': fi.values.round(4)})
        st.dataframe(fi_df.style.background_gradient(cmap='Blues', subset=['Importance']),
                     use_container_width=True, hide_index=True)
        st.markdown("""
        <div class='info-card' style='font-size:0.82rem;'>
        🏆 <b>Top features</b> driving churn prediction are highlighted above.
        Higher importance = stronger influence on the model's decisions.
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
#  PAGE 4 — PREDICT CHURN (USER INPUT)
# ═══════════════════════════════════════════
elif page == "🔮 Predict Churn":
    st.markdown("<div class='section-header'>🔮 Predict Customer Churn — Try Your Own Input</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-card'>
    Enter the customer details below and click <b>Predict</b> to find out whether this customer
    is likely to churn (not purchase travel insurance) or not.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**👤 Demographics**")
            age = st.slider("Age", min_value=18, max_value=65, value=30, step=1,
                            help="Customer's age in years")
            annual_income = st.selectbox(
                "Annual Income Class",
                options=["Low Income", "Middle Income", "High Income"],
                index=1,
                help="Customer's annual income bracket"
            )

        with col2:
            st.markdown("**✈️ Travel Profile**")
            frequent_flyer = st.selectbox(
                "Frequent Flyer",
                options=["No", "Yes"],
                index=0,
                help="Does the customer fly frequently?"
            )
            services_opted = st.slider(
                "Services Opted", min_value=1, max_value=6, value=3, step=1,
                help="Number of travel services opted by the customer"
            )

        with col3:
            st.markdown("**📱 Digital & Booking**")
            account_synced = st.selectbox(
                "Account Synced to Social Media",
                options=["No", "Yes"],
                index=0,
                help="Is the customer's account synced to social media?"
            )
            booked_hotel = st.selectbox(
                "Booked Hotel",
                options=["No", "Yes"],
                index=1,
                help="Has the customer booked a hotel?"
            )

        st.markdown("---")
        submitted = st.form_submit_button("🔮  Predict Now", use_container_width=True)

    if submitted:
        # Encode inputs exactly as in preprocessing
        income_map = {"Low Income": 0, "Middle Income": 1, "High Income": 2}
        binary_map = {"No": 0, "Yes": 1}

        input_data = pd.DataFrame([{
            "Age": age,
            "Frequent_Flyer": binary_map[frequent_flyer],
            "Annual_Income_Class": income_map[annual_income],
            "Services_Opted": services_opted,
            "Account_Synced_ToSocial_Media": binary_map[account_synced],
            "Booked_Hotel": binary_map[booked_hotel]
        }])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        st.markdown("---")
        col_res1, col_res2 = st.columns(2)

        with col_res1:
            if prediction == 1:
                st.markdown("""
                <div class='predict-churn'>
                    ⚠️ HIGH CHURN RISK<br>
                    <span style='font-size:1rem;font-weight:400;'>
                    This customer is likely to churn.<br>Consider targeted retention action.
                    </span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='predict-no-churn'>
                    ✅ LOW CHURN RISK<br>
                    <span style='font-size:1rem;font-weight:400;'>
                    This customer is likely to stay.<br>Continue engagement to retain loyalty.
                    </span>
                </div>""", unsafe_allow_html=True)

        with col_res2:
            st.markdown("**📊 Prediction Probabilities**")
            fig, ax = plt.subplots(figsize=(4, 2.5), facecolor='#1a1f2e')
            ax.set_facecolor('#1a1f2e')
            labels = ['No Churn', 'Churned']
            colors = ['#48bb78', '#fc8181']
            bars = ax.bar(labels, probability, color=colors, edgecolor='none', width=0.4)
            for bar, prob in zip(bars, probability):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{prob*100:.1f}%', ha='center', va='bottom',
                        color='white', fontsize=13, fontweight='bold')
            ax.set_ylim(0, 1.15)
            ax.set_ylabel("Probability", color='#a0aec0')
            ax.tick_params(colors='white')
            for spine in ax.spines.values(): spine.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Input summary
        st.markdown("**📋 Input Summary**")
        summary = pd.DataFrame({
            "Feature": ["Age", "Frequent Flyer", "Annual Income Class",
                        "Services Opted", "Account Synced to Social Media", "Booked Hotel"],
            "Value": [age, frequent_flyer, annual_income,
                      services_opted, account_synced, booked_hotel]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

        # Actionable insight
        churn_prob = probability[1]
        if churn_prob >= 0.7:
            insight = "🔴 <b>Very High Risk:</b> Immediate retention offer recommended — personalised discount or loyalty upgrade."
        elif churn_prob >= 0.5:
            insight = "🟠 <b>Moderate Risk:</b> Send a targeted email with travel insurance benefits and a limited-time offer."
        elif churn_prob >= 0.3:
            insight = "🟡 <b>Low-Moderate Risk:</b> Monitor engagement. Light nudges like in-app reminders may help."
        else:
            insight = "🟢 <b>Low Risk:</b> Customer appears satisfied. Continue regular engagement to maintain loyalty."

        st.markdown(f"<div class='info-card'><b>💡 Recommended Action:</b><br>{insight}</div>",
                    unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#4a5568;font-size:0.8rem;'>"
    "Customer Churn Prediction | B.Tech Gen AI — 2nd Semester Final Project | Random Forest Classifier"
    "</p>",
    unsafe_allow_html=True
)
