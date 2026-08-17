# ============================================================
#   MovieIQ — Predictive Analytics on Film Success
# ============================================================
# HOW TO RUN:
#   1. Put this file in your project folder
#   2. Put movies.csv in same folder
#   3. pip install streamlit pandas numpy scikit-learn seaborn matplotlib scipy
#   4. python -m streamlit run MovieIQ.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from scipy import stats
import ast
import warnings
warnings.filterwarnings('ignore')
import os
from pathlib import Path

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="MovieIQ",
    page_icon="🎬",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0d0d0d; }
    [data-testid="stSidebar"] { background-color: #1a1a2e; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #e94560;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .section-title {
        font-size: 20px;
        font-weight: bold;
        color: #e94560;
        margin: 20px 0 10px 0;
        padding-left: 10px;
        border-left: 4px solid #e94560;
    }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #e94560;
        border-radius: 10px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load & Prepare Data ──────────────────────────────────────
@st.cache_data
def load_and_prepare():

    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "movies.csv"

    # Load CSV
    df = pd.read_csv(csv_path)

    # Parse genres
    def parse_genres(genre_str):
        try:
            genres = ast.literal_eval(genre_str)
            return [g['name'] for g in genres]
        except:
            return []

    df['genre_list'] = df['genres'].apply(parse_genres)
    df['primary_genre'] = df['genre_list'].apply(
        lambda x: x[0] if len(x) > 0 else 'Unknown')

    # Create success column
    df['success'] = (df['revenue'] > df['budget']).astype(int)
    df['success_label'] = df['success'].map({1: 'Success', 0: 'Failure'})

    # Feature engineering
    df['profit'] = df['revenue'] - df['budget']
    df['roi'] = (df['profit'] / df['budget']) * 100

    return df

df = load_and_prepare()

# ── Train ML Model ───────────────────────────────────────────
@st.cache_resource
def train_model(df):
    features = ['budget', 'popularity', 'runtime', 'vote_average']
    X = df[features]
    y = df['success']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    return model, accuracy, precision, recall, cm, importance, X_test, y_test, y_pred

model, accuracy, precision, recall, cm, importance, X_test, y_test, y_pred = train_model(df)

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.markdown("""
<div style='text-align:center; padding:10px 0 20px 0;'>
    <span style='font-size:40px;'>🎬</span><br>
    <span style='font-size:22px; font-weight:bold; color:#e94560;'>MovieIQ</span><br>
    <span style='font-size:12px; color:#888;'>Predictive Analytics on Film Success</span>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigate", [
    "🏠 Home",
    "📊 EDA Analysis",
    "🧪 Statistical Tests",
    "🤖 ML Model",
    "🎯 Predict Movie"
])

# Sidebar Filters
st.sidebar.markdown("---")
st.sidebar.markdown("## 🔍 Filters")

all_genres = sorted(set([g for genres in df['genre_list'] for g in genres]))
selected_genre = st.sidebar.selectbox("🎭 Genre", ["All"] + all_genres)
min_votes = st.sidebar.slider("⭐ Min Vote Average", 0.0, 10.0, 0.0, 0.1)

# Apply filters
filtered = df.copy()
if selected_genre != "All":
    filtered = filtered[filtered['genre_list'].apply(lambda x: selected_genre in x)]
filtered = filtered[filtered['vote_average'] >= min_votes]

st.sidebar.markdown(f"**Showing:** {len(filtered):,} movies")

# ============================================================
# HOME PAGE
# ============================================================
if page == "🏠 Home":
    st.markdown("""
    <div style='text-align:center; padding:30px 0 10px 0;'>
        <span style='font-size:60px;'>🎬</span>
        <h1 style='color:#e94560; font-size:48px; margin:0;'>MovieIQ</h1>
        <p style='color:#888; font-size:18px;'>Predictive Analytics on Film Success</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # KPI Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("🎬 Total Movies", f"{len(filtered):,}")
    with c2:
        st.metric("✅ Successful", f"{filtered['success'].sum():,}")
    with c3:
        st.metric("❌ Failed", f"{(filtered['success']==0).sum():,}")
    with c4:
        success_rate = filtered['success'].mean() * 100
        st.metric("📊 Success Rate", f"{success_rate:.1f}%")
    with c5:
        st.metric("🤖 Model Accuracy", f"{accuracy*100:.1f}%")

    st.markdown("---")
    st.markdown('<div class="section-title">📋 About This Project</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style='background:#1a1a2e; border:1px solid #e94560; border-radius:10px; padding:20px;'>
        <h4 style='color:#e94560;'>🎯 Project Objective</h4>
        <p style='color:#ccc;'>Build an interactive ML-powered dashboard that analyzes and predicts 
        movie success based on budget, revenue, popularity, runtime, and vote averages.</p>
        <p style='color:#ccc;'><b>Success Definition:</b> A movie is successful when revenue > budget.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='background:#1a1a2e; border:1px solid #e94560; border-radius:10px; padding:20px;'>
        <h4 style='color:#e94560;'>📊 Dataset Overview</h4>
        <p style='color:#ccc;'>2,000 movies with 7 features covering budget, revenue, popularity, 
        runtime, vote average, title, and genres.</p>
        <p style='color:#ccc;'><b>ML Model:</b> Random Forest Classifier with 80/20 train-test split.</p>
        </div>
        """, unsafe_allow_html=True)

    # Quick overview charts
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">✅ Success vs Failure</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0d0d0d')
        ax.set_facecolor('#1a1a2e')
        counts = filtered['success_label'].value_counts()
        colors = ['#2ecc71', '#e94560']
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90,
               textprops={'color': 'white', 'fontsize': 12})
        ax.set_title('Movie Success Distribution', color='white', fontsize=14)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="section-title">🎭 Top Genres</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0d0d0d')
        ax.set_facecolor('#1a1a2e')
        genre_counts = filtered['primary_genre'].value_counts().head(8)
        bars = ax.barh(genre_counts.index, genre_counts.values, color='#e94560')
        ax.set_xlabel('Number of Movies', color='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#444')
        ax.spines['left'].set_color('#444')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title('Movies by Genre', color='white', fontsize=14)
        st.pyplot(fig)
        plt.close()

# ============================================================
# EDA PAGE
# ============================================================
elif page == "📊 EDA Analysis":
    st.markdown("## 📊 Exploratory Data Analysis")
    st.markdown("---")

    # Chart 1 — Budget vs Revenue
    st.markdown('<div class="section-title">💰 Budget vs Revenue</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#1a1a2e')
    colors = filtered['success'].map({1: '#2ecc71', 0: '#e94560'})
    ax.scatter(filtered['budget']/1e6, filtered['revenue']/1e6,
               c=colors, alpha=0.6, s=50)
    ax.set_xlabel('Budget (Millions $)', color='white')
    ax.set_ylabel('Revenue (Millions $)', color='white')
    ax.set_title('Budget vs Revenue (Green=Success, Red=Failure)',
                 color='white', fontsize=14)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)
    plt.close()
    st.info("💡 Higher budget movies generally earn higher revenue, but not always — some low budget films also succeed!")

    # Chart 2 — Genre Success Rate
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">🎭 Genre Success Rate</div>', unsafe_allow_html=True)
        genre_success = filtered.groupby('primary_genre')['success'].mean().sort_values(ascending=True) * 100
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor('#0d0d0d')
        ax.set_facecolor('#1a1a2e')
        bars = ax.barh(genre_success.index, genre_success.values,
                       color=['#2ecc71' if x > 50 else '#e94560' for x in genre_success.values])
        ax.set_xlabel('Success Rate (%)', color='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#444')
        ax.spines['left'].set_color('#444')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title('Success Rate by Genre', color='white', fontsize=12)
        ax.axvline(50, color='yellow', linestyle='--', alpha=0.5)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="section-title">⭐ Vote Average vs Success</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor('#0d0d0d')
        ax.set_facecolor('#1a1a2e')
        success_votes = filtered[filtered['success']==1]['vote_average']
        failure_votes = filtered[filtered['success']==0]['vote_average']
        ax.hist(success_votes, bins=20, alpha=0.7, color='#2ecc71', label='Success')
        ax.hist(failure_votes, bins=20, alpha=0.7, color='#e94560', label='Failure')
        ax.set_xlabel('Vote Average', color='white')
        ax.set_ylabel('Count', color='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#444')
        ax.spines['left'].set_color('#444')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title('Vote Average Distribution', color='white', fontsize=12)
        ax.legend(facecolor='#1a1a2e', labelcolor='white')
        st.pyplot(fig)
        plt.close()

    # Chart 3 — Popularity vs Success
    st.markdown('<div class="section-title">📈 Popularity vs Runtime vs Success</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0d0d0d')
        ax.set_facecolor('#1a1a2e')
        ax.boxplot([filtered[filtered['success']==1]['popularity'],
                    filtered[filtered['success']==0]['popularity']],
                   labels=['Success', 'Failure'],
                   patch_artist=True,
                   boxprops=dict(facecolor='#e94560', color='white'),
                   medianprops=dict(color='yellow'),
                   whiskerprops=dict(color='white'),
                   capprops=dict(color='white'),
                   flierprops=dict(markerfacecolor='white', markersize=3))
        ax.set_ylabel('Popularity', color='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#444')
        ax.spines['left'].set_color('#444')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title('Popularity by Success', color='white', fontsize=12)
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0d0d0d')
        ax.set_facecolor('#1a1a2e')
        ax.boxplot([filtered[filtered['success']==1]['runtime'],
                    filtered[filtered['success']==0]['runtime']],
                   labels=['Success', 'Failure'],
                   patch_artist=True,
                   boxprops=dict(facecolor='#3498db', color='white'),
                   medianprops=dict(color='yellow'),
                   whiskerprops=dict(color='white'),
                   capprops=dict(color='white'),
                   flierprops=dict(markerfacecolor='white', markersize=3))
        ax.set_ylabel('Runtime (mins)', color='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#444')
        ax.spines['left'].set_color('#444')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title('Runtime by Success', color='white', fontsize=12)
        st.pyplot(fig)
        plt.close()

    # Chart 4 — Correlation Heatmap
    st.markdown('<div class="section-title">🔥 Correlation Heatmap</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#1a1a2e')
    numeric_cols = ['budget', 'revenue', 'popularity', 'runtime', 'vote_average', 'success']
    corr = filtered[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
                ax=ax, linewidths=0.5,
                annot_kws={'color': 'white', 'size': 11})
    ax.set_title('Correlation Heatmap of Numeric Features',
                 color='white', fontsize=14)
    ax.tick_params(colors='white')
    st.pyplot(fig)
    plt.close()
    st.info("💡 Budget and Revenue have the strongest correlation (0.7+). Vote average shows moderate positive correlation with success.")

# ============================================================
# STATISTICAL TESTS PAGE
# ============================================================
elif page == "🧪 Statistical Tests":
    st.markdown("## 🧪 Statistical Testing")
    st.markdown("---")

    # T-Test
    st.markdown('<div class="section-title">📐 T-Test: Popularity vs Success</div>', unsafe_allow_html=True)

    success_pop = filtered[filtered['success']==1]['popularity']
    failure_pop = filtered[filtered['success']==0]['popularity']
    t_stat, p_value = stats.ttest_ind(success_pop, failure_pop)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("T-Statistic", f"{t_stat:.4f}")
    with col2:
        st.metric("P-Value", f"{p_value:.6f}")
    with col3:
        st.metric("Significant?", "YES ✅" if p_value < 0.05 else "NO ❌")

    st.markdown(f"""
    **Null Hypothesis (H₀):** There is no significant difference in popularity between successful and unsuccessful movies.

    **Result:** P-value = {p_value:.6f} which is {'less' if p_value < 0.05 else 'greater'} than 0.05.

    **Conclusion:** We {'REJECT' if p_value < 0.05 else 'FAIL TO REJECT'} the null hypothesis.
    {'Popularity significantly differs between successful and unsuccessful movies!' if p_value < 0.05 else 'No significant difference found.'}
    """)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#1a1a2e')
    ax.hist(success_pop, bins=30, alpha=0.7, color='#2ecc71', label=f'Success (mean={success_pop.mean():.1f})')
    ax.hist(failure_pop, bins=30, alpha=0.7, color='#e94560', label=f'Failure (mean={failure_pop.mean():.1f})')
    ax.set_xlabel('Popularity', color='white')
    ax.set_ylabel('Count', color='white')
    ax.set_title('Popularity Distribution: Success vs Failure', color='white', fontsize=14)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Chi-Square Test
    st.markdown('<div class="section-title">🔢 Chi-Square Test: Genre vs Success</div>', unsafe_allow_html=True)

    contingency = pd.crosstab(filtered['primary_genre'], filtered['success_label'])
    chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Chi-Square Stat", f"{chi2:.4f}")
    with col2:
        st.metric("P-Value", f"{p_chi:.6f}")
    with col3:
        st.metric("Significant?", "YES ✅" if p_chi < 0.05 else "NO ❌")

    st.markdown(f"""
    **Null Hypothesis (H₀):** Genre has no association with movie success.

    **Result:** P-value = {p_chi:.6f} which is {'less' if p_chi < 0.05 else 'greater'} than 0.05.

    **Conclusion:** We {'REJECT' if p_chi < 0.05 else 'FAIL TO REJECT'} the null hypothesis.
    {'Genre is significantly associated with movie success!' if p_chi < 0.05 else 'No significant association found.'}

    **What p-value means:** The probability of observing this result by chance alone.
    We use 0.05 (5%) as our threshold — if p < 0.05, the result is statistically significant.
    """)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#1a1a2e')
    genre_success_rate = filtered.groupby('primary_genre')['success'].mean().sort_values() * 100
    bars = ax.barh(genre_success_rate.index, genre_success_rate.values,
                   color=['#2ecc71' if x > 50 else '#e94560' for x in genre_success_rate.values])
    ax.axvline(50, color='yellow', linestyle='--', alpha=0.7, label='50% threshold')
    ax.set_xlabel('Success Rate (%)', color='white')
    ax.set_title('Success Rate by Genre (Chi-Square Test)', color='white', fontsize=14)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    st.pyplot(fig)
    plt.close()

# ============================================================
# ML MODEL PAGE
# ============================================================
elif page == "🤖 ML Model":
    st.markdown("## 🤖 Random Forest Classifier")
    st.markdown("---")

    # Model Performance
    st.markdown('<div class="section-title">📊 Model Performance</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Accuracy", f"{accuracy*100:.1f}%")
    with col2:
        st.metric("🎯 Precision", f"{precision*100:.1f}%")
    with col3:
        st.metric("🎯 Recall", f"{recall*100:.1f}%")
    with col4:
        f1 = 2 * (precision * recall) / (precision + recall)
        st.metric("🎯 F1 Score", f"{f1*100:.1f}%")

    st.markdown("""
    **How Random Forest works (in simple words):**
    Random Forest builds many decision trees using random subsets of the data and features.
    Each tree votes for a prediction, and the majority vote wins.
    This reduces overfitting and improves accuracy compared to a single decision tree.

    **Features used:** Budget, Popularity, Runtime, Vote Average
    **Target:** Success (1 = Revenue > Budget, 0 = Revenue ≤ Budget)
    **Train/Test Split:** 80% training, 20% testing (400 movies for testing)
    """)

    col1, col2 = st.columns(2)

    # Confusion Matrix
    with col1:
        st.markdown('<div class="section-title">🔢 Confusion Matrix</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('#0d0d0d')
        ax.set_facecolor('#1a1a2e')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                    ax=ax, linewidths=1,
                    xticklabels=['Failure', 'Success'],
                    yticklabels=['Failure', 'Success'],
                    annot_kws={'size': 16, 'color': 'white'})
        ax.set_xlabel('Predicted', color='white', fontsize=12)
        ax.set_ylabel('Actual', color='white', fontsize=12)
        ax.set_title('Confusion Matrix', color='white', fontsize=14)
        ax.tick_params(colors='white')
        st.pyplot(fig)
        plt.close()

    # Feature Importance
    with col2:
        st.markdown('<div class="section-title">⭐ Feature Importance</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('#0d0d0d')
        ax.set_facecolor('#1a1a2e')
        bars = ax.barh(importance['Feature'], importance['Importance'],
                       color=['#e94560', '#f39c12', '#3498db', '#2ecc71'])
        ax.set_xlabel('Importance Score', color='white')
        ax.set_title('Feature Importance', color='white', fontsize=14)
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#444')
        ax.spines['left'].set_color('#444')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for bar, val in zip(bars, importance['Importance']):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', color='white', fontsize=11)
        st.pyplot(fig)
        plt.close()

    # Classification Report
    st.markdown('<div class="section-title">📋 Classification Report</div>', unsafe_allow_html=True)
    report = classification_report(y_test, y_pred,
                                   target_names=['Failure', 'Success'],
                                   output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(3), use_container_width=True)

# ============================================================
# PREDICT PAGE
# ============================================================
elif page == "🎯 Predict Movie":
    st.markdown("## 🎯 Predict Movie Success")
    st.markdown("---")
    st.markdown("Enter your movie details below to predict whether it will be a **SUCCESS** or **FAILURE**!")

    col1, col2 = st.columns(2)

    with col1:
        budget = st.number_input("💰 Budget ($)", min_value=1000000, max_value=500000000,
                                  value=50000000, step=1000000,
                                  format="%d")
        popularity = st.slider("🔥 Popularity Score", 1.0, 100.0, 50.0, 0.1)

    with col2:
        runtime = st.slider("⏱️ Runtime (minutes)", 60, 240, 120, 1)
        vote_average = st.slider("⭐ Expected Vote Average", 1.0, 10.0, 6.5, 0.1)

    st.markdown("---")

    if st.button("🎬 PREDICT SUCCESS", use_container_width=True):
        input_data = pd.DataFrame({
            'budget': [budget],
            'popularity': [popularity],
            'runtime': [runtime],
            'vote_average': [vote_average]
        })

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        if prediction == 1:
            st.markdown("""
            <div style='background:linear-gradient(135deg, #0d2e0d, #1a4d1a);
                        border:2px solid #2ecc71; border-radius:16px;
                        padding:30px; text-align:center;'>
                <h1 style='color:#2ecc71; font-size:48px;'>🎉 SUCCESS!</h1>
                <p style='color:#ccc; font-size:18px;'>This movie is predicted to be a BOX OFFICE HIT!</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:linear-gradient(135deg, #2e0d0d, #4d1a1a);
                        border:2px solid #e94560; border-radius:16px;
                        padding:30px; text-align:center;'>
                <h1 style='color:#e94560; font-size:48px;'>❌ FAILURE</h1>
                <p style='color:#ccc; font-size:18px;'>This movie may struggle to recoup its budget.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("✅ Success Probability", f"{probability[1]*100:.1f}%")
        with col2:
            st.metric("❌ Failure Probability", f"{probability[0]*100:.1f}%")

        # Probability gauge
        fig, ax = plt.subplots(figsize=(10, 3))
        fig.patch.set_facecolor('#0d0d0d')
        ax.set_facecolor('#0d0d0d')
        ax.barh(['Probability'], [probability[0]], color='#e94560', label='Failure')
        ax.barh(['Probability'], [probability[1]], left=[probability[0]],
                color='#2ecc71', label='Success')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Probability', color='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#444')
        ax.spines['left'].set_color('#444')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(facecolor='#0d0d0d', labelcolor='white')
        ax.set_title('Success vs Failure Probability', color='white', fontsize=12)
        st.pyplot(fig)
        plt.close()

        st.markdown("---")
        st.markdown("### 💡 Key Factors in this Prediction:")
        for _, row in importance.iterrows():
            st.markdown(f"- **{row['Feature']}**: importance score {row['Importance']:.3f}")

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#444; font-size:12px;'>
    🎬 MovieIQ | Predictive Analytics on Film Success | Data Analytics Internship Project | 2026
</div>
""", unsafe_allow_html=True)
