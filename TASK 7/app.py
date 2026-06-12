import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Fraud Detection AI", page_icon="🛡️", layout="wide")

# ============================================
# BLACK BACKGROUND + DARK BLUE + LIGHT GLOW
# ============================================
st.markdown("""
<style>
    /* Pure Black Background */
    .stApp {
        background: #000000;
    }
    
    /* Light Glow Animation */
    @keyframes softGlow {
        0% { box-shadow: 0 0 5px rgba(0, 100, 200, 0.3); }
        50% { box-shadow: 0 0 15px rgba(0, 100, 200, 0.6); }
        100% { box-shadow: 0 0 5px rgba(0, 100, 200, 0.3); }
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes gentleFloat {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-3px); }
    }
    
    /* Main Header */
    .main-header {
        text-align: center;
        padding: 30px;
        background: #0a0a1a;
        border-radius: 20px;
        margin-bottom: 30px;
        border: 1px solid #1a2a4a;
        animation: fadeInUp 0.6s ease-out;
    }
    
    .main-title {
        font-size: 46px;
        font-weight: 700;
        color: #4a9eff;
        letter-spacing: -1px;
        text-shadow: 0 0 20px rgba(74, 158, 255, 0.3);
    }
    
    .main-subtitle {
        font-size: 13px;
        letter-spacing: 3px;
        color: #6688aa;
        margin-top: 8px;
    }
    
    .divider {
        width: 60px;
        height: 2px;
        background: #2a4a7a;
        margin: 15px auto;
    }
    
    /* Premium Metric Cards - Dark Blue with Light Glow */
    .metric-card {
        background: #0a0a1a;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        border: 1px solid #1a2a4a;
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease-out;
    }
    
    .metric-card:hover {
        border-color: #4a9eff;
        animation: softGlow 1.5s infinite;
        transform: translateY(-3px);
    }
    
    .metric-icon {
        font-size: 38px;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 34px;
        font-weight: 700;
        color: #4a9eff;
        margin: 8px 0;
    }
    
    .metric-label {
        font-size: 11px;
        color: #6688aa;
        letter-spacing: 1.5px;
    }
    
    /* Glass Cards - Dark Blue */
    .glass-card {
        background: #0a0a1a;
        border-radius: 20px;
        padding: 22px;
        border: 1px solid #1a2a4a;
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease-out;
    }
    
    .glass-card:hover {
        border-color: #4a9eff;
        box-shadow: 0 0 20px rgba(74, 158, 255, 0.1);
    }
    
    /* Result Cards */
    .result-safe {
        background: #0a1a0a;
        border: 1px solid #2a6a3a;
        border-radius: 24px;
        padding: 35px;
        text-align: center;
        animation: gentleFloat 3s ease-in-out infinite;
    }
    
    .result-fraud {
        background: #1a0a0a;
        border: 1px solid #7a2a2a;
        border-radius: 24px;
        padding: 35px;
        text-align: center;
        animation: softGlow 2s infinite;
    }
    
    /* Sidebar - Dark Blue */
    [data-testid="stSidebar"] {
        background: #050510;
        border-right: 1px solid #1a2a4a;
    }
    
    [data-testid="stSidebar"] * {
        color: #aaccff;
    }
    
    /* Navigation - Dark Blue Style */
    .stRadio label {
        background: #0a0a1a;
        padding: 10px 18px;
        border-radius: 30px;
        border: 1px solid #1a2a4a;
        transition: all 0.3s;
        font-weight: 500;
        color: #88aacc;
    }
    
    .stRadio label:hover {
        border-color: #4a9eff;
        background: #0f0f2a;
        transform: translateX(5px);
        color: #4a9eff;
    }
    
    /* Buttons - Light Blue Glow */
    .stButton > button {
        background: #0a0a1a;
        color: #4a9eff;
        border: 1px solid #2a4a7a;
        border-radius: 30px;
        padding: 12px;
        font-weight: 600;
        font-size: 15px;
        letter-spacing: 1px;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        border-color: #4a9eff;
        color: #4a9eff;
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(74, 158, 255, 0.3);
    }
    
    /* Headers */
    h2, h3 {
        color: #4a9eff !important;
        font-weight: 600 !important;
        border-left: 3px solid #4a9eff;
        padding-left: 15px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #446688;
        border-top: 1px solid #1a2a4a;
        margin-top: 40px;
        font-size: 10px;
        letter-spacing: 2px;
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: #4a9eff;
    }
    
    /* Info Box */
    .stAlert {
        background: #0a0a1a;
        border: 1px solid #2a4a7a;
        color: #88aacc;
    }
    
    /* Slider */
    .stSlider > div {
        color: #4a9eff;
    }
    
    /* Selectbox */
    .stSelectbox > div {
        background: #0a0a1a;
        border: 1px solid #1a2a4a;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <div class="main-title">🛡️ Fraud Detection AI</div>
    <div class="main-subtitle">INTELLIGENT TRANSACTION MONITORING</div>
    <div class="divider"></div>
</div>
""", unsafe_allow_html=True)

# ============================================
# CREATE SAMPLE DATA
# ============================================
@st.cache_data
def create_sample_data():
    np.random.seed(42)
    n_samples = 10000
    n_fraud = int(n_samples * 0.01)
    
    normal = pd.DataFrame({
        'V1': np.random.normal(0, 1, n_samples - n_fraud),
        'V2': np.random.normal(0, 1, n_samples - n_fraud),
        'V3': np.random.normal(0, 1, n_samples - n_fraud),
        'V4': np.random.normal(0, 1, n_samples - n_fraud),
        'V5': np.random.normal(0, 1, n_samples - n_fraud),
        'V6': np.random.normal(0, 1, n_samples - n_fraud),
        'V7': np.random.normal(0, 1, n_samples - n_fraud),
        'V8': np.random.normal(0, 1, n_samples - n_fraud),
        'V9': np.random.normal(0, 1, n_samples - n_fraud),
        'V10': np.random.normal(0, 1, n_samples - n_fraud),
        'V11': np.random.normal(0, 1, n_samples - n_fraud),
        'V12': np.random.normal(0, 1, n_samples - n_fraud),
        'V13': np.random.normal(0, 1, n_samples - n_fraud),
        'V14': np.random.normal(0, 1, n_samples - n_fraud),
        'Amount': np.random.uniform(10, 500, n_samples - n_fraud),
        'Class': 0
    })
    
    fraud = pd.DataFrame({
        'V1': np.random.normal(-3, 2, n_fraud),
        'V2': np.random.normal(-2, 2, n_fraud),
        'V3': np.random.normal(2, 2, n_fraud),
        'V4': np.random.normal(-1, 2, n_fraud),
        'V5': np.random.normal(1, 2, n_fraud),
        'V6': np.random.normal(-1, 2, n_fraud),
        'V7': np.random.normal(0.5, 2, n_fraud),
        'V8': np.random.normal(-0.5, 2, n_fraud),
        'V9': np.random.normal(0.5, 2, n_fraud),
        'V10': np.random.normal(-0.5, 2, n_fraud),
        'V11': np.random.normal(0.5, 2, n_fraud),
        'V12': np.random.normal(-0.5, 2, n_fraud),
        'V13': np.random.normal(0.5, 2, n_fraud),
        'V14': np.random.normal(-0.5, 2, n_fraud),
        'Amount': np.random.uniform(500, 3000, n_fraud),
        'Class': 1
    })
    
    df = pd.concat([normal, fraud], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)
    return df

df = create_sample_data()

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 Control Panel")
    st.markdown("---")
    
    page = st.radio("", [
        "📊 Dashboard",
        "📈 Model Performance",
        "🔬 Feature Insights",
        "🚨 Transaction Scanner"
    ])
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    test_size = st.slider("Test Split", 10, 40, 30) / 100
    model_choice = st.selectbox("ML Model", ["Random Forest", "Logistic Regression", "Decision Tree", "Gradient Boosting"])
    
    st.markdown("---")
    fraud_count = len(df[df['Class'] == 1])
    st.info(f"""
    📊 Dataset Stats
    ├─ Total: {len(df):,}
    ├─ Fraud: {fraud_count}
    ├─ Normal: {len(df)-fraud_count:,}
    └─ Fraud %: {fraud_count/len(df)*100:.2f}%
    """)

# Prepare data
X = df.drop('Class', axis=1)
y = df['Class']
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    results[name] = {
        'model': model,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }

# ============================================
# DASHBOARD PAGE
# ============================================
if page == "📊 Dashboard":
    st.markdown("## 📊 Live Dashboard")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">💳</div>
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">Total Transactions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🚨</div>
            <div class="metric-value" style="color: #ff6666;">{fraud_count}</div>
            <div class="metric-label">Fraud Detected</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">✅</div>
            <div class="metric-value">{len(df)-fraud_count:,}</div>
            <div class="metric-label">Legitimate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        best_acc = max(results.values(), key=lambda x: x['accuracy'])['accuracy']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🎯</div>
            <div class="metric-value">{best_acc:.1%}</div>
            <div class="metric-label">AI Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig1 = px.pie(values=[fraud_count, len(df)-fraud_count], 
                     names=['Fraud', 'Legitimate'],
                     title="Transaction Distribution",
                     color_discrete_sequence=['#ff6666', '#4a9eff'],
                     template='plotly_dark')
        fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          title_font_color='#4a9eff', font_color='#88aacc')
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig2 = px.histogram(df, x='Amount', nbins=50, log_y=True,
                           title="Amount Distribution",
                           color_discrete_sequence=['#4a9eff'],
                           template='plotly_dark')
        fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          title_font_color='#4a9eff', font_color='#88aacc')
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# MODEL PERFORMANCE PAGE
# ============================================
elif page == "📈 Model Performance":
    st.markdown("## 📈 Model Performance")
    st.markdown("---")
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    perf_df = pd.DataFrame([{
        'Model': name,
        'Accuracy': f"{res['accuracy']:.2%}",
        'Precision': f"{res['precision']:.2%}",
        'Recall': f"{res['recall']:.2%}",
        'F1 Score': f"{res['f1']:.2%}"
    } for name, res in results.items()])
    st.dataframe(perf_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### ROC Curves")
    
    fig_roc = go.Figure()
    for name, res in results.items():
        model = res['model']
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} (AUC={res['roc_auc']:.3f})", line=dict(width=2)))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash', color='gray')))
    fig_roc.update_layout(title="ROC Curves Comparison",
                         plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                         title_font_color='#4a9eff', font_color='#88aacc',
                         height=450)
    st.plotly_chart(fig_roc, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown(f"### Confusion Matrix - {model_choice}")
    selected_model = results[model_choice]['model']
    y_pred = selected_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    
    fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                       title=f"Confusion Matrix",
                       labels={'x': 'Predicted', 'y': 'Actual'},
                       color_continuous_scale='Blues',
                       template='plotly_dark')
    fig_cm.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        title_font_color='#4a9eff', height=400)
    st.plotly_chart(fig_cm, use_container_width=True)
    
    st.success(f"🏆 {model_choice} | Accuracy: {results[model_choice]['accuracy']:.2%} | F1: {results[model_choice]['f1']:.2%}")

# ============================================
# FEATURE INSIGHTS PAGE
# ============================================
elif page == "🔬 Feature Insights":
    st.markdown("## 🔬 Feature Insights")
    st.markdown("---")
    
    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Feature Importance")
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_scaled, y_train)
        importances = rf.feature_importances_
        
        imp_df = pd.DataFrame({'Feature': feature_names[:10], 'Importance': importances[:10]})
        imp_df = imp_df.sort_values('Importance', ascending=True)
        
        fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                        title="Top Features",
                        color='Importance', color_continuous_scale='Blues',
                        template='plotly_dark')
        fig_imp.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                             title_font_color='#4a9eff', height=500)
        st.plotly_chart(fig_imp, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_f2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Pattern Analysis")
        
        fraud_mean = X[y == 1].mean()
        normal_mean = X[y == 0].mean()
        categories = X.columns[:6].tolist()
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=[fraud_mean[cat] for cat in categories], theta=categories, fill='toself', name='Fraud', line=dict(color='#ff6666', width=2)))
        fig_radar.add_trace(go.Scatterpolar(r=[normal_mean[cat] for cat in categories], theta=categories, fill='toself', name='Normal', line=dict(color='#4a9eff', width=2)))
        fig_radar.update_layout(title="Fraud vs Normal Patterns",
                               polar=dict(bgcolor='rgba(0,0,0,0)'),
                               plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                               title_font_color='#4a9eff', height=500)
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# TRANSACTION SCANNER PAGE
# ============================================
else:
    st.markdown("## 🚨 Transaction Scanner")
    st.markdown("---")
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### Enter Transaction Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        v1 = st.slider("V1", -5.0, 5.0, 0.0, 0.1)
        v2 = st.slider("V2", -5.0, 5.0, 0.0, 0.1)
        v3 = st.slider("V3", -5.0, 5.0, 0.0, 0.1)
        v4 = st.slider("V4", -5.0, 5.0, 0.0, 0.1)
        v5 = st.slider("V5", -5.0, 5.0, 0.0, 0.1)
    
    with col2:
        v6 = st.slider("V6", -5.0, 5.0, 0.0, 0.1)
        v7 = st.slider("V7", -5.0, 5.0, 0.0, 0.1)
        v8 = st.slider("V8", -5.0, 5.0, 0.0, 0.1)
        v9 = st.slider("V9", -5.0, 5.0, 0.0, 0.1)
        v10 = st.slider("V10", -5.0, 5.0, 0.0, 0.1)
    
    with col3:
        v11 = st.slider("V11", -5.0, 5.0, 0.0, 0.1)
        v12 = st.slider("V12", -5.0, 5.0, 0.0, 0.1)
        v13 = st.slider("V13", -5.0, 5.0, 0.0, 0.1)
        v14 = st.slider("V14", -5.0, 5.0, 0.0, 0.1)
        amount = st.number_input("Amount", min_value=0.0, max_value=5000.0, value=100.0)
    
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        if st.button("Test Fraud", use_container_width=True):
            st.session_state['v1'] = -3.0
            st.session_state['v2'] = -2.0
            st.session_state['v3'] = 2.5
            st.session_state['amount'] = 2500
            st.rerun()
    with col_btn2:
        if st.button("Test Normal", use_container_width=True):
            st.session_state['v1'] = 0.5
            st.session_state['v2'] = -0.3
            st.session_state['v3'] = 0.2
            st.session_state['amount'] = 100
            st.rerun()
    with col_btn3:
        if st.button("Test Suspicious", use_container_width=True):
            st.session_state['v1'] = -1.5
            st.session_state['v2'] = -1.0
            st.session_state['v3'] = 1.2
            st.session_state['amount'] = 800
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("Analyze Transaction", use_container_width=True):
        input_data = pd.DataFrame([[
            st.session_state.get('v1', v1),
            st.session_state.get('v2', v2),
            st.session_state.get('v3', v3),
            st.session_state.get('v4', v4),
            st.session_state.get('v5', v5),
            st.session_state.get('v6', v6),
            st.session_state.get('v7', v7),
            st.session_state.get('v8', v8),
            st.session_state.get('v9', v9),
            st.session_state.get('v10', v10),
            st.session_state.get('v11', v11),
            st.session_state.get('v12', v12),
            st.session_state.get('v13', v13),
            st.session_state.get('v14', v14),
            st.session_state.get('amount', amount)
        ]], columns=feature_names)
        
        input_scaled = scaler.transform(input_data)
        selected_model = results[model_choice]['model']
        prediction = selected_model.predict(input_scaled)[0]
        fraud_prob = selected_model.predict_proba(input_scaled)[0][1]
        
        st.markdown("---")
        
        if prediction == 1:
            st.markdown(f"""
            <div class="result-fraud">
                <div style="font-size: 60px;">⚠️🚨⚠️</div>
                <div style="font-size: 28px; font-weight: bold; color: #ff6666;">Fraud Alert!</div>
                <div style="font-size: 46px; font-weight: bold;">Risk: {fraud_prob:.1%}</div>
                <div style="margin-top: 15px;">🔴 Block this transaction immediately</div>
                <div style="margin-top: 10px;">🤖 {model_choice}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
                <div style="font-size: 60px;">✅🛡️✅</div>
                <div style="font-size: 28px; font-weight: bold; color: #4a9eff;">Transaction Safe</div>
                <div style="font-size: 46px; font-weight: bold;">Risk: {fraud_prob:.1%}</div>
                <div style="margin-top: 15px;">🟢 Transaction approved</div>
                <div style="margin-top: 10px;">🤖 {model_choice}</div>
            </div>
            """, unsafe_allow_html=True)
        
        risk_level = int(fraud_prob * 100)
        if risk_level < 30:
            st.progress(risk_level, text=f"Risk Level: {risk_level}% - Low")
        elif risk_level < 70:
            st.progress(risk_level, text=f"Risk Level: {risk_level}% - Medium")
        else:
            st.progress(risk_level, text=f"Risk Level: {risk_level}% - High")

# Footer
st.markdown("""
<div class="footer">
    🛡️ FRAUD DETECTION AI | AI-POWERED TRANSACTION MONITORING
</div>
""", unsafe_allow_html=True)