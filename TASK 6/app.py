import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍇", layout="wide")

# CSS
st.markdown("""
<style>
    .stApp { background: linear-gradient(145deg, #2B1B17 0%, #3D2B1F 50%, #2B1B17 100%); }
    @keyframes grapeBounce { 0%, 100% { transform: translateY(0px); } 50% { transform: translateY(-8px); } }
    
    .title-section {
        text-align: center;
        padding: 30px 20px;
        background: linear-gradient(135deg, rgba(139, 69, 19, 0.3), rgba(101, 67, 33, 0.2));
        border-radius: 60px 60px 60px 20px;
        border: 1px solid rgba(218, 165, 32, 0.3);
        margin-bottom: 25px;
    }
    
    .main-title {
        font-size: 58px;
        font-weight: 800;
        background: linear-gradient(135deg, #DAA520, #CD853F, #8B4513);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .subtitle { font-size: 14px; letter-spacing: 6px; color: #DAA520; font-weight: 500; }
    .vintage-line { width: 100px; height: 3px; background: linear-gradient(90deg, #DAA520, #8B4513, #DAA520); margin: 18px auto; }
    
    .wine-card {
        background: rgba(139, 69, 19, 0.2);
        backdrop-filter: blur(10px);
        border-radius: 30px 30px 30px 15px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(218, 165, 32, 0.3);
        transition: all 0.4s ease;
    }
    .wine-card:hover { transform: translateY(-8px); border-color: #DAA520; }
    .wine-icon { font-size: 42px; }
    .wine-value { font-size: 34px; font-weight: 800; color: #DAA520; margin: 8px 0; }
    .wine-label { font-size: 11px; color: #CD853F; letter-spacing: 2px; }
    
    .barrel-card {
        background: rgba(101, 67, 33, 0.4);
        backdrop-filter: blur(12px);
        border-radius: 40px 20px 40px 20px;
        padding: 20px;
        border: 1px solid rgba(218, 165, 32, 0.2);
    }
    
    .result-card {
        background: linear-gradient(135deg, rgba(139, 69, 19, 0.25), rgba(101, 67, 33, 0.15));
        border-radius: 60px 60px 60px 30px;
        padding: 40px;
        text-align: center;
        border: 1px solid #DAA520;
        animation: grapeBounce 3s ease-in-out infinite;
    }
    
    .result-quality { font-size: 54px; font-weight: 800; background: linear-gradient(135deg, #DAA520, #CD853F); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #2B1B17 0%, #1A0F0A 100%); border-right: 1px solid rgba(218, 165, 32, 0.2); }
    
    .stRadio label {
        background: rgba(139, 69, 19, 0.3);
        padding: 10px 20px;
        border-radius: 40px;
        border: 1px solid rgba(218, 165, 32, 0.3);
    }
    .stRadio label:hover { border-color: #DAA520; background: rgba(218, 165, 32, 0.15); }
    
    .stButton > button {
        background: linear-gradient(135deg, #8B4513, #DAA520);
        color: white;
        border: none;
        border-radius: 40px;
        padding: 12px;
        font-weight: 700;
    }
    .stButton > button:hover { transform: scale(1.02); }
    
    h2, h3 { color: #DAA520 !important; border-left: 3px solid #CD853F; padding-left: 15px; }
    .footer { text-align: center; padding: 20px; color: #8B7355; border-top: 1px solid rgba(218, 165, 32, 0.15); margin-top: 40px; }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
<div class="title-section">
    <div class="main-title">🍇 Wine Quality Predictor</div>
    <div class="subtitle">VINTAGE ESTATE | PRECISION WINE CLASSIFICATION</div>
    <div class="vintage-line"></div>
</div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\wine_quality\WineQT.csv")
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df

df = load_data()

# Sidebar
with st.sidebar:
    st.markdown("## 🍷 CELLAR MENU")
    st.markdown("---")
    
    page = st.radio("", [
        "🏰 Wine Cellar",
        "📜 Grand Tasting",
        "⚗️ Vintage Analysis",
        "🍇 Sommelier AI"
    ])
    
    st.markdown("---")
    st.markdown("### ⚙️ Sommelier Settings")
    test_size = st.slider("Tasting Split", 10, 40, 20) / 100
    model_choice = st.selectbox("Expert Model", ["Random Forest", "Gradient Boosting", "SVM", "KNN", "SGD Classifier"])
    
    st.markdown("---")
    st.markdown("### 📜 Vintage Notes")
    st.info(f"🍇 Grand Crus: {len(df):,}\n🏺 Features: {len(df.columns)-1}\n⭐ Rating Scale: 3-8")

# Data prep
feature_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                'pH', 'sulphates', 'alcohol']

available_features = [col for col in feature_cols if col in df.columns]
X = df[available_features]
y = df['quality']
# Create multi-class classification (3 classes)
y_class = pd.cut(y, bins=[0, 5.5, 6.5, 10], labels=['Standard', 'Good', 'Premium'])

X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=test_size, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

feature_columns = X.columns.tolist()

# Train models
models_dict = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SGD Classifier": SGDClassifier(random_state=42)
}

# Train and store models with their accuracy
model_results = {}
for name, model in models_dict.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    model_results[name] = {
        'model': model,
        'accuracy': acc
    }

# ============================================
# WINE CELLAR PAGE
# ============================================
if page == "🏰 Wine Cellar":
    st.markdown("## 🏰 Grand Wine Cellar")
    st.markdown("---")
    
    premium_wines = len(df[df['quality'] >= 7])
    good_wines = len(df[(df['quality'] >= 6) & (df['quality'] < 7)])
    standard_wines = len(df[df['quality'] < 6])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="wine-card">
            <div class="wine-icon">🍇</div>
            <div class="wine-value">{len(df):,}</div>
            <div class="wine-label">Total Wines</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="wine-card">
            <div class="wine-icon">🏆</div>
            <div class="wine-value">{premium_wines}</div>
            <div class="wine-label">Premium (7-8)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="wine-card">
            <div class="wine-icon">⭐</div>
            <div class="wine-value">{good_wines}</div>
            <div class="wine-label">Good (6)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        best_acc = max(model_results.values(), key=lambda x: x['accuracy'])['accuracy']
        st.markdown(f"""
        <div class="wine-card">
            <div class="wine-icon">🎯</div>
            <div class="wine-value">{best_acc:.1%}</div>
            <div class="wine-label">Best Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown('<div class="barrel-card">', unsafe_allow_html=True)
        quality_counts = df['quality'].value_counts().sort_index()
        fig1 = px.bar(x=quality_counts.index, y=quality_counts.values,
                     title="Quality Distribution",
                     labels={'x': 'Quality Score', 'y': 'Count'},
                     color=quality_counts.index,
                     color_continuous_scale='Viridis',
                     template='plotly_dark')
        fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          title_font_color='#DAA520')
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_b:
        st.markdown('<div class="barrel-card">', unsafe_allow_html=True)
        fig2 = px.pie(values=[premium_wines, good_wines, standard_wines], 
                     names=['Premium (7-8)', 'Good (6)', 'Standard (3-5)'],
                     title="Wine Classification",
                     color_discrete_sequence=['#DAA520', '#CD853F', '#8B4513'],
                     template='plotly_dark')
        fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          title_font_color='#DAA520')
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown('<div class="barrel-card">', unsafe_allow_html=True)
    st.markdown("### 🍷 Sommelier's Notes - Feature Correlations")
    correlations = df.corr()['quality'].sort_values(ascending=False)
    
    fig3 = px.bar(x=correlations.index[1:], y=correlations.values[1:],
                 title="Feature Influence on Wine Quality",
                 labels={'x': 'Chemical Property', 'y': 'Correlation'},
                 color=correlations.values[1:],
                 color_continuous_scale='Viridis',
                 template='plotly_dark')
    fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      title_font_color='#DAA520', height=500)
    st.plotly_chart(fig3, use_container_width=True)
    
    st.info("🍷 **Sommelier Insight:** Alcohol shows strongest positive correlation. Lower volatile acidity = better quality.")
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# GRAND TASTING PAGE
# ============================================
elif page == "📜 Grand Tasting":
    st.markdown("## 📜 Grand Tasting - Expert Models")
    st.markdown("---")
    
    comp_data = []
    for name, res in model_results.items():
        comp_data.append({
            'Expert Model': name,
            'Accuracy': f"{res['accuracy']:.2%}"
        })
    
    st.markdown('<div class="barrel-card">', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown('<div class="barrel-card">', unsafe_allow_html=True)
    acc_data = [(name, res['accuracy']) for name, res in model_results.items()]
    fig_acc = px.bar(x=[x[0] for x in acc_data], y=[x[1] for x in acc_data],
                    title="Expert Model Accuracy Comparison",
                    labels={'x': 'Model', 'y': 'Accuracy'},
                    color=[x[1] for x in acc_data],
                    color_continuous_scale='Viridis',
                    template='plotly_dark')
    fig_acc.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                         title_font_color='#DAA520', height=450)
    st.plotly_chart(fig_acc, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# VINTAGE ANALYSIS PAGE
# ============================================
elif page == "⚗️ Vintage Analysis":
    st.markdown("## ⚗️ Vintage Chemical Analysis")
    st.markdown("---")
    
    st.markdown('<div class="barrel-card">', unsafe_allow_html=True)
    
    feature = st.selectbox("Select Chemical Property", df.columns[:-1])
    
    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        fig1 = px.histogram(df, x=feature, nbins=30,
                           title=f"Distribution of {feature.replace('_', ' ').title()}",
                           color_discrete_sequence=['#DAA520'],
                           template='plotly_dark')
        fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          title_font_color='#DAA520')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col_f2:
        fig2 = px.box(df, x='quality', y=feature,
                     title=f"{feature.replace('_', ' ').title()} by Quality",
                     color='quality',
                     template='plotly_dark')
        fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          title_font_color='#DAA520')
        st.plotly_chart(fig2, use_container_width=True)
    
    corr_value = df.corr()[feature]['quality']
    
    if feature == 'alcohol':
        st.success(f"🍷 **Sommelier Note:** {feature} has **positive correlation** ({corr_value:.2f}) with quality. Higher alcohol = better wine.")
    elif feature == 'volatile acidity':
        st.warning(f"⚠️ **Sommelier Note:** {feature} has **negative correlation** ({corr_value:.2f}) with quality. Lower is better.")
    else:
        st.info(f"📊 **Correlation with quality:** {corr_value:.2f}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# SOMMELIER AI PAGE
# ============================================
else:
    st.markdown("## 🍇 Sommelier AI - Wine Quality Prediction")
    st.markdown("---")
    
    st.markdown('<div class="barrel-card">', unsafe_allow_html=True)
    st.markdown("### 📝 Enter Wine Chemical Properties")
    st.caption("💡 **Pro Tip:** High Alcohol (>11.5%) + Low Volatile Acidity (<0.4) = Premium Wine!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 8.0, 0.1, help="Higher values give tartness")
        volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.6, 0.5, 0.01, help="Lower is better for quality")
        citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.3, 0.01, help="Adds freshness")
        residual_sugar = st.slider("Residual Sugar", 0.0, 16.0, 2.5, 0.1)
        chlorides = st.slider("Chlorides", 0.01, 0.15, 0.05, 0.01)
    
    with col2:
        free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1, 80, 15, 1)
        total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 6, 300, 40, 5)
        density = st.slider("Density", 0.990, 1.005, 0.997, 0.001)
        pH = st.slider("pH", 2.8, 4.2, 3.3, 0.05)
        sulphates = st.slider("Sulphates", 0.3, 2.0, 0.6, 0.05, help="Higher = better preservation")
        alcohol = st.slider("Alcohol (%)", 8.0, 15.0, 10.5, 0.1, help="Higher = better quality")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("🍷 TASTE PREDICTION 🍷", use_container_width=True):
        input_dict = {
            'fixed acidity': fixed_acidity,
            'volatile acidity': volatile_acidity,
            'citric acid': citric_acid,
            'residual sugar': residual_sugar,
            'chlorides': chlorides,
            'free sulfur dioxide': free_sulfur_dioxide,
            'total sulfur dioxide': total_sulfur_dioxide,
            'density': density,
            'pH': pH,
            'sulphates': sulphates,
            'alcohol': alcohol
        }
        
        input_data = pd.DataFrame([input_dict])
        input_data = input_data[feature_columns]
        input_scaled = scaler.transform(input_data)
        
        # Get the selected model
        selected_model = model_results[model_choice]['model']
        prediction = selected_model.predict(input_scaled)[0]
        
        if prediction == 'Premium':
            quality_text = "🏆 PREMIER CRU WINE 🏆"
            color = "#DAA520"
            icon = "🍷✨"
            description = "Exceptional quality! Outstanding characteristics. Highly recommended for connoisseurs."
        elif prediction == 'Good':
            quality_text = "⭐ GOOD QUALITY WINE ⭐"
            color = "#CD853F"
            icon = "🍷"
            description = "Good quality wine meeting premium standards. A reliable choice."
        else:
            quality_text = "🍇 STANDARD VILLAGE WINE 🍇"
            color = "#8B4513"
            icon = "🍇"
            description = "Standard quality wine suitable for casual occasions. Good value."
        
        st.markdown(f"""
        <div class="result-card">
            <div style="font-size: 48px;">{icon}</div>
            <div class="result-label">🍷 SOMMELIER'S VERDICT</div>
            <div class="result-quality" style="color: {color};">{quality_text}</div>
            <div class="result-model" style="margin-top: 15px;">🤖 {model_choice}</div>
            <div style="margin-top: 15px; font-size: 14px; color: #CD853F;">{description}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show tips
        st.markdown("---")
        st.markdown("#### 📊 How to Get Better Results")
        
        col_tip1, col_tip2, col_tip3 = st.columns(3)
        with col_tip1:
            if alcohol >= 11.5:
                st.success(f"✅ Alcohol: {alcohol}% - Excellent!")
            else:
                st.info(f"💡 Tip: Increase alcohol to >11.5% (currently {alcohol}%)")
        
        with col_tip2:
            if volatile_acidity <= 0.4:
                st.success(f"✅ Volatile Acidity: {volatile_acidity} - Excellent!")
            else:
                st.info(f"💡 Tip: Decrease volatile acidity to <0.4 (currently {volatile_acidity})")
        
        with col_tip3:
            if sulphates >= 0.7:
                st.success(f"✅ Sulphates: {sulphates} - Good level!")
            else:
                st.info(f"💡 Tip: Increase sulphates to >0.7 (currently {sulphates})")

# Footer
st.markdown("""
<div class="footer">
    🍇 WINE QUALITY PREDICTOR | VINTAGE ESTATE | SOMMELIER-GRADE AI 🍷
</div>
""", unsafe_allow_html=True)