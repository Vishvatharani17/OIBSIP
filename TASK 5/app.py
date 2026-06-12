import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import folium
from streamlit_folium import folium_static
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="HOUSE PRICE PREDICTOR", page_icon="🏠", layout="wide")

# ============================================
# ELEGANT COLOR SCHEME CSS
# ============================================
st.markdown("""
<style>
    /* Main Background - Deep Navy Blue */
    .stApp {
        background: linear-gradient(145deg, #0B1120 0%, #0F172A 50%, #0B1120 100%);
    }
    
    /* Smooth Animations */
    @keyframes softGlow {
        0%, 100% { opacity: 0.7; }
        50% { opacity: 1; }
    }
    
    @keyframes gentleFloat {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
    }
    
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    /* Title Section */
    .title-section {
        text-align: center;
        padding: 25px 20px;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(139, 92, 246, 0.05));
        border-radius: 40px;
        margin-bottom: 20px;
    }
    
    .main-title {
        font-size: 56px;
        font-weight: 800;
        background: linear-gradient(135deg, #60A5FA, #A78BFA, #F472B6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
        margin-bottom: 8px;
    }
    
    .subtitle {
        font-size: 14px;
        letter-spacing: 4px;
        color: #6B7280;
        font-weight: 500;
    }
    
    .elegant-line {
        width: 80px;
        height: 3px;
        background: linear-gradient(90deg, #60A5FA, #A78BFA);
        margin: 18px auto;
        border-radius: 3px;
    }
    
    /* Premium Metric Cards */
    .metric-card {
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px 12px;
        text-align: center;
        border: 1px solid rgba(96, 165, 250, 0.2);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(96, 165, 250, 0.5);
        background: rgba(15, 23, 42, 0.9);
        box-shadow: 0 20px 25px -12px rgba(0, 0, 0, 0.3);
    }
    
    .metric-icon {
        font-size: 38px;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #60A5FA;
        margin: 5px 0;
    }
    
    .metric-label {
        font-size: 12px;
        color: #9CA3AF;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    /* Glass Effect Cards */
    .glass-card {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(12px);
        border-radius: 24px;
        padding: 20px;
        border: 1px solid rgba(96, 165, 250, 0.15);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(96, 165, 250, 0.35);
        background: rgba(15, 23, 42, 0.7);
    }
    
    /* Prediction Result */
    .result-card {
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.12), rgba(139, 92, 246, 0.08));
        border-radius: 32px;
        padding: 35px;
        text-align: center;
        border: 1px solid rgba(96, 165, 250, 0.3);
        animation: gentleFloat 3s ease-in-out infinite;
    }
    
    .result-label {
        font-size: 13px;
        letter-spacing: 3px;
        color: #60A5FA;
        margin-bottom: 10px;
    }
    
    .result-price {
        font-size: 58px;
        font-weight: 800;
        background: linear-gradient(135deg, #60A5FA, #A78BFA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }
    
    .result-model {
        display: inline-block;
        background: rgba(96, 165, 250, 0.15);
        padding: 6px 20px;
        border-radius: 30px;
        font-size: 12px;
        color: #A78BFA;
        margin-top: 15px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(11, 17, 32, 0.95);
        border-right: 1px solid rgba(96, 165, 250, 0.15);
    }
    
    [data-testid="stSidebar"] * {
        color: #D1D5DB;
    }
    
    /* Navigation Radio Buttons */
    .stRadio > div {
        gap: 8px;
    }
    
    .stRadio label {
        background: rgba(15, 23, 42, 0.6);
        padding: 10px 18px;
        border-radius: 40px;
        border: 1px solid rgba(96, 165, 250, 0.25);
        transition: all 0.3s;
        font-weight: 500;
        font-size: 14px;
    }
    
    .stRadio label:hover {
        border-color: #60A5FA;
        background: rgba(96, 165, 250, 0.1);
        transform: translateX(5px);
    }
    
    .stRadio [data-baseweb="radio"]:checked + div {
        border-color: #60A5FA;
    }
    
    /* Primary Button */
    .stButton > button {
        background: linear-gradient(135deg, #60A5FA, #A78BFA);
        color: white;
        border: none;
        border-radius: 40px;
        padding: 12px;
        font-weight: 600;
        font-size: 15px;
        letter-spacing: 1px;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 20px -5px rgba(96, 165, 250, 0.4);
    }
    
    /* Slider */
    .stSlider > div {
        color: #60A5FA;
    }
    
    /* Selectbox */
    .stSelectbox > div {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(96, 165, 250, 0.2);
        border-radius: 12px;
    }
    
    /* Number Input */
    .stNumberInput > div > div > input {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(96, 165, 250, 0.2);
        border-radius: 12px;
        color: white;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #F3F4F6 !important;
        font-weight: 600 !important;
    }
    
    h2, h3 {
        border-left: 3px solid #60A5FA;
        padding-left: 15px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #6B7280;
        border-top: 1px solid rgba(96, 165, 250, 0.1);
        margin-top: 40px;
        font-size: 11px;
        letter-spacing: 2px;
    }
    
    /* Dataframe */
    .stDataFrame {
        border-radius: 16px;
        overflow: hidden;
    }
    
    /* Divider */
    hr {
        border-color: rgba(96, 165, 250, 0.15);
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0B1120;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #60A5FA, #A78BFA);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
<div class="title-section">
    <div class="main-title">🏠 House Price Predictor</div>
    <div class="subtitle">PRECISION REAL ESTATE VALUATION</div>
    <div class="elegant-line"></div>
</div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\house_price_prediction\housing.csv")
    return df

df = load_data()

# Sidebar
with st.sidebar:
    st.markdown("## Navigation")
    st.markdown("---")
    
    page = st.radio("", [
        "📊 Dashboard",
        "📈 Model Comparison",
        "🗺️ Property Map",
        "🔮 Price Predictor"
    ])
    
    st.markdown("---")
    st.markdown("### Settings")
    test_size = st.slider("Test Size", 10, 40, 20) / 100
    model_choice = st.selectbox("Algorithm", ["Linear Regression", "Ridge Regression", "Lasso Regression"])
    
    st.markdown("---")
    st.markdown("### Dataset Info")
    st.info(f"Properties: {len(df):,}\nFeatures: {len(df.columns)}")

# Data prep
df_clean = df.dropna()
X = df_clean.drop('median_house_value', axis=1)
y = df_clean['median_house_value']
X = pd.get_dummies(X, columns=['ocean_proximity'], drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=1.0)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        'r2': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }

# ============================================
# DASHBOARD PAGE
# ============================================
if page == "📊 Dashboard":
    st.markdown("## Market Overview")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🏘️</div>
            <div class="metric-value">{len(df_clean):,}</div>
            <div class="metric-label">Properties</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">💎</div>
            <div class="metric-value">${df_clean['median_house_value'].mean():,.0f}</div>
            <div class="metric-label">Average Price</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">📈</div>
            <div class="metric-value">{df_clean['median_income'].mean():.2f}</div>
            <div class="metric-label">Median Income</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        best_r2 = max(results.values(), key=lambda x: x['r2'])['r2']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🏆</div>
            <div class="metric-value">{best_r2:.3f}</div>
            <div class="metric-label">Best R² Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig1 = px.histogram(df_clean, x='median_house_value', nbins=50,
                           title="Price Distribution",
                           color_discrete_sequence=['#60A5FA'],
                           template='plotly_dark')
        fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          title_font_color='#60A5FA', font_color='#9CA3AF')
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_b:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig2 = px.scatter(df_clean, x='median_income', y='median_house_value',
                         title="Income vs Price",
                         color_discrete_sequence=['#A78BFA'],
                         template='plotly_dark')
        fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          title_font_color='#60A5FA', font_color='#9CA3AF')
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# MODEL COMPARISON PAGE
# ============================================
elif page == "📈 Model Comparison":
    st.markdown("## Model Performance")
    st.markdown("---")
    
    comp_data = []
    for name, res in results.items():
        comp_data.append({'Model': name, 'R² Score': f"{res['r2']:.4f}", 'RMSE': f"${res['rmse']:,.0f}"})
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    selected_model = models[model_choice]
    selected_model.fit(X_train, y_train)
    y_pred = selected_model.predict(X_test)
    
    fig = px.scatter(x=y_test, y=y_pred, title=f"{model_choice} - Actual vs Predicted",
                     labels={'x': 'Actual Price', 'y': 'Predicted Price'},
                     template='plotly_dark', color_discrete_sequence=['#60A5FA'])
    fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                              mode='lines', name='Perfect Prediction', line=dict(color='#F472B6', dash='dash', width=2)))
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                     title_font_color='#60A5FA', font_color='#9CA3AF', height=500)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# PROPERTY MAP PAGE
# ============================================
elif page == "🗺️ Property Map":
    st.markdown("## Property Locations")
    st.markdown("---")
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    map_data = df_clean.sample(min(300, len(df_clean)))
    center = [map_data['latitude'].mean(), map_data['longitude'].mean()]
    
    m = folium.Map(location=center, zoom_start=6, tiles='CartoDB dark_matter')
    
    for _, row in map_data.iterrows():
        price = row['median_house_value']
        if price < 150000:
            color = '#10B981'
        elif price < 300000:
            color = '#F59E0B'
        else:
            color = '#EF4444'
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.6,
            popup=f"💰 ${price:,.0f}"
        ).add_to(m)
    
    folium_static(m, width=800, height=550)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# PRICE PREDICTOR PAGE
# ============================================
else:
    st.markdown("## Price Predictor")
    st.markdown("---")
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Property Age", 1, 50, 25)
        rooms = st.number_input("Total Rooms", 500, 10000, 2000)
        bedrooms = st.number_input("Bedrooms", 50, 2000, 500)
    
    with col2:
        population = st.number_input("Area Population", 100, 10000, 1000)
        income = st.slider("Median Income ($K)", 1.0, 15.0, 5.0, 0.5)
        ocean = st.selectbox("Ocean Proximity", ['NEAR BAY', 'NEAR OCEAN', 'INLAND', 'ISLAND', '<1H OCEAN'])
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("Predict Price", use_container_width=True):
        input_df = pd.DataFrame({
            'longitude': [-118.0], 'latitude': [34.0], 'housing_median_age': [age],
            'total_rooms': [rooms], 'total_bedrooms': [bedrooms], 'population': [population],
            'households': [population//2], 'median_income': [income]
        })
        
        for cat in ['NEAR BAY', 'NEAR OCEAN', 'INLAND', 'ISLAND', '<1H OCEAN']:
            input_df[f'ocean_proximity_{cat}'] = [1 if ocean == cat else 0]
        
        for col in X.columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df = input_df[X.columns]
        input_scaled = scaler.transform(input_df)
        
        model = models[model_choice]
        model.fit(X_train, y_train)
        pred = model.predict(input_scaled)[0]
        
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">ESTIMATED MARKET VALUE</div>
            <div class="result-price">${pred:,.0f}</div>
            <div class="result-model">⚡ {model_choice}</div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    HOUSE PRICE PREDICTOR | AI-POWERED REAL ESTATE VALUATION
</div>
""", unsafe_allow_html=True)