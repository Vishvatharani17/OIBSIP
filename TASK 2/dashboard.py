import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Page setup
st.set_page_config(page_title="Customer Segmentation Dashboard", page_icon="🎯", layout="wide")

# Custom CSS - Black + Dark Blue + Cyan Theme
st.markdown("""
<style>
    /* Main background - Dark Blue to Black */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #0a0f1a 30%, #0a0a0f 100%);
    }
    
    /* Glowing text animation for main title - Cyan */
    @keyframes glow {
        0% { text-shadow: 0 0 5px #00e5ff, 0 0 10px #00e5ff; }
        50% { text-shadow: 0 0 20px #00e5ff, 0 0 30px #0088aa; }
        100% { text-shadow: 0 0 5px #00e5ff, 0 0 10px #00e5ff; }
    }
    
    .glow-title {
        animation: glow 3s ease-in-out infinite;
        text-align: center;
    }
    
    /* Metric Cards - Black with Cyan Border */
    .metric-card {
        background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 100%);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(0, 229, 255, 0.3);
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border: 1px solid #00e5ff;
        box-shadow: 0 10px 30px rgba(0, 229, 255, 0.2);
    }
    
    .metric-value {
        font-size: 36px;
        font-weight: bold;
        background: linear-gradient(135deg, #00e5ff, #00aaff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        color: #6699cc;
        font-size: 14px;
        letter-spacing: 1px;
    }
    
    /* Navigation Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0d1117, #0a0a0f);
        color: #00e5ff;
        border: 1px solid rgba(0, 229, 255, 0.3);
        border-radius: 25px;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #00e5ff, #00aaff);
        color: #0a0a0f;
        transform: scale(1.02);
        box-shadow: 0 5px 20px rgba(0, 229, 255, 0.4);
        border: 1px solid #00e5ff;
    }
    
    /* Sidebar - Black */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0f 0%, #05080c 100%);
        border-right: 1px solid rgba(0, 229, 255, 0.15);
    }
    
    [data-testid="stSidebar"] * {
        color: #88aacc;
    }
    
    /* Headers - Cyan */
    h1 {
        color: #00e5ff !important;
        font-weight: bold !important;
    }
    
    h2, h3 {
        color: #00d4ff !important;
        border-left: 4px solid #00e5ff;
        padding-left: 15px;
    }
    
    /* Segment Cards - Black with Cyan Border */
    .segment-card {
        background: linear-gradient(135deg, #0d1117 0%, #0a0a0f 100%);
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #00e5ff;
        border-right: 1px solid rgba(0, 229, 255, 0.1);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border: 1px solid rgba(0, 229, 255, 0.2);
        border-radius: 10px;
        background: #0a0a0f;
    }
    
    /* Divider */
    hr {
        border-color: rgba(0, 229, 255, 0.15);
    }
    
    /* Info boxes */
    .stAlert {
        background: rgba(0, 229, 255, 0.05);
        border: 1px solid rgba(0, 229, 255, 0.2);
    }
    
    /* Slider styling */
    .stSlider {
        color: #00e5ff;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #4488aa;
        border-top: 1px solid rgba(0, 229, 255, 0.15);
        margin-top: 30px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 10px;
        border: 1px solid rgba(0, 229, 255, 0.15);
    }
    
    /* Select box */
    .stSelectbox div {
        background: #0a0a0f;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'selected_clusters' not in st.session_state:
    st.session_state.selected_clusters = 5

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\seg_data\mall_customers.csv")
    return df

df = load_data()

# Navigation Sidebar
with st.sidebar:
    st.markdown("## 🎯 NAVIGATION")
    st.markdown("---")
    
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("🏠 HOME", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()
        if st.button("📊 CLUSTER", use_container_width=True):
            st.session_state.page = 'cluster'
            st.rerun()
    with col_nav2:
        if st.button("📈 INSIGHTS", use_container_width=True):
            st.session_state.page = 'segments'
            st.rerun()
        if st.button("📋 DATA", use_container_width=True):
            st.session_state.page = 'data'
            st.rerun()
    
    st.markdown("---")
    
    st.markdown("### ⚙️ SETTINGS")
    st.session_state.selected_clusters = st.slider("Number of Segments (K)", 2, 8, 5)
    
    st.markdown("### 📊 FEATURES")
    use_income = st.checkbox("Annual Income", value=True)
    use_spending = st.checkbox("Spending Score", value=True)
    use_age = st.checkbox("Age", value=False)

# Prepare features
features = []
if use_income:
    features.append('Annual Income (k$)')
if use_spending:
    features.append('Spending Score (1-100)')
if use_age:
    features.append('Age')

if len(features) >= 2:
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=st.session_state.selected_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features)

# ============================================
# HOME PAGE
# ============================================
if st.session_state.page == 'home':
    st.markdown("""
    <div class="glow-title">
        <h1 style="font-size: 52px;">🎯 Customer Segmentation Dashboard</h1>
        <p style="color: #6699cc; font-size: 18px; letter-spacing: 2px;">K-MEANS CLUSTERING | CUSTOMER ANALYTICS</p>
        <div style="height: 3px; background: linear-gradient(90deg, transparent, #00e5ff, #00aaff, #00e5ff, transparent); width: 60%; margin: 20px auto;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 40px;">👥</div>
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">TOTAL CUSTOMERS</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 40px;">💰</div>
            <div class="metric-value">${df['Annual Income (k$)'].mean():.0f}K</div>
            <div class="metric-label">AVG INCOME</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 40px;">⭐</div>
            <div class="metric-value">{df['Spending Score (1-100)'].mean():.0f}</div>
            <div class="metric-label">AVG SPENDING</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 40px;">🎯</div>
            <div class="metric-value">{st.session_state.selected_clusters}</div>
            <div class="metric-label">SEGMENTS</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("### 🥧 Gender Distribution")
        gender_counts = df['Gender'].value_counts()
        fig1 = px.pie(
            values=gender_counts.values,
            names=gender_counts.index,
            title="Customers by Gender",
            color_discrete_sequence=['#00e5ff', '#00aaff'],
            hole=0.4,
            template='plotly_dark'
        )
        fig1.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_color='#00d4ff',
            font_color='#88aacc',
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col_chart2:
        st.markdown("### 📊 Age Distribution")
        fig2 = px.histogram(
            df, x='Age', nbins=15,
            title="Age Distribution of Customers",
            color_discrete_sequence=['#00e5ff'],
            template='plotly_dark'
        )
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_color='#00d4ff',
            font_color='#88aacc',
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    # Quick Actions
    st.markdown("### 🚀 Quick Navigation")
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        if st.button("📊 View Cluster Analysis", use_container_width=True):
            st.session_state.page = 'cluster'
            st.rerun()
    with col_btn2:
        if st.button("📈 Explore Segments", use_container_width=True):
            st.session_state.page = 'segments'
            st.rerun()
    with col_btn3:
        if st.button("📋 See Customer Data", use_container_width=True):
            st.session_state.page = 'data'
            st.rerun()

# ============================================
# CLUSTER PAGE
# ============================================
elif st.session_state.page == 'cluster':
    st.markdown("## 📊 Cluster Analysis")
    st.markdown("*Visualizing customer segments using K-Means clustering*")
    st.markdown("---")
    
    if len(features) >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Customer Segments")
            fig1 = px.scatter(
                df,
                x=features[0],
                y=features[1],
                color=df['Cluster'].astype(str),
                size='Spending Score (1-100)',
                hover_data=['CustomerID', 'Age'],
                title=f"Customer Clusters (K={st.session_state.selected_clusters})",
                color_discrete_sequence=px.colors.qualitative.Set1,
                template='plotly_dark'
            )
            fig1.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_font_color='#00d4ff',
                font_color='#88aacc',
                height=500
            )
            fig1.add_trace(go.Scatter(
                x=centers[features[0]],
                y=centers[features[1]],
                mode='markers',
                marker=dict(size=15, symbol='x', color='#ff4444', line=dict(width=2, color='white')),
                name='Centers'
            ))
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Elbow Method")
            inertias = []
            K_range = range(1, 11)
            for k in K_range:
                kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans_temp.fit(X_scaled)
                inertias.append(kmeans_temp.inertia_)
            
            fig2 = px.line(
                x=list(K_range), y=inertias,
                markers=True,
                title="Finding Optimal K Value",
                labels={'x': 'Number of Clusters (K)', 'y': 'Inertia'},
                template='plotly_dark'
            )
            fig2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_font_color='#00d4ff',
                font_color='#88aacc',
                height=500
            )
            fig2.add_vline(x=st.session_state.selected_clusters, line_dash="dash", line_color="#ff4444")
            st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📊 Cluster Centers Analysis")
        centers_display = centers.copy()
        centers_display.index = [f"Segment {i}" for i in range(st.session_state.selected_clusters)]
        st.dataframe(centers_display.style.background_gradient(cmap='Blues').format("{:.0f}"))

# ============================================
# INSIGHTS PAGE
# ============================================
elif st.session_state.page == 'segments':
    st.markdown("## 📈 Customer Segment Insights")
    st.markdown("*Deep dive into each customer segment*")
    st.markdown("---")
    
    if len(features) >= 2:
        st.markdown("### 🥧 Segment Distribution")
        cluster_counts = df['Cluster'].value_counts().sort_index()
        fig_pie = px.pie(
            values=cluster_counts.values,
            names=[f"Segment {i}" for i in cluster_counts.index],
            title="Market Share by Segment",
            color_discrete_sequence=px.colors.qualitative.Set1,
            hole=0.4,
            template='plotly_dark'
        )
        fig_pie.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_color='#00d4ff',
            font_color='#88aacc',
            height=450
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 🎯 Segment Analysis")
        
        for i in range(st.session_state.selected_clusters):
            cluster_data = df[df['Cluster'] == i]
            percentage = (len(cluster_data) / len(df)) * 100
            avg_spend = cluster_data['Spending Score (1-100)'].mean()
            avg_income = cluster_data['Annual Income (k$)'].mean()
            avg_age = cluster_data['Age'].mean()
            
            if avg_spend > 70 and avg_income > 70:
                icon, label = "💎", "HIGH VALUE"
                rec = "Offer premium products and loyalty programs"
            elif avg_spend > 70 and avg_income < 40:
                icon, label = "💰", "BARGAIN HUNTER"
                rec = "Offer discounts and promotional deals"
            elif avg_spend < 30 and avg_income > 70:
                icon, label = "⚠️", "CAREFUL SPENDER"
                rec = "Build trust, offer value-based marketing"
            else:
                icon, label = "📌", "GENERAL"
                rec = "Standard marketing approach"
            
            with st.expander(f"{icon} Segment {i} - {label} ({len(cluster_data)} customers, {percentage:.1f}%)"):
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Average Age", f"{avg_age:.0f} years")
                col_b.metric("Average Income", f"${avg_income:.0f}K")
                col_c.metric("Spending Score", f"{avg_spend:.0f}")
                st.info(f"💡 **Marketing Recommendation:** {rec}")
                
                fig_mini = px.bar(
                    x=['Income (K$)', 'Spending Score'],
                    y=[avg_income, avg_spend],
                    title=f"Segment {i} Profile",
                    color=['Income', 'Spending'],
                    color_discrete_sequence=['#00e5ff', '#00aaff'],
                    template='plotly_dark'
                )
                fig_mini.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=300
                )
                st.plotly_chart(fig_mini, use_container_width=True)

# ============================================
# DATA PAGE
# ============================================
elif st.session_state.page == 'data':
    st.markdown("## 📋 Customer Data Explorer")
    st.markdown("*View and export customer data with segment labels*")
    st.markdown("---")
    
    search = st.text_input("🔍 Search Customer", placeholder="Enter Customer ID...")
    
    if search:
        filtered_df = df[df['CustomerID'].astype(str).str.contains(search)]
        st.dataframe(filtered_df, use_container_width=True, height=500)
    else:
        st.dataframe(df, use_container_width=True, height=500)
    
    st.markdown("---")
    
    csv = df.to_csv(index=False)
    st.download_button("📥 Download Customer Data (CSV)", csv, "customer_segments.csv", "text/csv")

# Footer
st.markdown("""
<div class="footer">
    🎯 Customer Segmentation Dashboard | Powered by K-Means Clustering | Black x Cyan Theme
</div>
""", unsafe_allow_html=True)