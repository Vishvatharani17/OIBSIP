import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Play Store Intelligence", page_icon="📱", layout="wide")

# ============================================
# BLACK BACKGROUND + DARK BLUE + PURPLE GLOW
# ============================================
st.markdown("""
<style>
    .stApp {
        background: #000000;
    }
    
    [data-testid="stSidebar"] {
        background: #0a0a0f;
        border-right: 1px solid #1a1a3a;
    }
    
    [data-testid="stSidebar"] * {
        color: #4a7acc;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Control Panel Category Buttons */
    .category-btn {
        background: #0a0a12;
        padding: 12px 15px;
        border-radius: 10px;
        border: 1px solid #1a1a3a;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s;
        margin-bottom: 8px;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 500;
        color: #4a7acc;
    }
    
    .category-btn:hover {
        border-color: #8a5aff;
        background: #12122a;
        box-shadow: 0 0 10px rgba(138, 90, 255, 0.3);
        transform: translateX(5px);
    }
    
    .category-active {
        border-color: #8a5aff;
        background: #1a1a3a;
        color: #8a5aff;
        box-shadow: 0 0 8px rgba(138, 90, 255, 0.4);
    }
    
    .main-header {
        text-align: center;
        padding: 20px;
        background: #0a0a12;
        border-radius: 16px;
        border: 1px solid #1a1a3a;
        margin-bottom: 25px;
    }
    
    .main-title {
        font-size: 34px;
        font-weight: 800;
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #4a7acc, #8a5aff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .main-subtitle {
        font-size: 11px;
        letter-spacing: 3px;
        color: #6a7a9a;
        margin-top: 5px;
    }
    
    .kpi-card {
        background: #0a0a12;
        border-radius: 14px;
        padding: 15px;
        text-align: center;
        border: 1px solid #1a1a3a;
        transition: all 0.3s;
    }
    
    .kpi-card:hover {
        border-color: #8a5aff;
        box-shadow: 0 0 10px rgba(138, 90, 255, 0.2);
    }
    
    .kpi-value {
        font-size: 28px;
        font-weight: 800;
        background: linear-gradient(135deg, #4a7acc, #8a5aff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .kpi-label {
        font-size: 10px;
        color: #6a7a9a;
        letter-spacing: 1px;
        margin-top: 5px;
    }
    
    .insight-card {
        background: #0a0a12;
        border-radius: 16px;
        padding: 18px;
        border: 1px solid #1a1a3a;
        margin-bottom: 20px;
    }
    
    .section-title {
        font-size: 18px;
        font-weight: 600;
        color: #8a5aff;
        border-left: 3px solid #8a5aff;
        padding-left: 12px;
        margin-bottom: 15px;
    }
    
    .footer {
        text-align: center;
        padding: 20px;
        color: #5a6a8a;
        border-top: 1px solid #1a1a3a;
        margin-top: 40px;
        font-size: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <div class="main-title">📱 PLAY STORE INTELLIGENCE</div>
    <div class="main-subtitle">INTERACTIVE MARKET ANALYTICS | LIVE CATEGORY INSIGHTS</div>
</div>
""", unsafe_allow_html=True)

# ============================================
# CREATE DATA
# ============================================
@st.cache_data
def create_data():
    np.random.seed(42)
    n_apps = 3000
    
    categories = ['GAME', 'FAMILY', 'TOOLS', 'BUSINESS', 'PHOTOGRAPHY', 'SOCIAL', 
                  'EDUCATION', 'ENTERTAINMENT', 'SHOPPING', 'LIFESTYLE', 'SPORTS', 'HEALTH',
                  'NEWS', 'TRAVEL', 'FINANCE', 'MUSIC', 'VIDEO_PLAYERS', 'BOOKS']
    
    apps = pd.DataFrame({
        'App': [f'App_{i}' for i in range(1, n_apps + 1)],
        'Category': np.random.choice(categories, n_apps),
        'Rating': np.round(np.random.uniform(2.5, 5.0, n_apps), 1),
        'Reviews': np.random.randint(10, 800000, n_apps),
        'Installs': np.random.choice(['1,000+', '10,000+', '100,000+', '1,000,000+', '10,000,000+', '50,000,000+', '100,000,000+', '500,000,000+'], n_apps),
        'Type': np.random.choice(['Free', 'Paid'], n_apps, p=[0.88, 0.12]),
        'Price': np.random.choice([0, 0.99, 1.99, 2.99, 4.99, 9.99, 19.99, 49.99, 99.99], n_apps),
        'Content Rating': np.random.choice(['Everyone', 'Teen', 'Mature 17+'], n_apps, p=[0.68, 0.22, 0.10]),
        'Android Ver': np.random.choice(['4.1', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0', '11.0', '12.0', '13.0'], n_apps),
        'Last Updated': pd.date_range('2020-01-01', '2024-12-31', periods=n_apps),
        'Size': np.random.choice(['5M', '10M', '20M', '30M', '50M', '100M', 'Varies'], n_apps),
    })
    
    review_texts = {
        'GAME': ['Awesome game! Addictive gameplay!', 'Best game ever!', 'Too many ads', 'Fun but repetitive'],
        'FAMILY': ['Great for kids!', 'Educational and fun', 'My children love it', 'Safe and engaging'],
        'TOOLS': ['Very useful tool', 'Saves time', 'Works perfectly', 'Essential app'],
        'BUSINESS': ['Productivity booster', 'Great for work', 'User friendly', 'Must have'],
        'SOCIAL': ['Love connecting with friends', 'Great platform', 'Easy to use', 'Addictive'],
        'ENTERTAINMENT': ['Endless content', 'Love streaming here', 'Best entertainment app', 'Great selection'],
        'EDUCATION': ['Learn so much', 'Great courses', 'Very informative', 'Helped me a lot'],
        'SHOPPING': ['Easy checkout', 'Great deals', 'Fast delivery', 'Love shopping here'],
        'LIFESTYLE': ['Life changing!', 'Very helpful tips', 'Love the content', 'Daily use'],
    }
    
    all_reviews = []
    for cat in categories:
        if cat in review_texts:
            texts = review_texts[cat]
        else:
            texts = ['Good app!', 'Nice experience', 'Could be better', 'Average app']
        all_reviews.extend(texts)
    
    reviews = pd.DataFrame({
        'App': np.random.choice([f'App_{i}' for i in range(1, n_apps + 1)], 10000),
        'Translated_Review': np.random.choice(all_reviews, 10000)
    })
    
    def clean_installs(x):
        return float(x.replace('+', '').replace(',', ''))
    
    apps['Installs_clean'] = apps['Installs'].apply(clean_installs)
    
    def get_sentiment(text):
        analysis = TextBlob(str(text))
        if analysis.sentiment.polarity > 0.1:
            return 'Positive'
        elif analysis.sentiment.polarity < -0.1:
            return 'Negative'
        else:
            return 'Neutral'
    
    reviews['Sentiment'] = reviews['Translated_Review'].apply(get_sentiment)
    
    return apps, reviews

apps, reviews = create_data()

# Get unique categories for control panel
all_categories = sorted(apps['Category'].unique())

# Sidebar - Control Panel
with st.sidebar:
    st.markdown("## 🎮 CONTROL PANEL")
    st.markdown("---")
    
    st.markdown("### 📂 SELECT CATEGORY")
    
    # Category selection with buttons
    selected_category = st.selectbox("", all_categories, index=0)
    
    st.markdown("---")
    st.markdown("### ⚙️ FILTERS")
    
    min_rating = st.slider("Min Rating", 1.0, 5.0, 3.0, 0.5)
    app_type = st.selectbox("App Type", ["All", "Free", "Paid"])
    
    st.markdown("---")
    st.markdown("### 📊 LIVE METRICS")
    
    # Show selected category stats
    cat_data = apps[apps['Category'] == selected_category]
    st.info(f"""
    📱 **{selected_category}**  
    ├─ Apps: {len(cat_data):,}  
    ├─ Avg Rating: {cat_data['Rating'].mean():.2f}  
    ├─ Free Apps: {(cat_data['Type'] == 'Free').mean()*100:.0f}%  
    └─ Total Installs: {cat_data['Installs_clean'].sum():,.0f}
    """)

# Filter data based on selection
filtered_apps = apps[apps['Category'] == selected_category]
if min_rating > 0:
    filtered_apps = filtered_apps[filtered_apps['Rating'] >= min_rating]
if app_type != "All":
    filtered_apps = filtered_apps[filtered_apps['Type'] == app_type]

# ============================================
# MAIN DASHBOARD - LIVE ANALYSIS
# ============================================

# KPI Row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{len(filtered_apps):,}</div>
        <div class="kpi-label">TOTAL APPS</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{filtered_apps['Rating'].mean():.2f}</div>
        <div class="kpi-label">AVG RATING</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{(filtered_apps['Type'] == 'Free').mean()*100:.0f}%</div>
        <div class="kpi-label">FREE APPS</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{filtered_apps['Installs_clean'].sum()/1e6:.1f}M</div>
        <div class="kpi-label">TOTAL INSTALLS</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    revenue = (filtered_apps[filtered_apps['Type'] == 'Paid']['Price'] * filtered_apps[filtered_apps['Type'] == 'Paid']['Installs_clean']).sum()
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="font-size: 22px;">${revenue/1e6:.1f}M</div>
        <div class="kpi-label">EST. REVENUE</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Charts Row 1
col_c1, col_c2 = st.columns(2)

with col_c1:
    st.markdown('<div class="insight-card">', unsafe_allow_html=True)
    st.markdown(f"#### 📊 Rating Distribution - {selected_category}")
    
    fig1 = px.histogram(filtered_apps, x='Rating', nbins=20,
                       title=f"Rating Distribution",
                       color_discrete_sequence=['#8a5aff'],
                       template='plotly_dark')
    fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      title_font_color='#8a5aff', font_color='#4a7acc', height=400)
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_c2:
    st.markdown('<div class="insight-card">', unsafe_allow_html=True)
    st.markdown(f"#### 💰 Price vs Rating - {selected_category}")
    
    fig2 = px.scatter(filtered_apps.sample(min(500, len(filtered_apps))), x='Price', y='Rating',
                     title="Price vs Rating Relationship",
                     color='Type', color_discrete_sequence=['#4a7acc', '#8a5aff'],
                     template='plotly_dark')
    fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      title_font_color='#8a5aff', font_color='#4a7acc', height=400)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Charts Row 2
col_c3, col_c4 = st.columns(2)

with col_c3:
    st.markdown('<div class="insight-card">', unsafe_allow_html=True)
    st.markdown(f"#### 📈 Installs by Rating - {selected_category}")
    
    fig3 = px.box(filtered_apps, x='Rating', y='Installs_clean', log_y=True,
                 title="Installs Distribution by Rating",
                 color_discrete_sequence=['#8a5aff'],
                 template='plotly_dark')
    fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      title_font_color='#8a5aff', font_color='#4a7acc', height=400)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_c4:
    st.markdown('<div class="insight-card">', unsafe_allow_html=True)
    st.markdown(f"#### 🥧 Content Rating - {selected_category}")
    
    content_counts = filtered_apps['Content Rating'].value_counts()
    fig4 = px.pie(values=content_counts.values, names=content_counts.index,
                 title="Content Rating Distribution",
                 color_discrete_sequence=px.colors.sequential.Blues_r,
                 template='plotly_dark')
    fig4.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      title_font_color='#8a5aff', font_color='#4a7acc', height=400)
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Charts Row 3 - Treemap & Android Version
col_c5, col_c6 = st.columns(2)

with col_c5:
    st.markdown('<div class="insight-card">', unsafe_allow_html=True)
    st.markdown(f"#### 🗺️ Top Apps in {selected_category}")
    
    top_apps = filtered_apps.nlargest(10, 'Installs_clean')[['App', 'Rating', 'Installs', 'Reviews']]
    st.dataframe(top_apps, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_c6:
    st.markdown('<div class="insight-card">', unsafe_allow_html=True)
    st.markdown(f"#### 📱 Android Version - {selected_category}")
    
    android_counts = filtered_apps['Android Ver'].value_counts().head(8)
    fig6 = px.bar(x=android_counts.index, y=android_counts.values,
                 title="Android Version Distribution",
                 color=android_counts.values, color_continuous_scale='Blues',
                 template='plotly_dark')
    fig6.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      title_font_color='#8a5aff', font_color='#4a7acc', height=400)
    st.plotly_chart(fig6, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Sentiment Analysis Section
st.markdown('<div class="insight-card">', unsafe_allow_html=True)
st.markdown(f"#### 💬 User Sentiment for {selected_category}")

# Get reviews for selected category apps
category_apps = filtered_apps['App'].tolist()
category_reviews = reviews[reviews['App'].isin(category_apps)]

if len(category_reviews) > 0:
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        sentiment_counts = category_reviews['Sentiment'].value_counts()
        fig_s1 = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                       title="Sentiment Distribution",
                       color_discrete_sequence=['#4a7acc', '#8a5aff', '#ffaa44'],
                       template='plotly_dark')
        fig_s1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                            title_font_color='#8a5aff', height=400)
        st.plotly_chart(fig_s1, use_container_width=True)
    
    with col_s2:
        st.markdown("#### Positive Reviews Word Cloud")
        positive_reviews = category_reviews[category_reviews['Sentiment'] == 'Positive']['Translated_Review'].dropna()
        if len(positive_reviews) > 0:
            positive_text = ' '.join(positive_reviews)
            wordcloud = WordCloud(width=500, height=350, background_color='#0a0a12', 
                                  colormap='Blues', max_words=80).generate(positive_text)
            fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis('off')
            st.pyplot(fig_wc)
        else:
            st.info("No positive reviews for this category")
else:
    st.info("No review data available for this category")

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    📱 PLAY STORE INTELLIGENCE | LIVE CATEGORY ANALYSIS | REAL-TIME MARKET INSIGHTS
</div>
""", unsafe_allow_html=True)