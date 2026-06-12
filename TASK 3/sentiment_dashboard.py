import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
import time
import os

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Page setup
st.set_page_config(page_title="Instagram Sentiment Analysis", layout="wide")

# INSTAGRAM BRAND CSS
st.markdown("""
<style>
    /* Instagram Brand Colors */
    .stApp {
        background: radial-gradient(circle at 20% 30%, #0a0a0a 0%, #1a1a2e 100%);
    }
    
    /* Instagram Gradient Animation */
    @keyframes instaGradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulseRing {
        0% { box-shadow: 0 0 0 0 rgba(255, 69, 0, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(255, 69, 0, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 69, 0, 0); }
    }
    
    @keyframes likeAnimation {
        0% { transform: scale(1); }
        50% { transform: scale(1.3); color: #ed4956; }
        100% { transform: scale(1); }
    }
    
    /* Main Header */
    .insta-header {
        text-align: center;
        padding: 25px 20px;
        animation: slideUp 0.6s ease-out;
    }
    
    .insta-header h1 {
        font-size: 52px;
        font-weight: 800;
        background: linear-gradient(135deg, #feda77, #f58529, #dd2a7b, #8134af, #515bd4);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: instaGradient 4s ease infinite;
        letter-spacing: -1px;
    }
    
    .insta-header p {
        color: #8e8e9e;
        font-size: 14px;
        letter-spacing: 2px;
        margin-top: 10px;
    }
    
    .insta-divider {
        height: 3px;
        background: linear-gradient(90deg, #f58529, #dd2a7b, #8134af, #515bd4);
        width: 40%;
        margin: 20px auto;
        border-radius: 5px;
    }
    
    /* Story Cards */
    .story-container {
        display: flex;
        gap: 15px;
        overflow-x: auto;
        padding: 15px 0;
    }
    
    .story-item {
        text-align: center;
        min-width: 80px;
        cursor: pointer;
        transition: transform 0.3s;
    }
    
    .story-item:hover {
        transform: translateY(-5px);
    }
    
    .story-ring {
        width: 70px;
        height: 70px;
        margin: 0 auto 8px;
        background: linear-gradient(135deg, #f58529, #dd2a7b, #8134af);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        animation: pulseRing 2s infinite;
    }
    
    .story-inner {
        width: 62px;
        height: 62px;
        background: #1a1a2e;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
    }
    
    .story-name {
        font-size: 11px;
        color: #8e8e9e;
        font-weight: 500;
    }
    
    /* Metric Cards */
    .insta-metric {
        background: rgba(26, 26, 46, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
        transition: all 0.3s ease;
    }
    
    .insta-metric:hover {
        transform: translateY(-3px);
        border-color: rgba(245, 133, 41, 0.4);
        background: rgba(26, 26, 46, 0.8);
    }
    
    .insta-metric-value {
        font-size: 38px;
        font-weight: 800;
        background: linear-gradient(135deg, #f58529, #dd2a7b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .insta-metric-label {
        color: #8e8e9e;
        font-size: 12px;
        letter-spacing: 1px;
        margin-top: 5px;
    }
    
    /* Instagram Post Card */
    .insta-post {
        background: #1a1a2e;
        border-radius: 20px;
        margin: 15px 0;
        border: 1px solid rgba(255,255,255,0.08);
        transition: all 0.3s ease;
        overflow: hidden;
    }
    
    .insta-post:hover {
        border-color: rgba(245, 133, 41, 0.3);
        transform: translateY(-2px);
    }
    
    .post-header {
        display: flex;
        align-items: center;
        padding: 15px 20px;
        gap: 12px;
    }
    
    .post-avatar {
        width: 42px;
        height: 42px;
        background: linear-gradient(135deg, #f58529, #dd2a7b);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 16px;
    }
    
    .post-user {
        flex: 1;
    }
    
    .post-username {
        font-weight: 600;
        color: #ffffff;
    }
    
    .post-time {
        font-size: 11px;
        color: #8e8e9e;
    }
    
    .post-badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 500;
    }
    
    .badge-positive {
        background: rgba(0, 229, 255, 0.15);
        color: #00e5ff;
        border: 1px solid rgba(0, 229, 255, 0.3);
    }
    
    .badge-negative {
        background: rgba(255, 75, 75, 0.15);
        color: #ff6b6b;
        border: 1px solid rgba(255, 75, 75, 0.3);
    }
    
    .badge-neutral {
        background: rgba(255, 170, 0, 0.15);
        color: #ffaa00;
        border: 1px solid rgba(255, 170, 0, 0.3);
    }
    
    .post-content {
        padding: 0 20px 15px 20px;
        color: #e0e0e0;
        line-height: 1.5;
    }
    
    .post-stats {
        display: flex;
        gap: 25px;
        padding: 12px 20px;
        border-top: 1px solid rgba(255,255,255,0.05);
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    
    .post-stat {
        display: flex;
        align-items: center;
        gap: 6px;
        color: #8e8e9e;
        font-size: 13px;
    }
    
    .post-stat:hover {
        color: #ed4956;
        cursor: pointer;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a, #0a0a0a);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #f58529, #dd2a7b);
        color: white;
        border: none;
        border-radius: 30px;
        transition: all 0.3s ease;
        width: 100%;
        padding: 10px;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 20px rgba(245, 133, 41, 0.4);
    }
    
    /* Footer */
    .insta-footer {
        text-align: center;
        padding: 25px;
        color: #8e8e9e;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 40px;
        font-size: 11px;
        letter-spacing: 1px;
    }
    
    /* Sentiment Analysis Result */
    .sentiment-result {
        border-radius: 24px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
        animation: slideUp 0.5s ease-out;
    }
    
    /* Glass Card */
    .glass-card {
        background: rgba(26, 26, 46, 0.5);
        backdrop-filter: blur(10px);
        border-radius: 24px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.08);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="insta-header">
    <h1> Instagram Sentiment Analysis</h1>
    <p>ANALYZE POSTS & COMMENTS | TRACK ENGAGEMENT | UNDERSTAND YOUR AUDIENCE</p>
    <div class="insta-divider"></div>
</div>
""", unsafe_allow_html=True)

# Color mapping
SENTIMENT_COLORS = {
    'Positive': '#00e5ff', 
    'Slightly Positive': '#0099ff', 
    'Neutral': '#ffaa00', 
    'Slightly Negative': '#ff8888', 
    'Negative': '#ff6b6b'
}

# Instagram influencers data
def create_instagram_data():
    data = {
        'username': [
            '@style_influencer', '@tech_guru', '@foodie_diaries', '@travel_addict', '@fitness_motivator',
            '@beauty_expert', '@memes_daily', '@fashion_icon', '@photography_lover', '@music_vibes',
            '@lifestyle_blogger', '@art_creator', '@business_tips', '@pet_lover', '@nature_captures'
        ],
        'post_text': [
            'Absolutely obsessed with this new collection! The quality is amazing 🔥🔥 #fashion',
            'This gadget is a game changer! Best purchase of 2024 💯 #tech',
            'The food was just okay. Not worth the hype tbh 🍕 #foodie',
            'Best vacation ever! Paradise found 🌴✨ #travelgram',
            'New workout routine is KILLING me! But results are showing 💪 #fitness',
            'This foundation is terrible. Complete waste of money 💔 #beauty',
            'LOL this is hilarious 😂 #memes #relatable',
            'Obsessed with this outfit! Where has this been all my life? 👗 #fashion',
            'Breathtaking sunset capture 📸 Nature is incredible #photography',
            'This album is pure perfection! Every song is a banger 🎵 #music',
            'Not what I expected. Pretty disappointing overall.',
            'Absolutely stunning artwork! The details are incredible 🎨 #art',
            'Great tips for growing your business! Highly recommend 📈 #entrepreneur',
            'My dog is the cutest! Love this little guy 🐕 #petlover',
            'Nature heals everything. So peaceful and calm 🌿 #nature'
        ],
        'likes': np.random.randint(500, 50000, 15),
        'comments': np.random.randint(10, 2000, 15),
        'shares': np.random.randint(5, 1000, 15),
        'followers': np.random.randint(5000, 500000, 15)
    }
    return pd.DataFrame(data)

@st.cache_data
def load_data():
    file_path = r"C:\senti_analysis\instagram_posts.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = create_instagram_data()
        os.makedirs(r"C:\senti_analysis", exist_ok=True)
        df.to_csv(file_path, index=False)
    return df

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity
    
    if polarity > 0.3:
        sentiment = 'Positive'
        confidence = min(1.0, polarity)
    elif polarity > 0.05:
        sentiment = 'Slightly Positive'
        confidence = polarity * 2
    elif polarity < -0.3:
        sentiment = 'Negative'
        confidence = min(1.0, abs(polarity))
    elif polarity < -0.05:
        sentiment = 'Slightly Negative'
        confidence = abs(polarity) * 2
    else:
        sentiment = 'Neutral'
        confidence = 0.5
    
    return sentiment, polarity, subjectivity, confidence

with st.spinner("Loading Instagram feed..."):
    df = load_data()
    df['cleaned_text'] = df['post_text'].apply(clean_text)
    df[['sentiment', 'polarity', 'subjectivity', 'confidence']] = df['post_text'].apply(
        lambda x: pd.Series(get_sentiment(x))
    )
    time.sleep(0.3)

# Sidebar
with st.sidebar:
    st.markdown("### 🔍 NAVIGATION")
    st.markdown("---")
    
    page = st.radio(
        "",
        ["🏠 INSTAGRAM FEED", "📊 ANALYTICS", "⚡ LIVE ANALYSIS", "📁 POST ARCHIVE"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 🎯 FILTER BY SENTIMENT")
    
    sentiment_filter = st.multiselect(
        "",
        options=['Positive', 'Slightly Positive', 'Neutral', 'Slightly Negative', 'Negative'],
        default=['Positive', 'Slightly Positive', 'Neutral', 'Slightly Negative', 'Negative']
    )
    
    st.markdown("---")
    st.markdown("### 📈 ACCOUNT STATS")
    
    total_engagement = df['likes'].sum() + df['comments'].sum() + df['shares'].sum()
    st.metric("👥 Total Followers", f"{df['followers'].sum():,}")
    st.metric("❤️ Total Engagement", f"{total_engagement:,}")
    st.metric("📸 Total Posts", len(df))
    
    st.markdown("---")
    st.markdown("### 💫 INSTAGRAM")
    st.caption("Analyzing sentiment across your Instagram presence")

filtered_df = df[df['sentiment'].isin(sentiment_filter)]

# ============================================
# INSTAGRAM FEED PAGE
# ============================================
if page == "🏠 INSTAGRAM FEED":
    
    # Stories Row
    st.markdown("### 📸 Stories")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    stories_data = [
        ("🎯", "Trending", f"{len(df[df['sentiment']=='Positive'])}%"),
        ("😊", "Positive", f"{len(df[df['sentiment']=='Positive'])/len(df)*100:.0f}%"),
        ("😐", "Neutral", f"{len(df[df['sentiment']=='Neutral'])/len(df)*100:.0f}%"),
        ("😟", "Negative", f"{len(df[df['sentiment']=='Negative'])/len(df)*100:.0f}%"),
        ("🔥", "Viral", "12%")
    ]
    
    for idx, (emoji, label, value) in enumerate(stories_data):
        cols = [col1, col2, col3, col4, col5]
        with cols[idx]:
            st.markdown(f"""
            <div class="story-item">
                <div class="story-ring">
                    <div class="story-inner">{emoji}</div>
                </div>
                <div class="story-name">{label}</div>
                <div class="story-name" style="font-size: 10px;">{value}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Metrics Row
    st.markdown("### 📊 Overview")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    pos_count = len(df[df['sentiment'] == 'Positive']) + len(df[df['sentiment'] == 'Slightly Positive'])
    neg_count = len(df[df['sentiment'] == 'Negative']) + len(df[df['sentiment'] == 'Slightly Negative'])
    
    with col_m1:
        st.markdown(f"""
        <div class="insta-metric">
            <div style="font-size: 32px;">📸</div>
            <div class="insta-metric-value">{len(df)}</div>
            <div class="insta-metric-label">Total Posts</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m2:
        st.markdown(f"""
        <div class="insta-metric">
            <div style="font-size: 32px;">❤️</div>
            <div class="insta-metric-value">{df['likes'].sum():,}</div>
            <div class="insta-metric-label">Total Likes</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m3:
        st.markdown(f"""
        <div class="insta-metric">
            <div style="font-size: 32px;">💬</div>
            <div class="insta-metric-value">{df['comments'].sum():,}</div>
            <div class="insta-metric-label">Comments</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m4:
        engagement_rate = (df['likes'].sum() + df['comments'].sum()) / df['followers'].sum() * 100
        st.markdown(f"""
        <div class="insta-metric">
            <div style="font-size: 32px;">📈</div>
            <div class="insta-metric-value">{engagement_rate:.1f}%</div>
            <div class="insta-metric-label">Engagement Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Instagram Feed
    st.markdown("### 📱 Feed")
    
    for idx, row in filtered_df.iterrows():
        if row['sentiment'] in ['Positive', 'Slightly Positive']:
            badge_class = "badge-positive"
            badge_icon = "😊"
        elif row['sentiment'] in ['Negative', 'Slightly Negative']:
            badge_class = "badge-negative"
            badge_icon = "😟"
        else:
            badge_class = "badge-neutral"
            badge_icon = "😐"
        
        # Engagement prediction based on sentiment
        sentiment_multiplier = 1.5 if row['sentiment'] == 'Positive' else 0.7 if row['sentiment'] == 'Negative' else 1.0
        predicted_likes = int(row['likes'] * sentiment_multiplier)
        
        st.markdown(f"""
        <div class="insta-post">
            <div class="post-header">
                <div class="post-avatar">{row['username'][1:3].upper()}</div>
                <div class="post-user">
                    <div class="post-username">{row['username']}</div>
                    <div class="post-time">2 hours ago • <span class="post-badge {badge_class}">{badge_icon} {row['sentiment']}</span></div>
                </div>
            </div>
            <div class="post-content">
                "{row['post_text']}"
            </div>
            <div class="post-stats">
                <div class="post-stat">❤️ {row['likes']:,}</div>
                <div class="post-stat">💬 {row['comments']:,}</div>
                <div class="post-stat">📤 {row['shares']:,}</div>
                <div class="post-stat">👥 {row['followers']:,}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# ANALYTICS PAGE
# ============================================
elif page == "📊 ANALYTICS":
    st.markdown("### 📊 Instagram Analytics")
    st.markdown("---")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Sentiment Distribution")
        sentiment_counts = filtered_df['sentiment'].value_counts()
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color=sentiment_counts.index,
            color_discrete_map=SENTIMENT_COLORS,
            hole=0.4,
            template='plotly_dark'
        )
        fig_pie.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            font=dict(color='#8e8e9e')
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_chart2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Engagement by Sentiment")
        engagement_by_sentiment = filtered_df.groupby('sentiment').agg({
            'likes': 'mean',
            'comments': 'mean',
            'shares': 'mean'
        }).reset_index()
        
        fig_bar = px.bar(
            engagement_by_sentiment,
            x='sentiment',
            y='likes',
            title="Average Likes by Sentiment",
            color='sentiment',
            color_discrete_map=SENTIMENT_COLORS,
            template='plotly_dark'
        )
        fig_bar.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            font=dict(color='#8e8e9e')
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col_wc1, col_wc2 = st.columns(2)
    
    with col_wc1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🔥 Trending Positive Words")
        positive_text = " ".join(df[df['sentiment'].isin(['Positive', 'Slightly Positive'])]['cleaned_text'])
        if positive_text:
            wordcloud_pos = WordCloud(width=500, height=350, background_color='black', 
                                      colormap='Oranges', max_words=80).generate(positive_text)
            fig_pos, ax_pos = plt.subplots(figsize=(10, 6))
            ax_pos.imshow(wordcloud_pos, interpolation='bilinear')
            ax_pos.axis('off')
            st.pyplot(fig_pos)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_wc2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📉 Common Negative Words")
        negative_text = " ".join(df[df['sentiment'].isin(['Negative', 'Slightly Negative'])]['cleaned_text'])
        if negative_text:
            wordcloud_neg = WordCloud(width=500, height=350, background_color='black', 
                                      colormap='Reds', max_words=80).generate(negative_text)
            fig_neg, ax_neg = plt.subplots(figsize=(10, 6))
            ax_neg.imshow(wordcloud_neg, interpolation='bilinear')
            ax_neg.axis('off')
            st.pyplot(fig_neg)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Top Performers
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 🏆 Top Performing Posts")
    top_posts = filtered_df.nlargest(5, 'likes')[['username', 'post_text', 'likes', 'sentiment']]
    for idx, row in top_posts.iterrows():
        st.markdown(f"""
        <div style="padding: 12px; margin: 8px 0; background: rgba(255,255,255,0.03); border-radius: 12px;">
            <div style="display: flex; justify-content: space-between;">
                <span style="font-weight: 600;">{row['username']}</span>
                <span style="color: #f58529;">❤️ {row['likes']:,}</span>
            </div>
            <div style="font-size: 13px; color: #8e8e9e;">"{row['post_text'][:100]}..."</div>
            <div style="font-size: 11px; margin-top: 5px;">Sentiment: {row['sentiment']}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# LIVE ANALYSIS PAGE
# ============================================
elif page == "⚡ LIVE ANALYSIS":
    st.markdown("### ⚡ Live Instagram Comment Analysis")
    st.markdown("---")
    
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 💬 Simulate an Instagram Comment")
        user_input = st.text_area(
            "",
            height=150,
            placeholder="Type a comment as if on Instagram...\n\nExample: 'OMG this is amazing! Love it! 🔥'",
            label_visibility="collapsed"
        )
        analyze_btn = st.button("📸 Analyze Comment", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_result:
        if analyze_btn and user_input:
            sentiment, polarity, subjectivity, confidence = get_sentiment(user_input)
            
            if sentiment in ['Positive', 'Slightly Positive']:
                bg_color = "rgba(0, 229, 255, 0.1)"
                border_color = "#00e5ff"
                emoji = "❤️"
                reaction = "Likely to go viral!"
            elif sentiment in ['Negative', 'Slightly Negative']:
                bg_color = "rgba(255, 75, 75, 0.1)"
                border_color = "#ff6b6b"
                emoji = "💔"
                reaction = "May attract negative replies"
            else:
                bg_color = "rgba(255, 170, 0, 0.1)"
                border_color = "#ffaa00"
                emoji = "😐"
                reaction = "Standard engagement"
            
            st.markdown(f"""
            <div class="sentiment-result" style="background: {bg_color}; border: 1px solid {border_color};">
                <div style="font-size: 64px;">{emoji}</div>
                <div style="font-size: 36px; font-weight: bold; color: {border_color};">{sentiment}</div>
                <div style="margin-top: 20px;">
                    <div style="display: flex; justify-content: center; gap: 30px;">
                        <div><span style="color: #8e8e9e;">Score</span><br><b style="font-size: 22px;">{polarity:.2f}</b></div>
                        <div><span style="color: #8e8e9e;">Confidence</span><br><b style="font-size: 22px;">{confidence:.0%}</b></div>
                    </div>
                </div>
                <div style="margin-top: 15px; font-size: 13px; color: {border_color};">🎯 {reaction}</div>
            </div>
            """, unsafe_allow_html=True)
        elif analyze_btn:
            st.warning("Please enter a comment to analyze")

# ============================================
# POST ARCHIVE PAGE
# ============================================
elif page == "📁 POST ARCHIVE":
    st.markdown("### 📁 All Instagram Posts")
    st.markdown("---")
    
    st.dataframe(df[['username', 'post_text', 'sentiment', 'likes', 'comments', 'shares']], 
                 use_container_width=True, height=500)
    
    csv = df.to_csv(index=False)
    st.download_button("📥 Export Data", csv, "instagram_sentiment_data.csv", "text/csv", use_container_width=True)

# Footer
st.markdown("""
<div class="insta-footer">
    📸 INSTAGRAM SENTIMENT ANALYSIS | POWERED BY NLP | REAL-TIME SOCIAL INSIGHTS
</div>
""", unsafe_allow_html=True)