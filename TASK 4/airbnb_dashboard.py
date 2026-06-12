import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time

# Page configuration
st.set_page_config(
    page_title="NYC Airbnb Analytics Dashboard",
    page_icon="🗽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme with animations
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #0a0a0a 100%);
    }
    
    /* Animated card */
    @keyframes slideUp {
        from { transform: translateY(50px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-15px); }
    }
    
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    @keyframes glowPulse {
        0% { box-shadow: 0 0 5px rgba(0,200,255,0.3); }
        50% { box-shadow: 0 0 30px rgba(0,200,255,0.8); }
        100% { box-shadow: 0 0 5px rgba(0,200,255,0.3); }
    }
    
    .animated-card {
        background: linear-gradient(135deg, rgba(0,200,255,0.15), rgba(108,99,255,0.08));
        border-radius: 25px;
        padding: 30px;
        text-align: center;
        border: 2px solid rgba(0,200,255,0.5);
        animation: slideUp 0.6s ease-out, glowPulse 2s ease-in-out infinite;
        margin: 20px 0;
    }
    
    .result-big-number {
        font-size: 72px;
        font-weight: bold;
        background: linear-gradient(135deg, #00c8ff, #6c63ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: bounce 1s ease-in-out;
    }
    
    .title-animation h1 {
        background: linear-gradient(135deg, #00c8ff 0%, #6c63ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        animation: glowPulse 3s ease-in-out infinite;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #1a2332 0%, #0f1419 100%);
        color: #00c8ff;
        border: 1px solid rgba(0, 200, 255, 0.3);
        border-radius: 12px;
        transition: all 0.3s ease;
        width: 100%;
        font-weight: 600;
        padding: 12px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #00c8ff 0%, #6c63ff 100%);
        color: white;
        transform: scale(1.02);
        box-shadow: 0 8px 25px rgba(0, 200, 255, 0.4);
    }
    
    .info-box {
        background: rgba(0, 200, 255, 0.05);
        border-left: 3px solid #00c8ff;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }
    
    .success-box {
        background: rgba(0, 255, 100, 0.08);
        border-left: 3px solid #00ff66;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }
    
    .footer {
        text-align: center;
        padding: 25px;
        color: #445566;
        font-size: 11px;
        border-top: 1px solid rgba(0, 200, 255, 0.1);
        margin-top: 40px;
    }
    
    .comparison-card {
        background: rgba(0, 0, 0, 0.4);
        border-radius: 20px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid rgba(0,200,255,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title-animation"><h1>🗽 NYC Airbnb Data Analytics Dashboard 🗽</h1></div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #8899aa;">INTERACTIVE DATA CLEANING & VISUALIZATION PLATFORM</p>', unsafe_allow_html=True)
st.markdown('<div style="height: 2px; background: linear-gradient(90deg, transparent, #00c8ff, #6c63ff, #00c8ff, transparent); width: 70%; margin: 20px auto;"></div>', unsafe_allow_html=True)

# Load data
@st.cache_data
def load_original_data():
    np.random.seed(42)
    neighborhoods = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
    room_types = ['Entire home/apt', 'Private room', 'Shared room']
    price_options = [80, 100, 120, 150, 180, 200, 250, 300, 400, 500, 800, 1200, 2500]
    
    data = {
        'Listing ID': range(1001, 1301),
        'Host Name': [f'Host_{np.random.randint(1, 60)}' for _ in range(300)],
        'Neighbourhood Group': np.random.choice(neighborhoods, 300, p=[0.35, 0.30, 0.20, 0.10, 0.05]),
        'Price': np.random.choice(price_options, 300),
        'Room Type': np.random.choice(room_types, 300, p=[0.45, 0.45, 0.10]),
        'Number of Reviews': np.random.randint(0, 200, 300),
        'Availability 365': np.random.randint(0, 365, 300),
    }
    df = pd.DataFrame(data)
    missing_indices = np.random.choice(df.index, size=int(len(df)*0.05), replace=False)
    df.loc[missing_indices, 'Price'] = np.nan
    df = pd.concat([df, df.iloc[0:8]], ignore_index=True)
    return df

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = load_original_data()
if 'original_df' not in st.session_state:
    st.session_state.original_df = load_original_data()
if 'cleaning_history' not in st.session_state:
    st.session_state.cleaning_history = []
if 'show_result_page' not in st.session_state:
    st.session_state.show_result_page = False
if 'last_action_result' not in st.session_state:
    st.session_state.last_action_result = None
if 'last_action_name' not in st.session_state:
    st.session_state.last_action_name = None

# Sidebar
with st.sidebar:
    st.markdown("## 🎛️ CONTROL CENTER")
    st.markdown("---")
    
    missing_pct = (st.session_state.df.isnull().sum().sum() / (len(st.session_state.df) * len(st.session_state.df.columns))) * 100
    duplicate_pct = (st.session_state.df.duplicated().sum() / len(st.session_state.df)) * 100 if len(st.session_state.df) > 0 else 0
    outlier_pct = ((st.session_state.df['Price'] > 1000).sum() / len(st.session_state.df)) * 100 if len(st.session_state.df) > 0 else 0
    health_score = max(0, min(100, 100 - (missing_pct + duplicate_pct + outlier_pct)))
    
    st.markdown(f"""
    <div style="text-align: center; margin: 10px 0;">
        <div style="font-size: 55px; font-weight: bold; color: {'#00ff66' if health_score > 80 else '#ffaa00' if health_score > 50 else '#ff4444'}">{health_score:.0f}%</div>
        <div style="color: #8899aa;">DATA QUALITY</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🧹 CLEANING ACTIONS")
    st.markdown("*Click any button to see animated result page*")
    
    # BUTTON 1: Remove Nulls
    if st.button("🗑️ REMOVE NULL VALUES", use_container_width=True):
        before_count = st.session_state.df.isnull().sum().sum()
        before_rows = len(st.session_state.df)
        st.session_state.df = st.session_state.df.dropna()
        after_count = st.session_state.df.isnull().sum().sum()
        after_rows = len(st.session_state.df)
        
        st.session_state.last_action_result = {
            'action': 'remove_nulls',
            'action_name': 'Remove Null Values',
            'before_value': before_count,
            'after_value': after_count,
            'before_rows': before_rows,
            'after_rows': after_rows,
            'description': f'Removed {before_count} missing values from {before_rows - after_rows} rows'
        }
        st.session_state.cleaning_history.append(f"🗑️ Removed {before_count} null values")
        st.session_state.show_result_page = True
        st.rerun()
    
    # BUTTON 2: Remove Duplicates
    if st.button("🔄 REMOVE DUPLICATES", use_container_width=True):
        before_dupes = st.session_state.df.duplicated().sum()
        before_rows = len(st.session_state.df)
        st.session_state.df = st.session_state.df.drop_duplicates()
        after_dupes = st.session_state.df.duplicated().sum()
        after_rows = len(st.session_state.df)
        
        st.session_state.last_action_result = {
            'action': 'remove_dupes',
            'action_name': 'Remove Duplicates',
            'before_value': before_dupes,
            'after_value': after_dupes,
            'before_rows': before_rows,
            'after_rows': after_rows,
            'description': f'Removed {before_dupes - after_dupes} duplicate rows'
        }
        st.session_state.cleaning_history.append(f"🔄 Removed {before_dupes - after_dupes} duplicates")
        st.session_state.show_result_page = True
        st.rerun()
    
    # BUTTON 3: Cap Prices
    if st.button("💰 CAP PRICES (>$1000)", use_container_width=True):
        before_outliers = (st.session_state.df['Price'] > 1000).sum()
        before_rows = len(st.session_state.df)
        st.session_state.df = st.session_state.df[st.session_state.df['Price'] <= 1000]
        after_outliers = (st.session_state.df['Price'] > 1000).sum() if len(st.session_state.df) > 0 else 0
        after_rows = len(st.session_state.df)
        
        st.session_state.last_action_result = {
            'action': 'cap_prices',
            'action_name': 'Cap Prices > $1000',
            'before_value': before_outliers,
            'after_value': after_outliers,
            'before_rows': before_rows,
            'after_rows': after_rows,
            'description': f'Removed {before_outliers - after_outliers} price outliers (>$1000)'
        }
        st.session_state.cleaning_history.append(f"💰 Capped {before_outliers - after_outliers} price outliers")
        st.session_state.show_result_page = True
        st.rerun()
    
    # BUTTON 4: Fill Missing Prices
    if st.button("📝 FILL MISSING PRICES", use_container_width=True):
        before_missing = st.session_state.df['Price'].isnull().sum()
        median_price = st.session_state.df['Price'].median()
        st.session_state.df['Price'] = st.session_state.df['Price'].fillna(median_price)
        after_missing = st.session_state.df['Price'].isnull().sum()
        
        st.session_state.last_action_result = {
            'action': 'fill_prices',
            'action_name': 'Fill Missing Prices',
            'before_value': before_missing,
            'after_value': after_missing,
            'median_price': median_price,
            'description': f'Filled {before_missing} missing prices with median value ${median_price:.0f}'
        }
        st.session_state.cleaning_history.append(f"📝 Filled {before_missing} missing prices with ${median_price:.0f}")
        st.session_state.show_result_page = True
        st.rerun()
    
    # BUTTON 5: Standardize Text
    if st.button("✨ STANDARDIZE TEXT", use_container_width=True):
        before_unique = st.session_state.df['Neighbourhood Group'].nunique()
        st.session_state.df['Neighbourhood Group'] = st.session_state.df['Neighbourhood Group'].str.title()
        st.session_state.df['Room Type'] = st.session_state.df['Room Type'].str.lower()
        after_unique = st.session_state.df['Neighbourhood Group'].nunique()
        
        st.session_state.last_action_result = {
            'action': 'standardize',
            'action_name': 'Standardize Text',
            'description': 'Standardized all neighborhood and room type names to proper case'
        }
        st.session_state.cleaning_history.append("✨ Standardized all text formatting")
        st.session_state.show_result_page = True
        st.rerun()
    
    st.markdown("---")
    
    # Reset Button
    if st.button("🔄 RESET ALL DATA", use_container_width=True, type="primary"):
        st.session_state.df = load_original_data()
        st.session_state.cleaning_history = []
        st.session_state.show_result_page = False
        st.session_state.last_action_result = None
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📋 ACTIVITY LOG")
    if st.session_state.cleaning_history:
        for action in reversed(st.session_state.cleaning_history[-5:]):
            st.markdown(f"<div class='success-box'>{action}</div>", unsafe_allow_html=True)

# ============================================
# RESULT PAGE (Shows when button is clicked)
# ============================================

if st.session_state.show_result_page and st.session_state.last_action_result:
    
    # Animated Result Page Header
    st.markdown("""
    <div style="text-align: center; margin: 20px 0;">
        <div style="font-size: 80px;">🎉</div>
        <h1 style="color: #00c8ff; font-size: 48px;">CLEANING COMPLETE!</h1>
        <div style="height: 3px; background: linear-gradient(90deg, transparent, #00c8ff, #6c63ff, #00c8ff, transparent); width: 50%; margin: 20px auto;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    result = st.session_state.last_action_result
    
    # Show what action was performed
    st.markdown(f"""
    <div class="animated-card">
        <div style="font-size: 24px; color: #00c8ff;">ACTION PERFORMED</div>
        <div style="font-size: 36px; font-weight: bold;">{result['action_name']}</div>
        <div style="color: #8899aa; margin-top: 10px;">{result['description']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create animated comparison charts based on action
    if result['action'] in ['remove_nulls', 'remove_dupes', 'cap_prices']:
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📉 BEFORE")
            fig_before = px.pie(
                values=[result['before_value'], max(1, result['before_rows'] - result['before_value'])],
                names=['Issues Found', 'Clean Data'],
                title=f"<b>Before: {result['before_value']} issues</b>",
                color_discrete_sequence=['#ff4444', '#2a2a2a'],
                hole=0.4
            )
            fig_before.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_font_color='#ff4444',
                font_color='#8899aa',
                height=450,
                annotations=[dict(text=f'{result["before_value"]}', x=0.5, y=0.5, font_size=28, showarrow=False, font_color='#ff4444')]
            )
            st.plotly_chart(fig_before, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 AFTER")
            fig_after = px.pie(
                values=[result['after_value'], max(1, result['after_rows'] - result['after_value'])],
                names=['Issues Remaining', 'Clean Data'],
                title=f"<b>After: {result['after_value']} issues</b>",
                color_discrete_sequence=['#00ff66', '#2a2a2a'],
                hole=0.4
            )
            fig_after.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_font_color='#00ff66',
                font_color='#8899aa',
                height=450,
                annotations=[dict(text=f'{result["after_value"]}', x=0.5, y=0.5, font_size=28, showarrow=False, font_color='#00ff66')]
            )
            st.plotly_chart(fig_after, use_container_width=True)
        
        # Show improvement in BIG NUMBERS
        improvement = result['before_value'] - result['after_value']
        st.markdown(f"""
        <div class="animated-card">
            <div style="font-size: 20px;">📊 IMPROVEMENT</div>
            <div class="result-big-number">+{improvement} issues fixed</div>
            <div style="color: #00ff66; font-size: 18px; margin-top: 10px;">✨ Data quality improved by {((result['before_value'] - result['after_value']) / max(1, result['before_value']) * 100):.0f}% ✨</div>
        </div>
        """, unsafe_allow_html=True)
    
    elif result['action'] == 'fill_prices':
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📉 BEFORE")
            fig_before = px.pie(
                values=[result['before_value'], 100 - result['before_value']],
                names=['Missing Prices', 'Available Prices'],
                title=f"<b>Before: {result['before_value']} missing prices</b>",
                color_discrete_sequence=['#ff4444', '#00c8ff'],
                hole=0.4
            )
            fig_before.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_font_color='#ff4444',
                height=450,
                annotations=[dict(text=f'{result["before_value"]}', x=0.5, y=0.5, font_size=28, showarrow=False, font_color='#ff4444')]
            )
            st.plotly_chart(fig_before, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 AFTER")
            fig_after = px.pie(
                values=[result['after_value'], 100],
                names=['Missing Prices', 'Available Prices'],
                title=f"<b>After: {result['after_value']} missing prices</b>",
                color_discrete_sequence=['#00ff66', '#00c8ff'],
                hole=0.4
            )
            fig_after.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_font_color='#00ff66',
                height=450,
                annotations=[dict(text='0', x=0.5, y=0.5, font_size=28, showarrow=False, font_color='#00ff66')]
            )
            st.plotly_chart(fig_after, use_container_width=True)
        
        st.markdown(f"""
        <div class="animated-card">
            <div style="font-size: 20px;">💰 FILLED WITH MEDIAN VALUE</div>
            <div class="result-big-number">${result['median_price']:.0f}</div>
            <div style="color: #00c8ff; margin-top: 10px;">All missing prices replaced with median value</div>
        </div>
        """, unsafe_allow_html=True)
    
    elif result['action'] == 'standardize':
        st.markdown(f"""
        <div class="animated-card">
            <div style="font-size: 70px;">✨</div>
            <div class="result-big-number">Text Standardized!</div>
            <div style="color: #00c8ff; margin-top: 10px;">All neighborhood names are now properly formatted</div>
            <div style="color: #8899aa; margin-top: 10px;">"brooklyn" → "Brooklyn" | "MANHATTAN" → "Manhattan"</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show example pie chart of cleaned neighborhoods
        neighborhood_counts = st.session_state.df['Neighbourhood Group'].value_counts().reset_index()
        neighborhood_counts.columns = ['Neighborhood', 'Count']
        fig_clean = px.pie(
            neighborhood_counts,
            values='Count',
            names='Neighborhood',
            title="<b>Cleaned Neighborhood Distribution</b>",
            color_discrete_sequence=px.colors.sequential.Magma_r,
            hole=0.3
        )
        fig_clean.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_color='#00c8ff',
            height=450
        )
        st.plotly_chart(fig_clean, use_container_width=True)
    
    # Back to Dashboard Button
    st.markdown("---")
    if st.button("🏠 BACK TO MAIN DASHBOARD", use_container_width=True):
        st.session_state.show_result_page = False
        st.rerun()

# ============================================
# MAIN DASHBOARD (Shows when no result page)
# ============================================

else:
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_price = st.session_state.df['Price'].mean() if len(st.session_state.df) > 0 and st.session_state.df['Price'].notna().any() else 0
        st.markdown(f"""
        <div class="comparison-card" style="text-align: center;">
            <div style="font-size: 36px;">💰</div>
            <div style="font-size: 32px; font-weight: bold; color: #00c8ff;">${avg_price:.0f}</div>
            <div style="color: #8899aa;">AVG PRICE</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="comparison-card" style="text-align: center;">
            <div style="font-size: 36px;">🏠</div>
            <div style="font-size: 32px; font-weight: bold; color: #00c8ff;">{len(st.session_state.df):,}</div>
            <div style="color: #8899aa;">LISTINGS</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        missing = st.session_state.df.isnull().sum().sum()
        st.markdown(f"""
        <div class="comparison-card" style="text-align: center;">
            <div style="font-size: 36px;">❌</div>
            <div style="font-size: 32px; font-weight: bold; color: {'#00ff66' if missing == 0 else '#ffaa00'};">{missing}</div>
            <div style="color: #8899aa;">MISSING VALUES</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        outliers = (st.session_state.df['Price'] > 1000).sum() if len(st.session_state.df) > 0 else 0
        st.markdown(f"""
        <div class="comparison-card" style="text-align: center;">
            <div style="font-size: 36px;">⚠️</div>
            <div style="font-size: 32px; font-weight: bold; color: {'#00ff66' if outliers == 0 else '#ffaa00'};">{outliers}</div>
            <div style="color: #8899aa;">PRICE OUTLIERS</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 LIVE DATA PREVIEW")
    st.dataframe(st.session_state.df.head(50), use_container_width=True, height=350)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <span>🗽 NYC AIRBNB DASHBOARD | CLICK ANY BUTTON IN SIDEBAR TO SEE ANIMATED RESULTS 🗽</span>
    </div>
    """, unsafe_allow_html=True)