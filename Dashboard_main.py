# social_media_dashboard.py
import re
import toml
import time
import folium
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd
import plotly.express as px
import matplotlib.pyplot as plt
import country_converter as coco
from textblob import TextBlob
from wordcloud import WordCloud
from pathlib import Path
from shapely.geometry import Point
from pyspark.sql import SparkSession
from datetime import datetime, timedelta
from streamlit_folium import folium_static
from pyspark.sql.functions import udf, col
from streamlit_autorefresh import st_autorefresh
from pyspark.sql.types import StringType, FloatType
from folium.plugins import HeatMap
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from gensim import corpora, models
from gensim.utils import simple_preprocess
import spacy
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# Load spaCy model for NLP
try:
    nlp = spacy.load("en_core_web_sm")
except:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Configuration
SQLITE_DB = "social_media_analysis.db"
DEFAULT_REFRESH = 80  # minutes
MAX_TWEETS = 30000
COLOR_PALETTE = px.colors.qualitative.Vivid

# Initialize Spark
spark = SparkSession.builder \
    .appName("SocialMediaAnalysis") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()

# Database Setup with enhanced schema
def init_db():
    conn = sqlite3.connect(SQLITE_DB)
    c = conn.cursor()
    
    # Create tables with enhanced schema for multiple platforms
    c.execute('''CREATE TABLE IF NOT EXISTS searches
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  topic TEXT NOT NULL,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  source TEXT DEFAULT 'csv',
                  platform TEXT DEFAULT 'multiple')''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS posts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  search_id INTEGER,
                  post_id TEXT,
                  text TEXT,
                  cleaned_text TEXT,
                  created_at DATETIME,
                  sentiment TEXT,
                  retweet_count INTEGER,
                  like_count INTEGER,
                  username TEXT,
                  user_followers INTEGER,
                  location TEXT DEFAULT NULL,
                  city TEXT DEFAULT NULL,
                  state TEXT DEFAULT NULL,
                  country TEXT DEFAULT NULL,
                  platform TEXT,
                  year INTEGER,
                  month INTEGER,
                  day INTEGER,
                  hour INTEGER,
                  FOREIGN KEY(search_id) REFERENCES searches(id))''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS hashtags
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  search_id INTEGER,
                  hashtag TEXT,
                  count INTEGER,
                  FOREIGN KEY(search_id) REFERENCES searches(id))''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS analytics
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  search_id INTEGER,
                  metric TEXT,
                  value TEXT,
                  FOREIGN KEY(search_id) REFERENCES searches(id))''')
    
    # Add indexes for performance
    c.execute("CREATE INDEX IF NOT EXISTS idx_searches_topic ON searches(topic)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_posts_search_id ON posts(search_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_hashtags_hashtag ON hashtags(hashtag)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_posts_platform ON posts(platform)")
    
    conn.commit()
    conn.close()

init_db()

## Enhanced Sentiment Analysis with broader spectrum
def get_enhanced_sentiment(text):
    if not text or not isinstance(text, str):
        return "Neutral"
    
    text_lower = text.lower()
    
    # Disaster detection (highest priority)
    disaster_keywords = [
        'emergency', 'disaster', 'help', 'urgent', 'ambulance', 'police',
        'flood', 'earthquake', 'fire', 'attack', 'danger', 'alert', 'alart', 'tsunami',
        'bomb', 'blast', 'explosion', 'crash', 'accident', 'casualty', 'victim',
        'hospital', 'rescue', 'evacuate', 'dangerous', 'crisis', 'emergency'
    ]
    
    if any(keyword in text_lower for keyword in disaster_keywords):
        return "Disaster"
    
    # Happy detection
    happy_keywords = [
        'happy', 'joy', 'excited', 'celebrate', 'celebration', 'congratulations',
        'win', 'won', 'victory', 'success', 'awesome', 'amazing', 'great', 'good',
        'wonderful', 'fantastic', 'excellent', 'love', 'loved', 'beautiful',
        'perfect', 'brilliant', 'outstanding', 'smile', 'laugh', 'fun'
    ]
    
    # Sad detection
    sad_keywords = [
        'sad', 'unhappy', 'depressed', 'cry', 'crying', 'tears', 'mourn',
        'grief', 'loss', 'miss', 'regret', 'sorry', 'apologize', 'tragic',
        'heartbroken', 'devastated', 'miserable', 'upset', 'disappointed'
    ]
    
    # TextBlob analysis for general sentiment
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity
    
    # Enhanced sentiment classification
    if any(keyword in text_lower for keyword in happy_keywords):
        return "Happy"
    elif any(keyword in text_lower for keyword in sad_keywords):
        return "Sad"
    elif polarity > 0.3:
        return "Positive"
    elif polarity < -0.3:
        return "Negative"
    elif -0.1 <= polarity <= 0.1 and subjectivity < 0.3:
        return "Neutral"
    elif polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

# Text Processing Functions
def clean_text(text):
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r'https?:\/\/\S+', '', text)
    return text.strip()

def extract_entities(text):
    if not text or not isinstance(text, str):
        return []
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def topic_modeling(texts):
    if not texts:
        return []
    texts = [simple_preprocess(str(text)) for text in texts if text]
    if not texts:
        return []
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=3, random_state=42)
    return lda_model.print_topics()

# Data Processing Functions
@st.cache_data(ttl=3600, show_spinner=False)
def extract_hashtags(text):
    return re.findall(r'#(\w+)', str(text))

def parse_location(location_str):
    """Parse location string into city, state, country components"""
    if not location_str or not isinstance(location_str, str):
        return None, None, None
    
    location_str = location_str.strip()
    if ',' in location_str:
        parts = [part.strip() for part in location_str.split(',')]
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]  # city, state, country
        elif len(parts) == 2:
            return parts[0], None, parts[1]  # city, country or state, country
        else:
            return None, None, location_str  # fallback
    else:
        return None, None, location_str  # single value, treat as country

@st.cache_data(ttl=3600, show_spinner=False)
def process_posts(posts_data, topic=None):
    if not posts_data:
        return [], {}
    
    processed_posts = []
    hashtag_counts = {}
    
    for post in posts_data:
        # Clean and analyze text
        cleaned_text = clean_text(str(post['text']))
        sentiment = get_enhanced_sentiment(cleaned_text)  # Use enhanced sentiment analysis
        
        # Parse location
        city, state, country = parse_location(post.get('location', ''))
        
        # Enhanced platform cleaning
        platform = post.get('platform', 'unknown')
        if platform and isinstance(platform, str):
            platform = platform.strip()
            # Standardize platform names
            platform_mapping = {
                'twitter': 'Twitter',
                'bluesky': 'Bluesky',
                'mastodon': 'Mastodon', 
                'reddit': 'Reddit',
                'instagram': 'Instagram',
                'facebook': 'Facebook',
                'unknown': 'Unknown'
            }
            platform = platform_mapping.get(platform.lower(), platform)
        
        # Extract hashtags
        hashtags = []
        if 'hashtags' in post and post['hashtags']:
            if isinstance(post['hashtags'], str):
                if post['hashtags'].startswith('['):
                    try:
                        hashtags = eval(post['hashtags'])
                    except:
                        hashtags = extract_hashtags(post['hashtags'])
                else:
                    hashtags = extract_hashtags(post['hashtags'])
            elif isinstance(post['hashtags'], list):
                hashtags = post['hashtags']
        
        # Also extract from text
        hashtags.extend(extract_hashtags(post['text']))
        
        # Count all hashtags (not filtered by topic)
        for tag in hashtags:
            hashtag_counts[tag] = hashtag_counts.get(tag, 0) + 1
        
        # Create post data structure
        post_data = {
            'post_id': str(post.get('post_id', hash(str(post['text'])))),
            'text': str(post['text']),
            'cleaned_text': cleaned_text,
            'created_at': post['created_at'],
            'sentiment': sentiment,  # Our analyzed sentiment, not from CSV
            'retweet_count': int(post.get('retweet_count', 0)),
            'like_count': int(post.get('like_count', 0)),
            'username': str(post.get('username', 'unknown')),
            'user_followers': int(post.get('user_followers', 0)),
            'location': str(post.get('location')) if post.get('location') else None,
            'city': city,
            'state': state,
            'country': country,
            'platform': platform,  # Use cleaned platform
            'year': post['created_at'].year,
            'month': post['created_at'].month,
            'day': post['created_at'].day,
            'hour': post['created_at'].hour
        }
        
        processed_posts.append(post_data)
    
    return processed_posts, hashtag_counts

# Database Functions
def save_to_db(topic, posts_data, hashtag_counts, source='csv', platform='multiple'):
    conn = sqlite3.connect(SQLITE_DB)
    c = conn.cursor()
    
    # Check if this topic exists in the last hour
    c.execute('''SELECT id FROM searches 
                 WHERE topic = ? AND timestamp > datetime('now', '-1 hour')
                 ORDER BY timestamp DESC LIMIT 1''', (topic,))
    result = c.fetchone()
    
    if result:
        search_id = result[0]  # Use existing search_id
    else:
        # Create new search entry
        c.execute("INSERT INTO searches (topic, source, platform) VALUES (?, ?, ?)", 
                 (topic, source, platform))
        search_id = c.lastrowid
    
    for post in posts_data:
        dt = pd.to_datetime(post['created_at'])
        c.execute('''INSERT INTO posts 
                     (search_id, post_id, text, cleaned_text, created_at, 
                      sentiment, retweet_count, like_count, username, 
                      user_followers, location, city, state, country, platform,
                      year, month, day, hour)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (search_id, str(post['post_id']), str(post['text']), str(post['cleaned_text']),
                   str(post['created_at']), str(post['sentiment']), int(post['retweet_count']),
                   int(post['like_count']), str(post['username']), int(post['user_followers']),
                   str(post.get('location')) if post.get('location') else None,
                   str(post.get('city')) if post.get('city') else None,
                   str(post.get('state')) if post.get('state') else None,
                   str(post.get('country')) if post.get('country') else None,
                   str(post.get('platform', 'unknown')),
                   dt.year, dt.month, dt.day, dt.hour))
    
    for tag, count in hashtag_counts.items():
        c.execute("INSERT INTO hashtags (search_id, hashtag, count) VALUES (?, ?, ?)",
                  (search_id, str(tag), int(count)))
    
    sentiment_counts = pd.Series([p['sentiment'] for p in posts_data]).value_counts().to_dict()
    for sentiment, count in sentiment_counts.items():
        c.execute("INSERT INTO analytics (search_id, metric, value) VALUES (?, ?, ?)",
                  (search_id, f"sentiment_{sentiment}", str(count)))
    
    conn.commit()
    conn.close()
    return search_id

@st.cache_data(ttl=600)
def get_search_history():
    conn = sqlite3.connect(SQLITE_DB)
    c = conn.cursor()
    c.execute("SELECT DISTINCT topic FROM searches ORDER BY timestamp DESC LIMIT 10")
    history = [row[0] for row in c.fetchall()]
    conn.close()
    return history

@st.cache_data(ttl=600, show_spinner=False)
def get_cached_data(topic):
    conn = sqlite3.connect(SQLITE_DB)
    
    # Get the most recent search_id for this topic
    search_id = pd.read_sql(
        f"SELECT id FROM searches WHERE topic = '{topic}' ORDER BY timestamp DESC LIMIT 1", 
        conn
    )['id'].iloc[0] if pd.read_sql(
        f"SELECT 1 FROM searches WHERE topic = '{topic}' LIMIT 1", 
        conn
    ).shape[0] > 0 else None
    
    if not search_id:
        conn.close()
        return None
    
    # Get only data for this specific search_id (topic)
    posts = pd.read_sql(f"SELECT * FROM posts WHERE search_id = {search_id}", conn)
    hashtags = pd.read_sql(f"SELECT hashtag, count FROM hashtags WHERE search_id = {search_id} ORDER BY count DESC LIMIT 10", conn)
    analytics = pd.read_sql(f"SELECT metric, value FROM analytics WHERE search_id = {search_id}", conn)
    
    conn.close()
    
    return {
        "posts": posts,
        "hashtags": hashtags,
        "analytics": analytics
    }

## Enhanced CSV Import Functionality with robust platform handling
def import_from_csv(uploaded_file, topic_name):
    try:
        # Read CSV with error handling
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read CSV file: {str(e)}")
            return False, None

        # Filter rows that contain the topic in Text column (case insensitive)
        topic_lower = topic_name.lower()
        df_filtered = df[df['Text'].str.lower().str.contains(topic_lower, na=False)]  # Search in Text column
        
        if len(df_filtered) == 0:
            st.error(f"No posts found containing the topic '{topic_name}' in text content")
            return False, None

        # Standardize column names with flexible mapping
        column_mapping = {
            'Text': 'text',
            'text': 'text',
            'Tweet': 'text',
            'Tweet-Text': 'text',
            'Date': 'created_at',
            'DateTime': 'created_at',
            'Time': 'created_at',
            'Timestamp': 'created_at',
            'User': 'username',
            'Username': 'username',
            'ScreenName': 'username',
            'Author': 'username',
            'Retweets': 'retweet_count',
            'Likes': 'like_count',
            'Favorites': 'like_count',
            'Country': 'location',
            'Location': 'location',
            'Place': 'location',
            'Source': 'platform',
            'Device': 'platform',
            'Platform': 'platform',  # Explicitly map 'Platform' column
            'Hashtags': 'hashtags',
            'hashtags': 'hashtags'
        }
        
        # Apply column name standardization
        df_filtered = df_filtered.rename(columns={k: v for k, v in column_mapping.items() if k in df_filtered.columns})
        
        # Ensure required columns exist
        if 'text' not in df_filtered.columns:
            st.error("CSV must contain a text column with post content")
            return False, None

        # Set defaults for missing columns
        defaults = {
            'created_at': datetime.now(),
            'retweet_count': 0,
            'like_count': 0,
            'username': 'unknown',
            'user_followers': 0,
            'location': None,
            'platform': 'unknown',
            'hashtags': ''
        }
        
        for col, default in defaults.items():
            if col not in df_filtered.columns:
                df_filtered[col] = default

        # Convert data types
        df_filtered['retweet_count'] = pd.to_numeric(df_filtered['retweet_count'], errors='coerce').fillna(0).astype(int)
        df_filtered['like_count'] = pd.to_numeric(df_filtered['like_count'], errors='coerce').fillna(0).astype(int)
        df_filtered['user_followers'] = pd.to_numeric(df_filtered['user_followers'], errors='coerce').fillna(0).astype(int)
        
        # Enhanced platform column cleaning and validation
        if 'platform' in df_filtered.columns:
            # Clean platform names: remove whitespace, handle NaN, standardize values
            df_filtered['platform'] = df_filtered['platform'].astype(str).str.strip()
            
            # Replace common variations with standardized names
            platform_mapping = {
                'twitter': 'Twitter',
                'Twitter': 'Twitter',
                'bluesky': 'Bluesky', 
                'Bluesky': 'Bluesky',
                'mastodon': 'Mastodon',
                'Mastodon': 'Mastodon',
                'reddit': 'Reddit',
                'Reddit': 'Reddit',
                'instagram': 'Instagram',
                'Instagram': 'Instagram',
                'facebook': 'Facebook',
                'Facebook': 'Facebook',
                'unknown': 'Unknown',
                'nan': 'Unknown',
                '': 'Unknown'
            }
            
            df_filtered['platform'] = df_filtered['platform'].map(platform_mapping).fillna(df_filtered['platform'])
        
        # Process each post with proper type conversion
        processed_posts = []
        
        for idx, row in df_filtered.iterrows():
            # Handle datetime conversion
            try:
                created_at = pd.to_datetime(row['created_at'])
            except:
                created_at = datetime.now()
            
            # Enhanced platform extraction
            platform_value = 'Unknown'
            if 'platform' in row and pd.notna(row['platform']):
                platform_value = str(row['platform']).strip()
                if platform_value in ['', 'nan', 'None']:
                    platform_value = 'Unknown'
            
            # Create post data structure
            post_data = {
                'post_id': f"csv_{idx}_{hash(str(row['text']))}",
                'text': str(row['text']),
                'created_at': created_at,
                'retweet_count': int(row['retweet_count']),
                'like_count': int(row['like_count']),
                'username': str(row['username']),
                'user_followers': int(row['user_followers']),
                'location': str(row['location']) if pd.notna(row['location']) else None,
                'platform': platform_value,  # Use cleaned platform value
                'hashtags': str(row['hashtags']) if pd.notna(row['hashtags']) else ''
            }
            
            processed_posts.append(post_data)
        
        # Process posts to get cleaned data and hashtags
        processed_data, hashtag_counts = process_posts(processed_posts, topic_name)
        
        # Save to database with transaction
        conn = sqlite3.connect(SQLITE_DB)
        try:
            c = conn.cursor()
            
            # First check if this topic exists in recent searches (last 24 hours)
            c.execute('''SELECT id FROM searches 
                        WHERE LOWER(topic) = LOWER(?)                         
                        ORDER BY timestamp DESC LIMIT 1''', (topic_name,))
            result = c.fetchone()

            if result:
                # Use existing search_id
                search_id = result[0]
                st.info(f"Appending to existing topic analysis for '{topic_name}'")
            else:
                # Create new search entry
                c.execute("INSERT INTO searches (topic, source, platform) VALUES (?, ?, ?)", 
                        (topic_name, 'csv', 'multiple'))
                search_id = c.lastrowid
                st.info(f"Created new topic analysis for '{topic_name}'")
            
            # Insert posts with explicit type conversion
            for post in processed_data:
                c.execute('''INSERT INTO posts 
                         (search_id, post_id, text, cleaned_text, created_at, 
                          sentiment, retweet_count, like_count, username, 
                          user_followers, location, city, state, country, platform,
                          year, month, day, hour)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                      (
                          int(search_id),
                          str(post['post_id']),
                          str(post['text']),
                          str(post['cleaned_text']),
                          str(post['created_at']),
                          str(post['sentiment']),  # Our analyzed sentiment
                          int(post['retweet_count']),
                          int(post['like_count']),
                          str(post['username']),
                          int(post['user_followers']),
                          str(post['location']) if post['location'] else None,
                          str(post['city']) if post['city'] else None,
                          str(post['state']) if post['state'] else None,
                          str(post['country']) if post['country'] else None,
                          str(post['platform']),  # Actual platform from CSV
                          int(post['year']),
                          int(post['month']),
                          int(post['day']),
                          int(post['hour'])
                      ))
            
            # Insert hashtags
            for tag, count in hashtag_counts.items():
                c.execute("INSERT INTO hashtags (search_id, hashtag, count) VALUES (?, ?, ?)",
                         (int(search_id), str(tag), int(count)))
            
            # Insert analytics
            sentiment_counts = pd.Series([p['sentiment'] for p in processed_data]).value_counts().to_dict()
            for sentiment, count in sentiment_counts.items():
                c.execute("INSERT INTO analytics (search_id, metric, value) VALUES (?, ?, ?)",
                         (int(search_id), f"sentiment_{sentiment}", str(count)))
            
            conn.commit()
            return True, df_filtered
            
        except sqlite3.Error as e:
            conn.rollback()
            st.error(f"Database error during import: {str(e)}")
            return False, None
        finally:
            conn.close()
            
    except Exception as e:
        st.error(f"CSV import failed: {str(e)}")
        return False, None
    
# Enhanced Visualization Functions
def plot_sentiment(data, topic):
    sentiment_counts = data['analytics'][data['analytics']['metric'].str.startswith('sentiment_')]
    sentiment_counts['metric'] = sentiment_counts['metric'].str.replace('sentiment_', '')
    sentiment_counts['value'] = pd.to_numeric(sentiment_counts['value'])
    
    # Define colors for enhanced sentiment spectrum
    sentiment_colors = {
        'Happy': '#00FF00',      # Green
        'Positive': '#90EE90',   # Light Green
        'Neutral': '#FFFF00',    # Yellow
        'Negative': '#FFA500',   # Orange
        'Sad': '#FF6347',       # Tomato Red
        'Disaster': '#FF0000'    # Red
    }
    
    fig = px.pie(sentiment_counts, names='metric', values='value', 
                 title=f'📊 Enhanced Sentiment Analysis for "{topic}"',
                 color='metric',
                 color_discrete_map=sentiment_colors,
                 hole=0.3)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

def plot_post_volume(data, topic):
    posts = data['posts']
    posts['hour'] = pd.to_datetime(posts['created_at']).dt.floor('H')
    volume = posts.groupby('hour').size().reset_index(name='count')
    
    fig = px.area(volume, x='hour', y='count',
                 title=f'📈 Post Volume Over Time for "{topic}"',
                 color_discrete_sequence=['#1f77b4'])
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

def plot_top_hashtags(data, topic):
    fig = px.bar(data['hashtags'], x='hashtag', y='count', 
                 title=f'🏷️ Top Hashtags for "{topic}"',
                 color='hashtag',
                 color_discrete_sequence=COLOR_PALETTE)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def plot_wordcloud(data, topic):
    text = " ".join(post for post in data['posts']['cleaned_text'] if isinstance(post, str))
    if not text:
        st.warning("No text available for word cloud")
        return
    
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='black',
                         colormap='viridis').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f'Word Cloud for "{topic}"', color='white', size=16)
    st.pyplot(plt)

def plot_platform_distribution(data, topic):
    platform_counts = data['posts']['platform'].value_counts().reset_index()
    platform_counts.columns = ['platform', 'count']
    
    fig = px.pie(platform_counts, names='platform', values='count',
                 title=f'📱 Platform Distribution for "{topic}"',
                 color_discrete_sequence=COLOR_PALETTE)
    st.plotly_chart(fig, use_container_width=True)

def plot_top_users(data, topic):
    col1, col2 = st.columns(2)
    
    with col1:
        # Top users by post count
        top_users_by_posts = data['posts'].groupby('username').agg({
            'post_id': 'count',
            'retweet_count': 'sum'
        }).sort_values('post_id', ascending=False).head(10)
        
        fig1 = px.bar(top_users_by_posts, 
                     x=top_users_by_posts.index, 
                     y='post_id',
                     title=f'👥 Top Users by Post Volume',
                     color=top_users_by_posts.index,
                     color_discrete_sequence=COLOR_PALETTE,
                     labels={'username': 'User', 'post_id': 'Post Count'})
        fig1.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Top users by follower count
        top_users_by_followers = data['posts'].sort_values('user_followers', ascending=False) \
            .drop_duplicates('username') \
            .head(10)[['username', 'user_followers']]
        
        fig2 = px.bar(top_users_by_followers,
                     x='username',
                     y='user_followers',
                     title=f'🌟 Top Influencers by Followers',
                     color='username',
                     color_discrete_sequence=COLOR_PALETTE,
                     labels={'username': 'User', 'user_followers': 'Followers Count'})
        fig2.update_layout(showlegend=False, xaxis_tickangle=-45)
        fig2.update_yaxes(type="log")
        st.plotly_chart(fig2, use_container_width=True)

def plot_engagement(data, topic):
    df = data['posts']
    fig = px.scatter(
        df, 
        x='retweet_count', 
        y='like_count', 
        color='sentiment',
        color_discrete_map={
            'Happy': '#00FF00',
            'Positive': '#90EE90',
            'Neutral': '#FFFF00',
            'Negative': '#FFA500',
            'Sad': '#FF6347',
            'Disaster': '#FF0000'
        },
        hover_data=['text'],
        title=f'💬 Engagement Analysis for "{topic}"',
        log_x=True,
        log_y=True
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_entities(data, topic):
    all_entities = []
    for text in data['posts']['cleaned_text']:
        if isinstance(text, str):
            all_entities.extend(extract_entities(text))
    
    if not all_entities:
        st.warning("No named entities found in posts")
        return
    
    entity_df = pd.DataFrame(all_entities, columns=['Entity', 'Type'])
    entity_counts = entity_df.groupby(['Entity', 'Type']).size().reset_index(name='Count')
    
    fig = px.bar(entity_counts.nlargest(20, 'Count'), 
                 x='Entity', y='Count', color='Type',
                 title=f'🔍 Named Entities in "{topic}" Discussions',
                 color_discrete_sequence=COLOR_PALETTE)
    st.plotly_chart(fig, use_container_width=True)

def plot_geographical_heatmap(data, topic):
    try:
        posts_df = data['posts']
        
        # Create location hierarchy: city -> state -> country
        location_data = []
        
        for _, post in posts_df.iterrows():
            if pd.notna(post['city']) and post['city'] not in ['', 'None', 'nan']:
                location_data.append({
                    'type': 'city',
                    'name': post['city'],
                    'state': post['state'],
                    'country': post['country'],
                    'lat': None,
                    'lon': None,
                    'count': 1
                })
            elif pd.notna(post['state']) and post['state'] not in ['', 'None', 'nan']:
                location_data.append({
                    'type': 'state', 
                    'name': post['state'],
                    'state': post['state'],
                    'country': post['country'],
                    'lat': None,
                    'lon': None,
                    'count': 1
                })
            elif pd.notna(post['country']) and post['country'] not in ['', 'None', 'nan']:
                location_data.append({
                    'type': 'country',
                    'name': post['country'],
                    'state': None,
                    'country': post['country'],
                    'lat': None,
                    'lon': None,
                    'count': 1
                })
        
        if not location_data:
            st.warning("No geographical data available for heatmap")
            return
        
        # Convert to DataFrame and aggregate counts
        location_df = pd.DataFrame(location_data)
        location_agg = location_df.groupby(['type', 'name', 'state', 'country']).agg({'count': 'sum'}).reset_index()
        
        geolocator = Nominatim(user_agent="social_media_dashboard", timeout=10)
        cc = coco.CountryConverter()
        
        def get_coordinates(location_name, location_type):
            try:
                if location_type == 'country':
                    # For countries, use country converter for standardization
                    country_standard = cc.convert(location_name, to='name_short')
                    if country_standard != 'not found':
                        location_name = country_standard
                
                location_obj = geolocator.geocode(location_name, timeout=10)
                if location_obj:
                    return (location_obj.latitude, location_obj.longitude)
                return None
            except (GeocoderTimedOut, GeocoderUnavailable, AttributeError):
                return None
        
        # Get coordinates for each location
        location_coords = {}
        for _, row in location_agg.iterrows():
            coords = get_coordinates(row['name'], row['type'])
            if coords:
                location_coords[row['name']] = coords
        
        # Prepare heatmap data
        heat_data = []
        marker_data = []
        
        for _, row in location_agg.iterrows():
            location_name = row['name']
            if location_name in location_coords:
                lat, lon = location_coords[location_name]
                count = row['count']
                
                # For heatmap
                heat_data.append([lat, lon, count])
                
                # For markers with detailed info
                if row['type'] == 'city':
                    popup_text = f"🏙️ {location_name}"
                    if pd.notna(row['state']):
                        popup_text += f", {row['state']}"
                    if pd.notna(row['country']):
                        popup_text += f", {row['country']}"
                elif row['type'] == 'state':
                    popup_text = f"🏛️ {location_name}"
                    if pd.notna(row['country']):
                        popup_text += f", {row['country']}"
                else:
                    popup_text = f"🌍 {location_name}"
                
                popup_text += f"<br>📊 {count} posts"
                
                marker_data.append({
                    'lat': lat,
                    'lon': lon,
                    'popup': popup_text,
                    'count': count,
                    'type': row['type']
                })
        
        if not heat_data:
            st.warning("Could not geocode locations for heatmap")
            return
        
        # Create the map with multiple layers
        lats = [point[0] for point in heat_data]
        lons = [point[1] for point in heat_data]
        center_lat = sum(lats) / len(lats) if lats else 20
        center_lon = sum(lons) / len(lons) if lons else 0
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=2)
        
        # Add heatmap layer
        HeatMap(heat_data, radius=20, blur=15, max_zoom=10, 
               gradient={0.4: 'blue', 0.6: 'lime', 0.8: 'orange', 1.0: 'red'}).add_to(m)
        
        # Add markers with different colors based on location type
        for marker in marker_data:
            if marker['type'] == 'city':
                icon_color = 'green'
                icon_type = 'info-sign'
            elif marker['type'] == 'state':
                icon_color = 'blue'
                icon_type = 'flag'
            else:  # country
                icon_color = 'red'
                icon_type = 'globe'
            
            # Scale marker size based on post count
            marker_size = min(30, 10 + marker['count'] * 2)
            
            folium.Marker(
                [marker['lat'], marker['lon']],
                popup=folium.Popup(marker['popup'], max_width=300),
                icon=folium.Icon(color=icon_color, icon=icon_type, prefix='fa')
            ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Display the map
        st.subheader(f"🌍 Advanced Geographical Heatmap for '{topic}'")
        folium_static(m, width=1300, height=600)
        
        # Display location statistics
        st.subheader("📍 Location Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Top Cities
            city_stats = posts_df[pd.notna(posts_df['city']) & (posts_df['city'] != '')]
            if not city_stats.empty:
                top_cities = city_stats['city'].value_counts().head(5)
                st.metric("🏙️ Top City", f"{top_cities.index[0]} ({top_cities.iloc[0]} posts)")
        
        with col2:
            # Top States
            state_stats = posts_df[pd.notna(posts_df['state']) & (posts_df['state'] != '')]
            if not state_stats.empty:
                top_states = state_stats['state'].value_counts().head(5)
                st.metric("🏛️ Top State", f"{top_states.index[0]} ({top_states.iloc[0]} posts)")
        
        with col3:
            # Top Countries
            country_stats = posts_df[pd.notna(posts_df['country']) & (posts_df['country'] != '')]
            if not country_stats.empty:
                top_countries = country_stats['country'].value_counts().head(5)
                st.metric("🌍 Top Country", f"{top_countries.index[0]} ({top_countries.iloc[0]} posts)")
        
        # Detailed location breakdown
        st.subheader("📊 Detailed Location Breakdown")
        
        tab1, tab2, tab3 = st.tabs(["🏙️ Cities", "🏛️ States", "🌍 Countries"])
        
        with tab1:
            city_breakdown = posts_df[pd.notna(posts_df['city']) & (posts_df['city'] != '')]
            if not city_breakdown.empty:
                city_counts = city_breakdown['city'].value_counts().reset_index()
                city_counts.columns = ['City', 'Post Count']
                fig_cities = px.bar(city_counts.head(10), x='City', y='Post Count',
                                  title=f'Top 10 Cities for "{topic}"',
                                  color='Post Count',
                                  color_continuous_scale='viridis')
                st.plotly_chart(fig_cities, use_container_width=True)
            else:
                st.info("No city data available")
        
        with tab2:
            state_breakdown = posts_df[pd.notna(posts_df['state']) & (posts_df['state'] != '')]
            if not state_breakdown.empty:
                state_counts = state_breakdown['state'].value_counts().reset_index()
                state_counts.columns = ['State', 'Post Count']
                fig_states = px.bar(state_counts.head(10), x='State', y='Post Count',
                                  title=f'Top 10 States for "{topic}"',
                                  color='Post Count',
                                  color_continuous_scale='plasma')
                st.plotly_chart(fig_states, use_container_width=True)
            else:
                st.info("No state data available")
        
        with tab3:
            country_breakdown = posts_df[pd.notna(posts_df['country']) & (posts_df['country'] != '')]
            if not country_breakdown.empty:
                country_counts = country_breakdown['country'].value_counts().reset_index()
                country_counts.columns = ['Country', 'Post Count']
                fig_countries = px.bar(country_counts.head(10), x='Country', y='Post Count',
                                     title=f'Top 10 Countries for "{topic}"',
                                     color='Post Count',
                                     color_continuous_scale='rainbow')
                st.plotly_chart(fig_countries, use_container_width=True)
                
                # World map choropleth
                fig_world = px.choropleth(country_counts, 
                                        locations='Country',
                                        locationmode='country names',
                                        color='Post Count',
                                        hover_name='Country',
                                        color_continuous_scale='viridis',
                                        title=f'Global Distribution for "{topic}"')
                st.plotly_chart(fig_world, use_container_width=True)
            else:
                st.info("No country data available")
        
        # Platform distribution by location
        st.subheader("📱 Platform Distribution by Location")
        
        if not posts_df.empty:
            # Platform by country
            platform_country = posts_df.groupby(['country', 'platform']).size().reset_index(name='count')
            if not platform_country.empty:
                fig_platform_country = px.sunburst(platform_country, 
                                                 path=['country', 'platform'], 
                                                 values='count',
                                                 title=f'Platform Distribution by Country for "{topic}"')
                st.plotly_chart(fig_platform_country, use_container_width=True)
            
            # Sentiment by location
            sentiment_location = posts_df.groupby(['country', 'sentiment']).size().reset_index(name='count')
            if not sentiment_location.empty:
                fig_sentiment_location = px.treemap(sentiment_location,
                                                  path=['country', 'sentiment'],
                                                  values='count',
                                                  title=f'Sentiment Distribution by Country for "{topic}"',
                                                  color='sentiment',
                                                  color_discrete_map={
                                                      'Happy': '#00FF00',
                                                      'Positive': '#90EE90',
                                                      'Neutral': '#FFFF00',
                                                      'Negative': '#FFA500',
                                                      'Sad': '#FF6347',
                                                      'Disaster': '#FF0000'
                                                  })
                st.plotly_chart(fig_sentiment_location, use_container_width=True)
                
    except Exception as e:
        st.error(f"Could not generate advanced geographical analysis: {str(e)}")

def detect_crisis(data, topic):
    try:
        posts = data['posts']
        
        # Count disaster sentiment posts
        disaster_posts = posts[posts['sentiment'] == 'Disaster']
        disaster_count = len(disaster_posts)
        total_posts = len(posts)
        crisis_score = min(100, (disaster_count / total_posts) * 1000) if total_posts > 0 else 0
        
        # Create a prominent crisis alert section
        st.markdown("---")
        st.subheader("🚨 Crisis Detection & Alert System")
        
        # Crisis score with color coding
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if crisis_score > 75:
                st.error(f"🔴 EXTREME CRISIS ALERT - Score: {crisis_score:.1f}")
                st.error("Immediate attention required! High concentration of disaster-related content detected.")
            elif crisis_score > 55:
                st.warning(f"🟠 HIGH ALERT - Score: {crisis_score:.1f}")
                st.warning("Elevated crisis indicators detected. Monitor closely.")
            elif crisis_score > 25:
                st.info(f"🟡 MODERATE ALERT - Score: {crisis_score:.1f}")
                st.info("Some crisis-related content detected. Stay vigilant.")
            else:
                st.success(f"🟢 NORMAL - Score: {crisis_score:.1f}")
                st.success("No significant crisis indicators detected.")
        
        # Crisis metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Disaster-related Posts", f"{disaster_count}/{total_posts}")
        with col2:
            st.metric("Crisis Detection Score", f"{crisis_score:.1f}")
        
        if disaster_count > 0:
            with st.expander("🔍 View disaster-related posts"):
                st.dataframe(disaster_posts[['text', 'sentiment', 'platform']].head(15))
                
    except Exception as e:
        st.error(f"Crisis detection error: {str(e)}")
        
# Streamlit App
def main():
    st.set_page_config(
        page_title="Social Media Hashtag Analysis",
        layout="wide",
        page_icon="📊",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .contributor-section {
            # background-color: #5c0899;
            background-image: linear-gradient(to right, #495aff 10%, #0acffe 100%);
            # background-image: linear-gradient(to right, #6a11cb 10%, #2575fc 100%);
            # background-image: linear-gradient(to right, #f9d423 0%, #f83600 100%);
            # background-image: linear-gradient(-60deg, #ff5858 0%, #f09819 100%);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .crisis-alert {
            background-color: #ffcccc;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #ff0000;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header Section
    st.markdown('<h1 class="main-header">📊 Social Media Hashtag Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now() - timedelta(minutes=DEFAULT_REFRESH + 1)
    if 'current_topic' not in st.session_state:
        st.session_state.current_topic = ""
    if 'search_history' not in st.session_state:
        st.session_state.search_history = get_search_history()
    if 'selected_platform' not in st.session_state:
        st.session_state.selected_platform = "Database (CSV)"
    
    # Right Sidebar
    with st.sidebar:
        # Contributor Information at top
        st.markdown("""
            <div class='contributor-section'>
                <h3>👨‍💻 Contributor </h3>
                <p><strong>Harshal Shinde - </strong>
                <em>M24DE3037</em></p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.header("🎛️ Dashboard Controls")
        
        # Data source selection
        st.subheader("Data Source")
        data_source = st.radio(
            "Select Data Source:",
            ["📁 Database (CSV)", "🌐 Live API"],
            index=0,
            help="Choose between CSV database or live API data"
        )
        
        # API-specific controls
        if data_source == "🌐 Live API":
            st.subheader("API Platform")
            api_platform = st.radio(
                "Select Platform:",
                ["𝕏 Twitter/𝕏", "🦋 Bluesky", "🐘 Mastodon", "🕹️ Reddit"],
                index=0
            )
            
            refresh_interval = st.slider(
                "🕐 Auto-refresh interval (minutes)",
                min_value=15,
                max_value=120,
                value=DEFAULT_REFRESH,
                step=5
            )
            
            post_limit = st.slider(
                "📊 Max posts per request",
                min_value=10,
                max_value=100,
                value=20,
                step=5
            )
            
            # Manual refresh button
            if st.button("🔄 Refresh Now"):
                st.session_state.last_refresh = datetime.now() - timedelta(minutes=refresh_interval + 1)
                st.rerun()
        
        st.markdown("---")
        
        # CSV import section
        st.subheader("📤 Import CSV Data")
        uploaded_file = st.file_uploader(
            "Upload social media data CSV", 
            type=["csv"],
            help="Expected columns: Text, Timestamp, User, Hashtags, Retweets, Likes, Country, Platform"
        )
        
        if uploaded_file:
            topic_name = st.text_input("Enter topic name for this import")
            if st.button("🚀 Import to Database") and topic_name:
                with st.spinner(f"Importing posts about '{topic_name}'..."):
                    success, imported_df = import_from_csv(uploaded_file, topic_name)
                    if success:
                        st.success(f"✅ Successfully imported {len(imported_df)} posts about '{topic_name}'")
                        if topic_name not in st.session_state.search_history:
                            st.session_state.search_history.insert(0, topic_name)
                            st.session_state.search_history = st.session_state.search_history[:10]
                        st.rerun()

        st.markdown("---")
        
        # Search history
        st.subheader("📚 Search History")
        for topic in st.session_state.search_history:
            if st.button(f"{topic}", key=f"history_{topic}"):
                st.session_state.current_topic = topic
                st.rerun()
    
    # Main content area
    col1, col2 = st.columns([4, 1])
    with col1:
        topic = st.text_input(
            "🔍 Enter a hashtag or topic to analyze:",
            value=st.session_state.current_topic,
            key="search_input",
            placeholder="e.g., DelhiCarBlast, DubaiAirshow, GlobalEconomicForum"
        )
    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        if st.button("🚀 Analyze", use_container_width=True):
            if data_source == "🌐 Live API" and api_platform != "🐦 Twitter":
                st.error("🚫 API Limit Exceeded: Live API access is unavailable because the pull request limit has been reached. Please switch to Database mode or try again later.")
            else:
                st.session_state.current_topic = topic
                if topic and topic not in st.session_state.search_history:
                    st.session_state.search_history.insert(0, topic)
                    if len(st.session_state.search_history) > 10:
                        st.session_state.search_history = st.session_state.search_history[:10]
                if data_source == "🌐 Live API":
                    st.session_state.last_refresh = datetime.now() - timedelta(minutes=refresh_interval + 1)
                st.rerun()
    
    # Display analysis if we have a topic
    if st.session_state.current_topic:
        current_topic = st.session_state.current_topic
        st.markdown(f"<h2 style='text-align: center; color: #1f77b4;'>Analysis for: #{current_topic}</h2>", unsafe_allow_html=True)
        
        # Get data based on selected source
        cached_data = get_cached_data(current_topic)
        should_fetch = (data_source == "🌐 Live API")
        
        if cached_data and not should_fetch:
            st.info("💾 Showing data from database. Switch to 'Live API' mode for real-time data.")
        
        # Fetch new data if needed (only for Twitter API)
        if should_fetch and api_platform == "🐦 Twitter":
            try:
                # This would be your Twitter API integration
                # For demo purposes, we'll show an info message
                st.info("🔗 Connected to Twitter API. Fetching real-time data...")
                # Actual API implementation would go here
                
            except Exception as e:
                st.error(f"Failed to fetch posts: {str(e)}")
                cached_data = get_cached_data(current_topic)
        
        # Display visualizations if we have data
        if cached_data:
            # Key Metrics Row
            st.markdown("---")
            st.subheader("📈 Key Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_posts = len(cached_data['posts'])
                st.metric("Total Posts", total_posts)
            
            with col2:
                positive = cached_data['analytics'][
                    cached_data['analytics']['metric'] == 'sentiment_Positive'
                ]['value'].iloc[0] if 'sentiment_Positive' in cached_data['analytics']['metric'].values else 0
                st.metric("Positive Posts", positive)
            
            with col3:
                avg_retweets = round(cached_data['posts']['retweet_count'].mean(), 1)
                st.metric("Avg Engagement", avg_retweets)
            
            with col4:
                platforms = cached_data['posts']['platform'].nunique()
                st.metric("Platforms", platforms)
            
            # Crisis Detection (Highlighted Feature)
            detect_crisis(cached_data, current_topic)
            
            # Sentiment and Platform Analysis
            st.markdown("---")
            st.subheader("🎭 Enhanced Sentiment & Platform Analysis")
            col1, col2 = st.columns(2)
            with col1:
                plot_sentiment(cached_data, current_topic)
            with col2:
                plot_platform_distribution(cached_data, current_topic)
            
            # Time Series and Hashtags
            st.markdown("---")
            st.subheader("📊 Content Analysis")
            col1, col2 = st.columns(2)
            with col1:
                plot_post_volume(cached_data, current_topic)
            with col2:
                plot_top_hashtags(cached_data, current_topic)
            
            # User Analysis
            st.markdown("---")
            st.subheader("👥 User Analysis")
            plot_top_users(cached_data, current_topic)
            
            # Engagement Analysis
            st.markdown("---")
            st.subheader("💬 Engagement Analysis")
            plot_engagement(cached_data, current_topic)
            
            # Advanced NLP Features
            st.markdown("---")
            st.subheader("🔍 Advanced NLP Analysis")
            plot_entities(cached_data, current_topic)
            
            # Word Cloud
            st.markdown("---")
            st.subheader("☁️ Word Cloud Analysis")
            plot_wordcloud(cached_data, current_topic)
            
            # Geographical Heatmap
            st.markdown("---")
            plot_geographical_heatmap(cached_data, current_topic)
            
        else:
            st.warning(f"❌ No data found for topic '{current_topic}'. Please import CSV data first or check your search term.")
    
    # Auto-refresh component (only for API mode)
    if data_source == "🌐 Live API":
        st_autorefresh(interval=refresh_interval * 60 * 1000, key="auto_refresh")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "📊 Social Media Hashtag Analysis Dashboard | "
        "Built with Streamlit | "
        "Contributor: Harshal Shinde (M24DE3037)"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()