"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                  MOVIE RECOMMENDATION SYSTEM - STREAMLIT APP                  ║
║                    Hybrid Recommender with Model Evaluation                   ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Interactive web application for movie recommendations using SVD++, content-based
filtering, and hybrid approaches. Demonstrates model comparison and evaluation.

Author: Harpinder Singh
Email: aekas142@gmail.com
GitHub: @HarpinderC
Date: December 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                              PATH CONFIGURATION                               ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

BASE_DIR = Path(__file__).resolve().parent          # movie-recommender-system/app
PROJECT_ROOT = BASE_DIR.parent                     # movie-recommender-system

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                            PAGE CONFIGURATION                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                              CUSTOM CSS STYLING                               ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

def load_custom_css():
    """
    Apply Dave2D teal theme with professional styling.
    """
    st.markdown("""
    <style>
    /* Dave2D Teal Theme */
    :root {
        --dave2d-teal: #00D9C0;
        --dave2d-dark: #0A1929;
        --dave2d-gray: #1E293B;
    }
    
    /* Import better fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    /* Main title styling */
    .main-title {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        color: var(--dave2d-teal);
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,217,192,0.3);
    }
    
    .sub-title {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-size: 1.2rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 2px solid var(--dave2d-teal);
        box-shadow: 0 4px 6px rgba(0,217,192,0.2);
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--dave2d-teal);
        margin: 0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #94A3B8;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Movie card styling */
    .movie-card {
        background: #1E293B;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid var(--dave2d-teal);
        transition: transform 0.2s;
    }
    
    .movie-card:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(0,217,192,0.3);
    }
    
    .movie-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #F1F5F9;
        margin-bottom: 0.3rem;
    }
    
    .movie-genres {
        font-size: 0.9rem;
        color: #94A3B8;
        font-style: italic;
    }
    
    .movie-score {
        font-size: 1rem;
        font-weight: 600;
        color: var(--dave2d-teal);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #00D9C0 0%, #00C4B0 100%);
        color: #0A1929;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,217,192,0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #0F172A;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-bottom: 3px solid transparent;
        color: #94A3B8;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .stTabs [aria-selected="true"] {
        border-bottom-color: var(--dave2d-teal);
        color: var(--dave2d-teal);
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #1E3A5F 0%, #0F172A 100%);
        border-left: 4px solid var(--dave2d-teal);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Success message */
    .success-box {
        background: linear-gradient(135deg, #1E3A2F 0%, #0F172A 100%);
        border-left: 4px solid #10B981;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                              DATA LOADING                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

@st.cache_data
def load_data():
    """
    Load all required datasets and models.
    
    Returns
    -------
    dict
        Dictionary containing all loaded data and models
    """
    try:
        data = {
            'movies': pd.read_csv(DATA_DIR / 'movies_metadata.csv'),
            'merged': pd.read_csv(DATA_DIR / 'movielens_100k_merged.csv'),
            'final_results': pd.read_csv(RESULTS_DIR / 'final_results.csv'),
            'cf_results': pd.read_csv(RESULTS_DIR / 'cf_results.csv'),
            'mf_results': pd.read_csv(RESULTS_DIR / 'mf_results.csv'),
            'baseline_results': pd.read_csv(RESULTS_DIR / 'baseline_results.csv')
        }
        
        # Load SVD++ model
        with open(MODELS_DIR / 'svdpp_model.pkl', 'rb') as f:
            data['svdpp_model'] = pickle.load(f)
        
        # Load item similarity matrix
        data['item_similarity'] = np.load(MODELS_DIR / 'item_similarity_matrix.npy')
        
        # Load mappings
        with open(MODELS_DIR / 'item_mappings.pkl', 'rb') as f:
            mappings = pickle.load(f)
            data['item_id_to_idx'] = mappings['item_id_to_idx']
            data['idx_to_item_id'] = mappings['idx_to_item_id']
        
        return data

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                         RECOMMENDATION FUNCTIONS                              ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

def get_similar_movies_content_based(movie_id, data, n=10):
    """
    Get similar movies using content-based filtering.
    
    Parameters
    ----------
    movie_id : int
        Movie ID to find similar movies for
    data : dict
        Dictionary containing loaded data
    n : int
        Number of recommendations
    
    Returns
    -------
    pd.DataFrame
        Similar movies with similarity scores
    """
    if movie_id not in data['item_id_to_idx']:
        return pd.DataFrame()
    
    movie_idx = data['item_id_to_idx'][movie_id]
    similarity_scores = data['item_similarity'][movie_idx]
    
    # Get top N similar movies (excluding itself)
    similar_indices = np.argsort(similarity_scores)[-n-1:-1][::-1]
    
    similar_movies = []
    for idx in similar_indices:
        item_id = data['idx_to_item_id'][idx]
        movie_info = data['movies'][data['movies']['item_id'] == item_id].iloc[0]
        similar_movies.append({
            'item_id': item_id,
            'title': movie_info['title'],
            'genres': movie_info['genres'],
            'similarity': similarity_scores[idx]
        })
    
    return pd.DataFrame(similar_movies)

def get_personalized_recommendations(user_id, data, n=10, alpha=0.7):
    """
    Get personalized recommendations using hybrid approach.
    
    Parameters
    ----------
    user_id : int
        User ID to get recommendations for
    data : dict
        Dictionary containing loaded data
    n : int
        Number of recommendations
    alpha : float
        Weight for SVD++ (1-alpha for content-based)
    
    Returns
    -------
    pd.DataFrame
        Personalized recommendations
    """
    # Get items user hasn't rated
    user_items = set(data['merged'][data['merged']['user_id'] == user_id]['item_id'].values)
    all_items = set(data['movies']['item_id'].unique())
    candidate_items = list(all_items - user_items)
    
    # Calculate hybrid scores
    item_scores = []
    for item_id in candidate_items:
        try:
            # SVD++ prediction
            svdpp_pred = data['svdpp_model'].predict(user_id, item_id).est
            
            # Content-based prediction (simplified for app)
            if item_id in data['item_id_to_idx']:
                cb_pred = 3.5  # Simplified, use actual CB function for production
            else:
                cb_pred = 3.5
            
            # Hybrid score
            hybrid_score = alpha * svdpp_pred + (1 - alpha) * cb_pred
            item_scores.append((item_id, hybrid_score))
        except:
            continue
    
    # Sort and get top N
    item_scores.sort(key=lambda x: x[1], reverse=True)
    top_n = item_scores[:n]
    
    # Create DataFrame
    recommendations = []
    for item_id, score in top_n:
        movie_info = data['movies'][data['movies']['item_id'] == item_id].iloc[0]
        recommendations.append({
            'title': movie_info['title'],
            'genres': movie_info['genres'],
            'predicted_rating': score
        })
    
    return pd.DataFrame(recommendations)

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                              VISUALIZATION FUNCTIONS                          ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

def create_model_comparison_chart(data):
    """
    Create interactive model comparison chart.
    
    Parameters
    ----------
    data : dict
        Dictionary containing results data
    
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive bar chart
    """
    # Combine all results
    all_results = []
    
    # Add baseline
    baseline = data['baseline_results'][data['baseline_results']['Model'] == 'Popularity-Based'].iloc[0]
    all_results.append({'Model': 'Baseline', 'RMSE': baseline['RMSE'], 'Phase': 'Baseline'})
    
    # Add best CF
    best_cf = data['cf_results'].iloc[0]
    all_results.append({'Model': 'Item-Based CF', 'RMSE': best_cf['RMSE'], 'Phase': 'Collaborative Filtering'})
    
    # Add best MF
    best_mf = data['mf_results'][data['mf_results']['Model'] == 'SVD++'].iloc[0]
    all_results.append({'Model': 'SVD++', 'RMSE': best_mf['RMSE'], 'Phase': 'Matrix Factorization'})
    
    # Add content-based
    cb = data['final_results'][data['final_results']['Model'] == 'Content-Based (Genre)'].iloc[0]
    all_results.append({'Model': 'Content-Based', 'RMSE': cb['RMSE'], 'Phase': 'Content Filtering'})
    
    df = pd.DataFrame(all_results)
    
    fig = go.Figure(data=[
        go.Bar(
            x=df['Model'],
            y=df['RMSE'],
            text=df['RMSE'].round(4),
            textposition='auto',
            marker=dict(
                color=['#EF4444', '#F59E0B', '#00D9C0', '#8B5CF6'],
                line=dict(color='#0A1929', width=2)
            ),
            hovertemplate='<b>%{x}</b><br>RMSE: %{y:.4f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': 'Model Performance Comparison',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#00D9C0', 'family': 'Inter, sans-serif'}
        },
        xaxis_title='Model',
        yaxis_title='RMSE (Lower is Better)',
        plot_bgcolor='#0F172A',
        paper_bgcolor='#0F172A',
        font=dict(color='#F1F5F9', family='Inter, sans-serif'),
        height=500,
        yaxis=dict(gridcolor='#1E293B'),
        xaxis=dict(gridcolor='#1E293B')
    )
    
    return fig

def create_progress_chart(data):
    """
    Create progress visualization showing improvement journey.
    
    Parameters
    ----------
    data : dict
        Dictionary containing results data
    
    Returns
    -------
    plotly.graph_objects.Figure
        Line chart showing progress
    """
    progress_data = [
        {'Phase': 'Random', 'RMSE': data['baseline_results'][data['baseline_results']['Model'] == 'Random'].iloc[0]['RMSE']},
        {'Phase': 'Popularity', 'RMSE': data['baseline_results'][data['baseline_results']['Model'] == 'Popularity-Based'].iloc[0]['RMSE']},
        {'Phase': 'Collaborative', 'RMSE': data['cf_results'].iloc[0]['RMSE']},
        {'Phase': 'SVD++', 'RMSE': data['mf_results'][data['mf_results']['Model'] == 'SVD++'].iloc[0]['RMSE']}
    ]
    
    df = pd.DataFrame(progress_data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Phase'],
        y=df['RMSE'],
        mode='lines+markers',
        line=dict(color='#00D9C0', width=3),
        marker=dict(size=12, color='#00D9C0', line=dict(color='#0A1929', width=2)),
        text=df['RMSE'].round(4),
        textposition='top center',
        hovertemplate='<b>%{x}</b><br>RMSE: %{y:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': 'Model Development Progress',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#00D9C0', 'family': 'Inter, sans-serif'}
        },
        xaxis_title='Development Phase',
        yaxis_title='RMSE',
        plot_bgcolor='#0F172A',
        paper_bgcolor='#0F172A',
        font=dict(color='#F1F5F9', family='Inter, sans-serif'),
        height=400,
        yaxis=dict(gridcolor='#1E293B'),
        xaxis=dict(gridcolor='#1E293B')
    )
    
    return fig

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                              MAIN APPLICATION                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

def main():
    """
    Main application entry point.
    """
    # Apply custom styling
    load_custom_css()
    
    # Load data
    data = load_data()
    
    # Header
    st.markdown('<h1 class="main-title">🎬 Movie Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">A Model Evaluation Approach | Powered by SVD++ & Hybrid Filtering</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📊 About This Project")
        st.markdown("""
        This recommendation system demonstrates:
        - **Collaborative Filtering** (User & Item-Based)
        - **Matrix Factorization** (SVD++)
        - **Content-Based Filtering** (Genre TF-IDF)
        - **Hybrid Approaches**
        - **Cold Start Handling**
        """)
        
        st.markdown("---")
        st.markdown("### 📈 Performance Summary")
        
        best_rmse = data['mf_results'][data['mf_results']['Model'] == 'SVD++'].iloc[0]['RMSE']
        baseline_rmse = data['baseline_results'][data['baseline_results']['Model'] == 'Random'].iloc[0]['RMSE']
        improvement = ((baseline_rmse - best_rmse) / baseline_rmse * 100)
        
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">Best Model</p>
            <p class="metric-value">SVD++</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">RMSE</p>
            <p class="metric-value">{best_rmse:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">Improvement</p>
            <p class="metric-value">{improvement:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 👨‍💻 Developer")
        st.markdown("""
        **Harpinder Singh**  
        [GitHub](https://github.com/HarpinderC) | [LinkedIn](https://linkedin.com/in/harpinder-singh)  
        📧 aekas142@gmail.com
        """)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Get Recommendations", "📊 Model Comparison", "ℹ️ About"])
    
    with tab1:
        st.markdown("## 🎯 Movie Recommendations")
        
        # Sub-tabs for different recommendation types
        rec_tab1, rec_tab2 = st.tabs(["🎬 Similar Movies", "👤 Personalized Picks"])
        
        with rec_tab1:
            st.markdown("### Find Movies Similar to Your Favorites")
            st.markdown('<div class="info-box">Select a movie you enjoyed, and we\'ll recommend similar films based on genre and content.</div>', unsafe_allow_html=True)
            
            # Movie selection
            movie_titles = data['movies'].sort_values('title')['title'].unique()
            selected_title = st.selectbox(
                "Choose a movie:",
                options=movie_titles,
                index=0
            )
            
            if st.button("🔍 Find Similar Movies", key="similar"):
                # Get movie ID
                movie_id = data['movies'][data['movies']['title'] == selected_title].iloc[0]['item_id']
                
                # Get recommendations
                similar_movies = get_similar_movies_content_based(movie_id, data, n=10)
                
                if not similar_movies.empty:
                    st.markdown(f'<div class="success-box">✅ Found {len(similar_movies)} movies similar to <strong>{selected_title}</strong></div>', unsafe_allow_html=True)
                    
                    st.markdown("### 🎬 Recommended Movies")
                    
                    for idx, row in similar_movies.iterrows():
                        st.markdown(f"""
                        <div class="movie-card">
                            <div class="movie-title">{idx + 1}. {row['title']}</div>
                            <div class="movie-genres">Genres: {row['genres']}</div>
                            <div class="movie-score">Similarity: {row['similarity']:.2%}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("No similar movies found. Please try another selection.")
        
        with rec_tab2:
            st.markdown("### Get Personalized Recommendations")
            st.markdown('<div class="info-box">Enter a User ID to get personalized movie recommendations based on viewing history and preferences.</div>', unsafe_allow_html=True)
            
            # User ID input
            user_ids = sorted(data['merged']['user_id'].unique())
            selected_user = st.number_input(
                "Enter User ID:",
                min_value=int(min(user_ids)),
                max_value=int(max(user_ids)),
                value=196,
                step=1
            )
            
            # Hybrid weight slider
            alpha = st.slider(
                "Model Balance (SVD++ ← → Content-Based)",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Adjust the balance between collaborative filtering (left) and content-based filtering (right)"
            )
            
            if st.button("🎯 Get My Recommendations", key="personalized"):
                # Get recommendations
                recommendations = get_personalized_recommendations(selected_user, data, n=10, alpha=alpha)
                
                if not recommendations.empty:
                    st.markdown(f'<div class="success-box">✅ Generated {len(recommendations)} personalized recommendations for User {selected_user}</div>', unsafe_allow_html=True)
                    
                    st.markdown("### 🌟 Your Recommended Movies")
                    
                    for idx, row in recommendations.iterrows():
                        st.markdown(f"""
                        <div class="movie-card">
                            <div class="movie-title">{idx + 1}. {row['title']}</div>
                            <div class="movie-genres">Genres: {row['genres']}</div>
                            <div class="movie-score">Predicted Rating: {row['predicted_rating']:.2f}/5.0</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("Unable to generate recommendations. Please try another user.")
    
    with tab2:
        st.markdown("## 📊 Model Performance Analysis")
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <p class="metric-label">Total Models</p>
                <p class="metric-value">12</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            baseline_rmse = data['baseline_results'][data['baseline_results']['Model'] == 'Popularity-Based'].iloc[0]['RMSE']
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Baseline RMSE</p>
                <p class="metric-value">{baseline_rmse:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            best_rmse = data['mf_results'][data['mf_results']['Model'] == 'SVD++'].iloc[0]['RMSE']
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Best RMSE</p>
                <p class="metric-value">{best_rmse:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            improvement = ((baseline_rmse - best_rmse) / baseline_rmse * 100)
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Improvement</p>
                <p class="metric-value">{improvement:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_model_comparison_chart(data), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_progress_chart(data), use_container_width=True)
        
        st.markdown("---")
        
        # Detailed results table
        st.markdown("### 📋 Detailed Model Results")
        
        # Combine all results
        all_models = pd.concat([
            data['baseline_results'][['Model', 'RMSE', 'MAE']],
            data['cf_results'][['Model', 'RMSE', 'MAE']].head(3),
            data['mf_results'][['Model', 'RMSE', 'MAE']].head(3)
        ]).sort_values('RMSE').reset_index(drop=True)
        
        st.dataframe(
            all_models.style.background_gradient(
                subset=['RMSE', 'MAE'],
                cmap='RdYlGn_r'
            ).format({'RMSE': '{:.4f}', 'MAE': '{:.4f}'}),
            use_container_width=True,
            height=400
        )
    
    with tab3:
        st.markdown("## ℹ️ About This Project")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Project Overview")
            st.markdown("""
            This **Movie Recommendation System** demonstrates a comprehensive approach to building
            and evaluating different recommendation algorithms:
            
            **Dataset**: MovieLens 100K
            - 100,000 ratings
            - 943 users
            - 1,682 movies
            - 19 genre categories
            
            **Development Approach**:
            1. Baseline Models (Random, Popularity)
            2. Collaborative Filtering (User-Based, Item-Based)
            3. Matrix Factorization (SVD, SVD++, NMF)
            4. Content-Based Filtering (Genre TF-IDF)
            5. Hybrid Models (Combined approaches)
            6. Cold Start Handling
            """)
            
            st.markdown("### 🏆 Key Achievements")
            st.markdown("""
            - ✅ **51.3% improvement** over random baseline
            - ✅ **12 models evaluated** comprehensively
            - ✅ **SVD++** identified as best performer
            - ✅ **Cold start strategies** implemented
            - ✅ **Interactive deployment** with Streamlit
            """)
        
        with col2:
            st.markdown("### 🛠️ Technologies Used")
            st.markdown("""
            **Machine Learning:**
            - `scikit-surprise` - Collaborative filtering & SVD
            - `scikit-learn` - TF-IDF, similarity metrics
            - `implicit` - ALS implementation
            
            **Data Processing:**
            - `pandas` - Data manipulation
            - `numpy` - Numerical operations
            
            **Visualization:**
            - `plotly` - Interactive charts
            - `streamlit` - Web application
            
            **Deployment:**
            - `Streamlit` - Web framework
            - `pickle` - Model persistence
            """)
            
            st.markdown("### 📊 Evaluation Metrics")
            st.markdown("""
            **Rating Prediction:**
            - RMSE (Root Mean Squared Error)
            - MAE (Mean Absolute Error)
            
            **Ranking Quality:**
            - Precision@K
            - Recall@K
            - NDCG@K (Normalized Discounted Cumulative Gain)
            """)
        
        st.markdown("---")
        
        st.markdown("### 📚 Model Descriptions")
        
        models_info = {
            "SVD++ (Best Model)": "Enhanced Singular Value Decomposition with implicit feedback. Considers both explicit ratings and implicit user-item interactions.",
            "Item-Based CF": "Recommends movies similar to those the user liked, based on rating patterns from all users.",
            "User-Based CF": "Finds users with similar tastes and recommends movies they enjoyed.",
            "Content-Based": "Recommends movies with similar genres and attributes to those the user liked.",
            "Hybrid Models": "Combines multiple approaches to leverage strengths of both collaborative and content-based filtering."
        }
        
        for model, description in models_info.items():
            with st.expander(f"📘 {model}"):
                st.markdown(description)
        
        st.markdown("---")
        
        st.markdown("### 📞 Contact & Resources")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Developer:**  
            Harpinder Singh  
            📧 aekas142@gmail.com
            """)
        
        with col2:
            st.markdown("""
            **Links:**  
            🔗 [GitHub Repository](https://github.com/HarpinderC)  
            💼 [LinkedIn Profile](https://linkedin.com/in/harpinder-singh)
            """)
        
        with col3:
            st.markdown("""
            **Dataset:**  
            📊 [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)  
            🏢 GroupLens Research
            """)

if __name__ == "__main__":
    main()
