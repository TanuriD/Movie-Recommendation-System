import streamlit as st
import pandas as pd

# Import project functions
from collaborative import train_model, recommend_movies
from content_based import load_movies, build_content_model, get_recommendations


# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# -------------------------------------------------
# Load Data
# -------------------------------------------------

@st.cache_data
def load_movie_data():
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")

    return movies, ratings


# -------------------------------------------------
# Train Collaborative Filtering Model
# -------------------------------------------------

@st.cache_resource
def load_collaborative_model():
    model, ratings = train_model()
    return model, ratings


# -------------------------------------------------
# Build Content-Based Model
# -------------------------------------------------

@st.cache_resource
def load_content_model(movies):
    similarity_matrix = build_content_model(movies)
    return similarity_matrix


# -------------------------------------------------
# Load everything
# -------------------------------------------------

movies, ratings = load_movie_data()


# -------------------------------------------------
# Header
# -------------------------------------------------

st.title("🎬 Movie Recommendation System")

st.write(
    "Get personalized movie recommendations using "
    "Collaborative Filtering and Content-Based Filtering."
)


# -------------------------------------------------
# Sidebar
# -------------------------------------------------

st.sidebar.header("⚙️ Recommendation Settings")

method = st.sidebar.selectbox(
    "Choose Recommendation Method",
    [
        "Collaborative Filtering",
        "Content-Based Filtering"
    ]
)


# =================================================
# COLLABORATIVE FILTERING
# =================================================

if method == "Collaborative Filtering":

    st.header("🤖 Collaborative Filtering")

    st.write(
        "Recommendations are generated using an SVD "
        "matrix factorization model based on user ratings."
    )

    # Get available users
    user_ids = sorted(ratings["userId"].unique())

    user_id = st.selectbox(
        "Select User ID",
        user_ids
    )

    number_of_movies = st.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=20,
        value=10
    )

    if st.button("🎯 Get Recommendations"):

        with st.spinner("Training SVD model and generating recommendations..."):

            model, rating_data = load_collaborative_model()

            recommendations = recommend_movies(
                model=model,
                user_id=user_id,
                movies_df=movies,
                ratings_df=rating_data,
                n=number_of_movies
            )

        st.success("Recommendations generated successfully!")

        st.subheader(
            f"🎥 Top {number_of_movies} Recommendations for User {user_id}"
        )

        st.dataframe(
            recommendations[
                ["movieId", "title", "genres"]
            ],
            use_container_width=True,
            hide_index=True
        )


# =================================================
# CONTENT-BASED FILTERING
# =================================================

elif method == "Content-Based Filtering":

    st.header("🎭 Content-Based Filtering")

    st.write(
        "Recommendations are generated based on movie genres "
        "using TF-IDF and cosine similarity."
    )

    # Movie selector
    movie_titles = movies["title"].tolist()

    selected_movie = st.selectbox(
        "Select a Movie",
        movie_titles
    )

    number_of_movies = st.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=20,
        value=10
    )

    if st.button("🎯 Get Recommendations"):

        with st.spinner("Building content-based model..."):

            similarity_matrix = load_content_model(movies)

            recommendations = get_recommendations(
                title=selected_movie,
                movies_df=movies,
                similarity_matrix=similarity_matrix,
                n=number_of_movies
            )

        st.success("Recommendations generated successfully!")

        st.subheader(
            f"🎬 Movies Similar to {selected_movie}"
        )

        st.dataframe(
            recommendations,
            use_container_width=True,
            hide_index=True
        )


# =================================================
# DATASET INFORMATION
# =================================================

st.sidebar.markdown("---")

st.sidebar.subheader("📊 Dataset Information")

st.sidebar.write(
    f"🎬 Movies: {len(movies):,}"
)

st.sidebar.write(
    f"👤 Users: {ratings['userId'].nunique():,}"
)

st.sidebar.write(
    f"⭐ Ratings: {len(ratings):,}"
)

st.sidebar.write(
    f"🎭 Genres: {movies['genres'].nunique():,}"
)


# =================================================
# Footer
# =================================================

st.markdown("---")

st.caption(
    "Movie Recommendation System | "
    "Content-Based + Collaborative Filtering"
)