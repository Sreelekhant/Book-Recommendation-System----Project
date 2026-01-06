import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Book Recommendation System", layout="wide")

# ---------------- LOAD DATA (CACHED) ----------------
@st.cache_data
def load_data():
    books = pd.read_csv("Books.csv")
    ratings = pd.read_csv("Ratings.csv",encoding='latin-1')
    return books, ratings

books, ratings = load_data()
ratings_with_books = ratings.merge(books, on="ISBN")

# ---------------- POPULARITY BASED ----------------
@st.cache_data
def compute_popular_df(df):
    popular = (
        df.groupby(['Book-Title', 'Book-Author', 'Image-URL-M'])
        .agg(num_ratings=('Book-Rating', 'count'),
             avg_rating=('Book-Rating', 'mean'))
        .reset_index()
    )
    return popular[popular['num_ratings'] >= 100] \
        .sort_values('avg_rating', ascending=False)

popular_df = compute_popular_df(ratings_with_books)

# ---------------- COLLABORATIVE FILTERING ----------------
@st.cache_data
def create_pivot(df):
    filtered = df.groupby('User-ID').filter(lambda x: len(x) >= 50)
    pt = filtered.pivot_table(
        index='Book-Title',
        columns='User-ID',
        values='Book-Rating'
    ).fillna(0)
    return pt.astype(np.float32)   # 🔥 BIG SPEED BOOST

pt = create_pivot(ratings_with_books)

# ---------------- UI ----------------
st.title(" Book Recommendation System")

tab1, tab2 = st.tabs([" Popular Books", " Recommend Books"])

with tab1:
    cols = st.columns(5)
    for i in range(min(20, len(popular_df))):
        with cols[i % 5]:
            st.image(popular_df.iloc[i]['Image-URL-M'], use_container_width=True)
            st.markdown(f"**{popular_df.iloc[i]['Book-Title']}**")
            st.caption(popular_df.iloc[i]['Book-Author'])

with tab2:
    book_name = st.selectbox("Select a book", pt.index)

    if st.button("Recommend"):
        book_vector = pt.loc[book_name].values.reshape(1, -1)

        # similarity only computed once per click (cached pivot)
        similarity = cosine_similarity(book_vector, pt.values)[0]

        top_indices = np.argsort(similarity)[-6:-1][::-1]

        cols = st.columns(5)
        for col, idx in zip(cols, top_indices):
            temp = books[books['Book-Title'] == pt.index[idx]]
            col.image(temp.iloc[0]['Image-URL-M'], use_container_width=True)
            col.markdown(f"**{temp.iloc[0]['Book-Title']}**")
            col.caption(temp.iloc[0]['Book-Author'])
