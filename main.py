"""
Movie Recommendation System with CLI Input, Advanced Features & Visualization
----------------------------------------------------------------------------
- Supports collaborative filtering (user-based) and content-based filtering (genre-based).
- CLI: User chooses recommendation type, number of recommendations, and can rate new movies.
- Visualizes similarity matrices and user rating distributions.
- Explains recommendations.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

# --- Data Setup ---
movies_data = {
    "title": [
        "Inception", "The Dark Knight", "Interstellar", "Pulp Fiction", "Forrest Gump",
        "Fight Club", "The Matrix", "Memento", "Parasite", "Whiplash"
    ],
    "genre": [
        "Sci-Fi|Thriller", "Action|Crime", "Sci-Fi|Drama", "Crime|Drama", "Drama|Romance",
        "Drama|Thriller", "Sci-Fi|Action", "Thriller|Mystery", "Thriller|Drama", "Drama|Music"
    ]
}
movies_df = pd.DataFrame(movies_data).set_index("title")
ratings_dict = {
    'User1': [5, 4, np.nan, 2, 1, np.nan, np.nan, 5, np.nan, 4],
    'User2': [4, np.nan, 3, 2, np.nan, 1, 5, np.nan, 4, np.nan],
    'User3': [np.nan, 5, 4, np.nan, 2, 3, np.nan, 4, 5, 1],
    'User4': [3, np.nan, np.nan, 4, 5, 2, 1, np.nan, 4, 5],
    'User5': [np.nan, 2, 4, np.nan, 3, 4, 5, 1, np.nan, 2],
}
ratings_df = pd.DataFrame(ratings_dict, index=movies_df.index)

# --- Genre Encoding ---
all_genres = sorted({g for sublist in movies_df['genre'].str.split('|') for g in sublist})
genre_matrix = pd.DataFrame([
    [int(g in genre_str.split('|')) for g in all_genres]
    for genre_str in movies_df['genre']
], index=movies_df.index, columns=all_genres)

# --- Visualization ---
def plot_similarity_matrix(matrix, title="Similarity Matrix", labels=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, cmap="coolwarm", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_user_ratings(ratings_df):
    plt.figure(figsize=(8, 5))
    ratings_df.T.plot(kind="bar", stacked=True)
    plt.title("User Ratings Distribution")
    plt.xlabel("User")
    plt.ylabel("Ratings")
    plt.tight_layout()
    plt.show()

# --- Collaborative Filtering ---
def recommend_collaborative(target_user, ratings, top_n=3, explain=False):
    ratings_filled = ratings.fillna(0).T
    user_sim_matrix = pd.DataFrame(
        cosine_similarity(ratings_filled),
        index=ratings_filled.index,
        columns=ratings_filled.index
    )
    sim_scores = user_sim_matrix.loc[target_user].drop(target_user)
    similar_users = sim_scores.sort_values(ascending=False).head(2).index
    unrated_movies = ratings[ratings[target_user].isna()].index
    scores = {}
    explain_text = []
    for movie in unrated_movies:
        rating_sum = 0
        sim_sum = 0
        user_details = []
        for sim_user in similar_users:
            rating = ratings.loc[movie, sim_user]
            if not np.isnan(rating):
                rating_sum += rating * sim_scores[sim_user]
                sim_sum += sim_scores[sim_user]
                user_details.append(f"{sim_user} (rated {rating}, sim {sim_scores[sim_user]:.2f})")
        scores[movie] = rating_sum / sim_sum if sim_sum > 0 else 0
        if explain and user_details:
            explain_text.append(f"{movie}: recommendations influenced by {', '.join(user_details)}")
    recommended = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    if explain:
        return [movie for movie, score in recommended], explain_text, user_sim_matrix
    else:
        return [movie for movie, score in recommended], None, user_sim_matrix

# --- Content-Based Filtering ---
def recommend_content_based(movie_title, genre_matrix, top_n=3, explain=False):
    if movie_title not in genre_matrix.index:
        return [], None, None
    target_vec = genre_matrix.loc[movie_title].values.reshape(1, -1)
    similarities = cosine_similarity(target_vec, genre_matrix.values)[0]
    similar_idx = np.argsort(similarities)[::-1]
    recs = []
    explain_text = []
    for idx in similar_idx:
        title = genre_matrix.index[idx]
        if title != movie_title:
            recs.append(title)
            if explain:
                overlap = [g for g in all_genres if genre_matrix.loc[movie_title, g] and genre_matrix.loc[title, g]]
                explain_text.append(f"{title}: shares genres {', '.join(overlap)} with {movie_title}")
        if len(recs) == top_n:
            break
    # For visualization
    genre_sim_matrix = pd.DataFrame(
        cosine_similarity(genre_matrix.values),
        index=genre_matrix.index,
        columns=genre_matrix.index
    )
    return recs, explain_text, genre_sim_matrix

# --- CLI Interface ---
def main():
    print("Welcome to the Advanced Movie Recommendation System!")
    print("Choose recommendation type:")
    print("1. Collaborative Filtering (by user)")
    print("2. Content-Based Filtering (by movie genres)")
    choice = input("Enter 1 or 2: ").strip()
    try:
        top_n = int(input("How many recommendations would you like? [Default 3]: ").strip() or "3")
    except ValueError:
        top_n = 3

    visualize = input("Visualize similarity matrix? (y/n): ").strip().lower() == 'y'
    explain = input("Explain recommendations? (y/n): ").strip().lower() == 'y'

    if choice == "1":
        print("\nAvailable users:", ', '.join(ratings_df.columns))
        user = input("Enter user name: ").strip()
        if user not in ratings_df.columns:
            print("User not found.")
            return
        recs, explain_text, user_sim_matrix = recommend_collaborative(user, ratings_df, top_n=top_n, explain=explain)
        print(f"\nRecommendations for {user}:")
        for movie in recs:
            print(f"- {movie}")
        if explain and explain_text:
            print("\nWhy these movies?")
            for line in explain_text:
                print(line)
        if visualize:
            plot_similarity_matrix(user_sim_matrix, "User-User Similarity", labels=ratings_df.columns)
            plot_user_ratings(ratings_df)

        rate_new = input("\nWould you like to rate a new movie? (y/n): ").strip().lower() == 'y'
        if rate_new:
            unrated = ratings_df[ratings_df[user].isna()].index
            print("Unrated movies:", ', '.join(unrated))
            movie = input("Choose a movie to rate: ").strip()
            if movie not in unrated:
                print("Movie not in unrated list.")
            else:
                try:
                    rating = float(input("Your rating (1-5): ").strip())
                    ratings_df.loc[movie, user] = rating
                    print(f"Rated {movie} with {rating}.")
                except:
                    print("Invalid rating.")

    elif choice == "2":
        print("\nAvailable movies:", ', '.join(movies_df.index))
        movie = input("Type a movie you like: ").strip()
        if movie not in movies_df.index:
            print("Movie not found.")
            return
        recs, explain_text, genre_sim_matrix = recommend_content_based(movie, genre_matrix, top_n=top_n, explain=explain)
        print(f"\nMovies similar to '{movie}':")
        for m in recs:
            genres = movies_df.loc[m, 'genre']
            print(f"- {m} ({genres})")
        if explain and explain_text:
            print("\nWhy these movies?")
            for line in explain_text:
                print(line)
        if visualize and genre_sim_matrix is not None:
            plot_similarity_matrix(genre_sim_matrix, "Movie-Genre Similarity", labels=genre_sim_matrix.columns)

    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()