from Movie_recommender_GUI import run_gui
import pandas as pd
import numpy as np

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
all_genres = sorted({g for sublist in movies_df['genre'].str.split('|') for g in sublist})
genre_matrix = pd.DataFrame([
    [int(g in genre_str.split('|')) for g in all_genres]
    for genre_str in movies_df['genre']
], index=movies_df.index, columns=all_genres)

if __name__ == "__main__":
    run_gui(movies_df, ratings_df, genre_matrix)