import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommenderGUI(tk.Tk):
    def __init__(self, movies_df, ratings_df, genre_matrix):
        super().__init__()
        self.title("Movie Recommendation System")
        self.configure(bg="#e3f3ff")
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.genre_matrix = genre_matrix

        # Title label
        tk.Label(self, text="Movie Recommender", font=("Arial", 20, "bold"),
                 bg="#e3f3ff", fg="#0a3e62").pack(pady=12)

        # Filter type selection
        self.filter_type = tk.StringVar(value="Collaborative")
        filter_frame = tk.Frame(self, bg="#e3f3ff")
        filter_frame.pack(pady=5)
        tk.Radiobutton(filter_frame, text="Collaborative Filtering", variable=self.filter_type,
                       value="Collaborative", font=("Arial", 12), bg="#e3f3ff").pack(side=tk.LEFT)
        tk.Radiobutton(filter_frame, text="Content-Based Filtering", variable=self.filter_type,
                       value="Content", font=("Arial", 12), bg="#e3f3ff").pack(side=tk.LEFT)

        # Selection frame
        select_frame = tk.Frame(self, bg="#e3f3ff")
        select_frame.pack(pady=4)

        tk.Label(select_frame, text="User:", font=("Arial", 12), bg="#e3f3ff").grid(row=0, column=0)
        self.user_var = tk.StringVar(value=list(self.ratings_df.columns)[0])
        user_dropdown = ttk.Combobox(select_frame, textvariable=self.user_var,
                                     values=list(self.ratings_df.columns), width=10, state="readonly")
        user_dropdown.grid(row=0, column=1)

        tk.Label(select_frame, text="Movie:", font=("Arial", 12), bg="#e3f3ff").grid(row=1, column=0)
        self.movie_var = tk.StringVar(value=self.movies_df.index[0])
        movie_dropdown = ttk.Combobox(select_frame, textvariable=self.movie_var,
                                      values=list(self.movies_df.index), width=20, state="readonly")
        movie_dropdown.grid(row=1, column=1)

        # Number of recommendations entry
        tk.Label(self, text="Number of Recommendations:", font=("Arial", 12), bg="#e3f3ff").pack()
        self.top_n_var = tk.IntVar(value=3)
        top_n_entry = tk.Entry(self, textvariable=self.top_n_var, width=5)
        top_n_entry.pack()

        # Results box
        self.result_box = tk.Text(self, height=7, width=50, font=("Arial", 12), bg="#f9fcff", fg="#0a3e62")
        self.result_box.pack(pady=10)
        self.result_box.insert(tk.END, "Recommendations will appear here.")

        # Button frames: Two buttons per row for better layout
        # Row 1: Get Recommendations & Rate a New Movie
        btn_frame1 = tk.Frame(self, bg="#e3f3ff")
        btn_frame1.pack(pady=8)
        rec_btn = tk.Button(btn_frame1, text="Get Recommendations",
                            font=("Arial", 13), bg="#79b4f7", fg="#0a3e62",
                            command=self.recommend_click)
        rec_btn.pack(side=tk.LEFT, padx=8, ipadx=10, ipady=2)

        rate_btn = tk.Button(btn_frame1, text="Rate a New Movie",
                            font=("Arial", 13), bg="#ffa07a", fg="#0a3e62",
                            command=self.rate_new_movie)
        rate_btn.pack(side=tk.LEFT, padx=8, ipadx=10, ipady=2)

        # Row 2: Similarity and Distribution buttons
        btn_frame2 = tk.Frame(self, bg="#e3f3ff")
        btn_frame2.pack(pady=8)
        user_matrix_btn = tk.Button(btn_frame2, text="Show User-User Similarity",
                                    font=("Arial", 13), bg="#90ee90", fg="#0a3e62",
                                    command=self.user_user_matrix_click)
        user_matrix_btn.pack(side=tk.LEFT, padx=8, ipadx=10, ipady=2)

        genre_matrix_btn = tk.Button(btn_frame2, text="Show Movie-Genre Similarity",
                                     font=("Arial", 13), bg="#ffec7a", fg="#0a3e62",
                                     command=self.genre_matrix_click)
        genre_matrix_btn.pack(side=tk.LEFT, padx=8, ipadx=10, ipady=2)

        rating_dist_btn = tk.Button(btn_frame2, text="Show User Ratings Distribution",
                                   font=("Arial", 13), bg="#bda1ff", fg="#0a3e62",
                                   command=self.user_rating_dist_click)
        rating_dist_btn.pack(side=tk.LEFT, padx=8, ipadx=10, ipady=2)

    def recommend_collaborative(self, target_user, top_n=3):
        ratings_filled = self.ratings_df.fillna(0).T
        user_sim_matrix = pd.DataFrame(
            cosine_similarity(ratings_filled),
            index=ratings_filled.index,
            columns=ratings_filled.index
        )
        sim_scores = user_sim_matrix.loc[target_user].drop(target_user)
        similar_users = sim_scores.sort_values(ascending=False).head(2).index
        unrated_movies = self.ratings_df[self.ratings_df[target_user].isna()].index
        scores = {}
        for movie in unrated_movies:
            rating_sum = 0
            sim_sum = 0
            for sim_user in similar_users:
                rating = self.ratings_df.loc[movie, sim_user]
                if not np.isnan(rating):
                    rating_sum += rating * sim_scores[sim_user]
                    sim_sum += sim_scores[sim_user]
            scores[movie] = rating_sum / sim_sum if sim_sum > 0 else 0
        recommended = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [movie for movie, score in recommended]

    def recommend_content_based(self, movie_title, top_n=3):
        if movie_title not in self.genre_matrix.index:
            return []
        target_vec = self.genre_matrix.loc[movie_title].values.reshape(1, -1)
        similarities = cosine_similarity(target_vec, self.genre_matrix.values)[0]
        similar_idx = np.argsort(similarities)[::-1]
        recs = []
        for idx in similar_idx:
            title = self.genre_matrix.index[idx]
            if title != movie_title:
                recs.append(title)
            if len(recs) == top_n:
                break
        return recs

    def recommend_click(self):
        self.result_box.delete(1.0, tk.END)
        top_n = self.top_n_var.get()
        if self.filter_type.get() == "Collaborative":
            user = self.user_var.get()
            recs = self.recommend_collaborative(user, top_n=top_n)
            self.result_box.insert(tk.END, f"Top recommendations for {user}:\n")
            for i, movie in enumerate(recs, 1):
                self.result_box.insert(tk.END, f"{i}. {movie}\n")
        else:
            movie = self.movie_var.get()
            recs = self.recommend_content_based(movie, top_n=top_n)
            self.result_box.insert(tk.END, f"Movies similar to '{movie}':\n")
            for i, m in enumerate(recs, 1):
                genres = self.movies_df.loc[m, 'genre']
                self.result_box.insert(tk.END, f"{i}. {m} ({genres})\n")

    def user_user_matrix_click(self):
        ratings_filled = self.ratings_df.fillna(0).T
        user_sim_matrix = pd.DataFrame(
            cosine_similarity(ratings_filled),
            index=ratings_filled.index, columns=ratings_filled.index
        )
        plot_similarity_matrix(user_sim_matrix, "User-User Similarity Matrix", labels=user_sim_matrix.columns)

    def genre_matrix_click(self):
        genre_sim_matrix = pd.DataFrame(
            cosine_similarity(self.genre_matrix.values),
            index=self.genre_matrix.index, columns=self.genre_matrix.index
        )
        plot_similarity_matrix(genre_sim_matrix, "Movie-Genre Similarity Matrix", labels=genre_sim_matrix.columns)

    def user_rating_dist_click(self):
        plt.figure(figsize=(8, 5))
        self.ratings_df.T.plot(kind="bar", stacked=True, ax=plt.gca())
        plt.title("User Ratings Distribution")
        plt.xlabel("User")
        plt.ylabel("Ratings")
        plt.tight_layout()
        plt.show()

    def rate_new_movie(self):
        user = self.user_var.get()
        unrated_movies = self.ratings_df[self.ratings_df[user].isna()].index
        if len(unrated_movies) == 0:
            messagebox.showinfo("Rate Movie", "All movies have been rated by this user.")
            return
        # Dialog to choose a movie
        movie_dialog = tk.Toplevel(self)
        movie_dialog.title("Rate a New Movie")
        movie_dialog.configure(bg="#e3f3ff")

        tk.Label(movie_dialog, text=f"Choose a movie for {user} to rate:",
                 bg="#e3f3ff", font=("Arial", 12)).pack(pady=5)
        movie_var = tk.StringVar(value=unrated_movies[0])
        movie_menu = ttk.Combobox(movie_dialog, textvariable=movie_var,
                                  values=list(unrated_movies), state="readonly", width=20)
        movie_menu.pack(pady=6)

        tk.Label(movie_dialog, text="Enter rating (1-5):", bg="#e3f3ff", font=("Arial", 12)).pack(pady=5)
        rating_var = tk.StringVar()
        rating_entry = tk.Entry(movie_dialog, textvariable=rating_var, width=5)
        rating_entry.pack(pady=6)

        def submit_rating():
            movie = movie_var.get()
            try:
                rating = float(rating_var.get())
                if rating < 1 or rating > 5:
                    messagebox.showerror("Invalid Rating", "Rating must be between 1 and 5.")
                    return
                self.ratings_df.loc[movie, user] = rating
                messagebox.showinfo("Success", f"{user} rated '{movie}' with {rating}.")
                movie_dialog.destroy()
            except ValueError:
                messagebox.showerror("Invalid Rating", "Please enter a number between 1 and 5.")

        submit_btn = tk.Button(movie_dialog, text="Submit Rating", bg="#79b4f7", fg="#0a3e62",
                               font=("Arial", 12), command=submit_rating)
        submit_btn.pack(pady=10)

def plot_similarity_matrix(matrix, title="Similarity Matrix", labels=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def run_gui(movies_df, ratings_df, genre_matrix):
    app = MovieRecommenderGUI(movies_df, ratings_df, genre_matrix)
    app.mainloop()