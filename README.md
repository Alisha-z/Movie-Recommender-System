# Movie-Recommender-System

A beginner-to-intermediate Python project demonstrating both collaborative and content-based filtering for movie recommendations. The project features a Tkinter GUI and a CLI-based interface for greater flexibility and learning.

---

## 📁 Project Structure

This repository contains three main files:

### 1. `movie_recommender.py`
- **Purpose:** Acts as the connector and controller for the GUI application.
- **Functionality:**  
  - Contains core logic for recommendation algorithms and data setup.
  - Imports and launches the GUI from `Movie_recommender_GUI.py`.
  - Ensures the GUI is supplied with the necessary movie and user rating data.
- **Usage:**  
  Run this file to launch the graphical interface for the recommender system.

### 2. `Movie_recommender_GUI.py`
- **Purpose:** Contains all code related to the graphical user interface (GUI).
- **Functionality:**  
  - Implements the Tkinter GUI for movie recommendations.
  - Allows users to choose between collaborative and content-based filtering.
  - Features include:
    - Displaying movie recommendations for users
    - Showing user-user similarity matrices
    - Visualizing movie-genre similarities
    - Displaying user ratings distribution
    - Rating new movies interactively
  - GUI elements are arranged for clarity and usability, with vibrant colors and grouped buttons.
- **Usage:**  
  Do **not** run directly. This file is imported and launched via `movie_recommender.py`.

### 3. `main.py`
- **Purpose:** Provides a fully-featured CLI (command-line interface) version of the recommender system.
- **Functionality:**  
  - Contains all core recommendation logic and data setup.
  - Presents an interactive, menu-driven CLI experience.
  - All features found in the GUI (recommendations, similarity matrices, rating, visualizations) are available through the CLI.
  - Does **not** include any graphical interface.
- **Usage:**  
  Run this file if you want to use the recommender system in your terminal.

---

## 🚀 Getting Started

1. **Requirements:**  
   - Python 3.x  
   - `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`  
   Install with:  
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

2. **Run the GUI version:**  
   ```bash
   python movie_recommender.py
   ```

3. **Run the CLI version:**  
   ```bash
   python main.py
   ```

---

## 📝 Features

- **Collaborative Filtering:**  
  Recommends movies to a user based on similar users' ratings.
- **Content-Based Filtering:**  
  Recommends movies similar to those you like, based on genres.
- **Similarity Visualizations:**  
  - User-user similarity matrix
  - Movie-genre similarity matrix
  - User ratings distribution
- **Rate New Movies:**  
  Interactively rate unrated movies for users (GUI and CLI).

---

## 📚 How the Files Work Together

- **`movie_recommender.py`** sets up data and calls the GUI code in **`Movie_recommender_GUI.py`**.
- **`Movie_recommender_GUI.py`** displays the interactive window, handles user input, and shows all the project features visually.
- **`main.py`** is a standalone script for users who prefer command-line interfaces, with all recommendation and visualization features (but no GUI).

---

## 🏗️ Extending the Project

- Add more users or movies by editing the data setup in the files.
- Connect to external datasets (e.g., MovieLens CSVs) for larger use cases.
- Enhance GUI with more filters or visual styles.
- Integrate model-based collaborative filtering for advanced recommendations.

---

## 📄 License

Feel free to use and extend this project for learning, research, or personal use!
