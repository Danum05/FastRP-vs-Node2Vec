import re
import logging
from py2neo import Graph, Node, Relationship
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download stopwords for NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Connect to Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "211524037"))

# Function to load data from JSON file
def load_data_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Function to remove HTML tags
def remove_html_tags(text):
    return BeautifulSoup(text, "html.parser").get_text()

# Function for case folding
def case_folding(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = text.replace("-", " ")       # Replace hyphens with spaces
    text = re.sub(r"\d+", "", text)     # Remove digits
    return text

# Function to remove stopwords
def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in stop_words])

# Function for text preprocessing
def preprocess_text(text):
    text = remove_html_tags(text)
    text = case_folding(text)
    text = remove_stopwords(text)
    return text

# Function to create a DataFrame from data
def create_dataframe(data):
    df = pd.DataFrame.from_dict(data)
    return df[["title", "genre", "overview", "actors"]]

# Function to save data to Neo4j
def save_to_neo4j(df):
    for idx, row in df.iterrows():
        # Preprocess overview text
        cleaned_overview = preprocess_text(row["overview"])
        
        # Create Movie node
        movie_node = Node(
            "Movie",
            id=int(row["id"]),
            title=row["title"],
            type=row["type"]
        )
        graph.merge(movie_node, "Movie", "id")
        
        # Create Overview node
        overview_node = Node("Overview", text=cleaned_overview)
        graph.merge(overview_node, "Overview", "text")
        
        # Create relationship HAS_OVERVIEW
        graph.merge(Relationship(movie_node, "HAS_OVERVIEW", overview_node))
        
        # Create Genre nodes and relationships
        for genre in row["genre"]:
            genre_node = Node("Genre", name=genre)
            graph.merge(genre_node, "Genre", "name")
            graph.merge(Relationship(movie_node, "HAS_GENRE", genre_node))
        
        # Create Actor nodes and relationships
        for actor in row["actors"]:
            actor_node = Node("Actor", name=actor)
            graph.merge(actor_node, "Actor", "name")
            graph.merge(Relationship(movie_node, "FEATURES", actor_node))

def main():
    file_path_movies = 'movie.json'
    file_path_history = 'history.json'

    data_movies = load_data_from_json(file_path_movies)
    data_history = load_data_from_json(file_path_history)

    df_movies = create_dataframe(data_movies)
    df_history = create_dataframe(data_history)

    df_movies["type"] = "movie"
    df_history["type"] = "history"

    df_movies["id"] = df_movies.index + 1
    df_history["id"] = df_history.index + len(df_movies) + 1

    df_combined = pd.concat([df_movies, df_history]).reset_index(drop=True)

    if df_combined["id"].isnull().any():
        logger.error("The 'id' column contains null values")
        return

    save_to_neo4j(df_combined)
    
    df_combined.to_json('processed_movie_data.json', orient='records', force_ascii=False)

if __name__ == "__main__":
    main()
