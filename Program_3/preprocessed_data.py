import re
import logging
from py2neo import Graph, Node, Relationship
import pandas as pd
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
import nltk
import json
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download stopwords for NLTK
nltk.download('stopwords')

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

# Function to remove stopwords using NLTK
stop_words = set(stopwords.words('english'))

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

# Function to create sentence embeddings
def create_sentence_embeddings(data):
    # Initialize the model (you can choose different models)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings
    embeddings = model.encode(data["processed_overview"].tolist(), 
                            show_progress_bar=True, 
                            batch_size=32)
    
    # Save embeddings to CSV
    df_embeddings = pd.DataFrame(
        embeddings,
        index=data.index
    )
    df_embeddings.to_csv('sentence_embeddings.csv', index=True)
    
    return embeddings

# Function to save data to Neo4j
def save_to_neo4j(df, embeddings):
    for idx, row in df.iterrows():
        embedding_vector = embeddings[idx]
        
        movie_node = Node(
            "Movie",
            id=int(row["id"]),
            title=row["title"],
            overview=row["overview"],
            processed_overview=row["processed_overview"],
            embedding=json.dumps(embedding_vector.tolist()),
            type=row["type"]
        )
        graph.merge(movie_node, "Movie", "id")
        
        for genre in row["genre"]:
            genre_node = Node("Genre", name=genre)
            graph.merge(genre_node, "Genre", "name")
            graph.merge(Relationship(movie_node, "HAS_GENRE", genre_node))
        
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

    df_combined["processed_overview"] = df_combined["overview"].apply(preprocess_text)

    embeddings = create_sentence_embeddings(df_combined)

    logger.info(f"Generated embeddings shape: {embeddings.shape}")

    save_to_neo4j(df_combined, embeddings)
    
    df_combined.to_json('processed_movie_data.json', orient='records', force_ascii=False)

if __name__ == "__main__":
    main()