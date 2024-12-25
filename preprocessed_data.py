import re
import logging
from py2neo import Graph, Node, Relationship
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
import json

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

# Load data
file_path_movies = 'movie.json'  # Update with your movie JSON file path
file_path_history = 'history.json'  # Update with your history JSON file path

data_movies = load_data_from_json(file_path_movies)
data_history = load_data_from_json(file_path_history)

# Create DataFrames
df_movies = create_dataframe(data_movies)
df_history = create_dataframe(data_history)

# Add 'type' column to distinguish between movies and history
df_movies["type"] = "movie"
df_history["type"] = "history"

# Ensure IDs are unique (use index or create custom IDs)
df_movies["id"] = df_movies.index + 1  # Start ID from 1
df_history["id"] = df_history.index + len(df_movies) + 1  # Continue IDs after movies

# Combine DataFrames
df_combined = pd.concat([df_movies, df_history])

# Ensure no null values in 'id' column
if df_combined["id"].isnull().any():
    logger.error("The 'id' column contains null values. Ensure it is properly filled.")
else:
    logger.info("'id' column is properly filled.")

# Preprocess 'overview' column
df_combined["processed_overview"] = df_combined["overview"].apply(preprocess_text)

# Debugging preprocessed text
for index, text in enumerate(df_combined["overview"][:5]):  # Check the first 5 entries
    logger.info(f"Original overview {index}: {text}")
    logger.info(f"Processed overview {index}: {preprocess_text(text)}")

# TF-IDF Vectorization on processed text
tfidf_vectorizer = TfidfVectorizer(max_features=10)
tfidf_matrix = tfidf_vectorizer.fit_transform(df_combined["processed_overview"])
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

# Convert TF-IDF matrix to DataFrame for visualization
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_feature_names, index=df_combined["id"])

# Display TF-IDF DataFrame
logger.info(f"TF-IDF Data:\n{df_tfidf}")

# Function to save data to Neo4j
def save_to_neo4j(df, tfidf_matrix, tfidf_feature_names):
    for index, row in df.iterrows():
        tfidf_row = tfidf_matrix[index].toarray()[0].tolist()

        # Create a movie node with embedding field and type
        movie_node = Node("Movie", id=row["id"], title=row["title"], overview=row["overview"],
                          processed_overview=row["processed_overview"], embedding=tfidf_row, type=row["type"])
        graph.merge(movie_node, "Movie", "id")

        # Iterate over genres
        for genre in row["genre"]:
            genre_node = Node("Genre", name=genre)
            graph.merge(genre_node, "Genre", "name")
            genre_rel = Relationship(movie_node, "HAS_GENRE", genre_node)
            graph.merge(genre_rel)

        # Iterate over actors
        for actor in row["actors"]:
            actor_node = Node("Actor", name=actor)
            graph.merge(actor_node, "Actor", "name")
            actor_rel = Relationship(movie_node, "FEATURES", actor_node)
            graph.merge(actor_rel)

        # Iterate over TF-IDF features
        for i, feature in enumerate(tfidf_feature_names):
            weight = float(tfidf_row[i])
            if weight > 0:
                tag_node = Node("Tag", name=feature)
                graph.merge(tag_node, "Tag", "name")
                
                # Create bidirectional TAGGED relationships
                tagged_rel_forward = Relationship(movie_node, "TAGGED", tag_node, weight=weight)
                graph.merge(tagged_rel_forward)
                
                tagged_rel_backward = Relationship(tag_node, "TAGGED", movie_node, weight=weight)
                graph.merge(tagged_rel_backward)

# Save data to Neo4j with TF-IDF weights
save_to_neo4j(df_combined, tfidf_matrix, tfidf_feature_names)

# Save combined DataFrame to JSON file
output_file_path = 'processed_movie_data.json'  # Update with your desired path
df_combined.to_json(output_file_path, orient='records', force_ascii=False)

# Save TF-IDF DataFrame to CSV file
output_file_path = 'TF-IDF_movie.csv' 
df_tfidf.to_csv(output_file_path, index=True, encoding='utf-8')
