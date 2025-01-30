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

# Function to create TF-IDF embedding
def create_tfidf_embedding(data, type_name):
    
    vectorizer = TfidfVectorizer(
        max_features=10,
        stop_words='english',
        ngram_range=(1, 1)
    )
    
    # Fit and transform data
    tfidf_matrix = vectorizer.fit_transform(data["processed_overview"])
    feature_names = vectorizer.get_feature_names_out()
    
    # Create DataFrame for visualization
    df_tfidf = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=feature_names,
        index=data["id"]
    )
    
    # Save to CSV for analysis
    output_file = f'TF-IDF_{type_name}.csv'
    df_tfidf.to_csv(output_file, index=True)
    
    return tfidf_matrix, feature_names, vectorizer

# Function to save data to Neo4j
def save_to_neo4j(df, tfidf_matrix, feature_names, movie_type):
    
    for index, row in df.iterrows():
        # Get index position in tfidf_matrix
        matrix_index = df.index.get_loc(row.name)
        tfidf_row = tfidf_matrix[matrix_index].toarray()[0]
        
        # Create node with appropriate properties
        movie_node = Node(
            "Movie",
            id=row["id"],
            title=row["title"],
            overview=row["overview"],
            processed_overview=row["processed_overview"],
            embedding=tfidf_row.tolist(),
            type=movie_type
        )
        
        # Merge node to database
        graph.merge(movie_node, "Movie", "id")
        
        # Process genres
        for genre in row["genre"]:
            genre_node = Node("Genre", name=genre)
            graph.merge(genre_node, "Genre", "name")
            rel = Relationship(movie_node, "HAS_GENRE", genre_node)
            graph.merge(rel)
        
        # Process actors
        for actor in row["actors"]:
            actor_node = Node("Actor", name=actor)
            graph.merge(actor_node, "Actor", "name")
            rel = Relationship(movie_node, "FEATURES", actor_node)
            graph.merge(rel)
        
        # Process TF-IDF features
        for i, feature in enumerate(feature_names):
            weight = float(tfidf_row[i])
            if weight > 0:
                tag_node = Node("Tag", name=feature)
                graph.merge(tag_node, "Tag", "name")
                
                rel = Relationship(
                    movie_node, 
                    "TAGGED", 
                    tag_node, 
                    weight=weight,
                    movie_type=movie_type
                )
                graph.merge(rel)

def main():
    # Load data
    file_path_movies = 'movie.json'
    file_path_history = 'history.json'

    data_movies = load_data_from_json(file_path_movies)
    data_history = load_data_from_json(file_path_history)

    # Create DataFrames
    df_movies = create_dataframe(data_movies)
    df_history = create_dataframe(data_history)

    # Add 'type' column
    df_movies["type"] = "movie"
    df_history["type"] = "history"

    # Create unique IDs
    df_movies["id"] = df_movies.index + 1
    df_history["id"] = df_history.index + len(df_movies) + 1

    # Combine DataFrames
    df_combined = pd.concat([df_movies, df_history])

    # Check for null IDs
    if df_combined["id"].isnull().any():
        logger.error("The 'id' column contains null values")
        return

    # Preprocess overview text
    df_combined["processed_overview"] = df_combined["overview"].apply(preprocess_text)

    # Split data by type
    movies_data = df_combined[df_combined['type'] == 'movie']
    history_data = df_combined[df_combined['type'] == 'history']

    # Process movies and history separately
    movie_tfidf, movie_features, movie_vectorizer = create_tfidf_embedding(movies_data, "movies")
    history_tfidf, history_features, history_vectorizer = create_tfidf_embedding(history_data, "history")

    # Convert TF-IDF matrix to DataFrame for visualization
    movie_tfidf_df = pd.DataFrame(movie_tfidf.toarray(), columns=movie_features, index=movies_data["id"])
    history_tfidf_df = pd.DataFrame(history_tfidf.toarray(), columns=history_features, index=history_data["id"])

    # Display TF-IDF DataFrame in log
    logger.info(f"TF-IDF Data Movie:\n{movie_tfidf_df}")
    logger.info(f"TF-IDF Data History:\n{history_tfidf_df}")

    # Save to Neo4j
    save_to_neo4j(movies_data, movie_tfidf, movie_features, "movie")
    save_to_neo4j(history_data, history_tfidf, history_features, "history")

    # Save combined DataFrame to JSON
    output_json = 'processed_movie_data.json'
    df_combined.to_json(output_json, orient='records', force_ascii=False)

if __name__ == "__main__":
    main()