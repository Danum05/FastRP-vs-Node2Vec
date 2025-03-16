import re
import logging
from py2neo import Graph, Node, Relationship
import pandas as pd
from bs4 import BeautifulSoup
import nltk
import json

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Download stopwords NLTK (hanya jika belum ada)
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

# Koneksi ke Neo4j
try:
    logger.info("Menghubungkan ke database Neo4j...")
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "211524037"))
    logger.info("Koneksi ke Neo4j berhasil.")
except Exception as e:
    logger.error(f"Gagal terhubung ke Neo4j: {e}")
    exit(1)  # Keluar jika koneksi gagal

# Fungsi untuk memuat data dari JSON
def load_data_from_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except Exception as e:
        logger.error(f"Error membaca file JSON {file_path}: {e}")
        return []

# Fungsi untuk membersihkan teks dari HTML
def remove_html_tags(text):
    return BeautifulSoup(text, "html.parser").get_text()

# Fungsi untuk normalisasi teks (case folding, hapus angka, dan tanda baca)
def case_folding(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Hapus tanda baca
    text = text.replace("-", " ")       # Ganti tanda hubung dengan spasi
    text = re.sub(r"\d+", "", text)     # Hapus angka
    return text

# Fungsi untuk menghapus stopwords
def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in stop_words])

# Fungsi preprocessing teks
def preprocess_text(text):
    text = remove_html_tags(text)
    text = case_folding(text)
    text = remove_stopwords(text)
    return text

# Fungsi untuk membuat DataFrame dari JSON
def create_dataframe(data):
    df = pd.DataFrame.from_dict(data)
    if not {"title", "genre", "overview", "actors"}.issubset(df.columns):
        logger.error("Kolom dalam JSON tidak sesuai format yang diharapkan.")
        return pd.DataFrame()
    return df[["title", "genre", "overview", "actors"]]

# Fungsi untuk menyimpan data ke Neo4j dengan bobot pada relasi
def save_to_neo4j(df):
    for idx, row in df.iterrows():
        try:
            # Preprocessing teks sinopsis
            cleaned_overview = preprocess_text(row["overview"])

            # Membuat node Movie
            movie_node = Node(
                "Movie",
                id=int(row["id"]),
                title=row["title"],
                type=row["type"]
            )
            graph.merge(movie_node, "Movie", "id")

            # Membuat node Overview
            overview_node = Node("Overview", text=cleaned_overview)
            graph.merge(overview_node, "Overview", "text")

            # Membuat relasi HAS_OVERVIEW dengan bobot 0.3
            graph.merge(Relationship(movie_node, "HAS_OVERVIEW", overview_node, weight=0.3))

            # Membuat node Genre dan relasi HAS_GENRE dengan bobot 0.5
            for genre in row["genre"]:
                genre_node = Node("Genre", name=genre)
                graph.merge(genre_node, "Genre", "name")
                graph.merge(Relationship(movie_node, "HAS_GENRE", genre_node, weight=0.5))

            # Membuat node Actor dan relasi FEATURES dengan bobot 0.2
            for actor in row["actors"]:
                actor_node = Node("Actor", name=actor)
                graph.merge(actor_node, "Actor", "name")
                graph.merge(Relationship(movie_node, "FEATURES", actor_node, weight=0.2))

        except Exception as e:
            logger.error(f"Error saat menyimpan data ke Neo4j untuk film {row['title']}: {e}")

def main():
    file_path_movies = 'movie.json'
    file_path_history = 'history.json'

    # Load data dari JSON
    data_movies = load_data_from_json(file_path_movies)
    data_history = load_data_from_json(file_path_history)

    # Buat DataFrame
    df_movies = create_dataframe(data_movies)
    df_history = create_dataframe(data_history)

    if df_movies.empty or df_history.empty:
        logger.error("Data tidak valid, program dihentikan.")
        return

    # Tambahkan kolom type untuk membedakan Movie dan History
    df_movies["type"] = "movie"
    df_history["type"] = "history"

    # Menentukan ID unik untuk setiap film
    df_movies["id"] = df_movies.index + 1
    df_history["id"] = df_history.index + len(df_movies) + 1

    df_combined = pd.concat([df_movies, df_history]).reset_index(drop=True)

    if df_combined["id"].isnull().any():
        logger.error("Kolom 'id' mengandung nilai kosong, program dihentikan.")
        return

    # Simpan data ke Neo4j
    save_to_neo4j(df_combined)

    # Simpan hasil preprocessing ke file JSON
    df_combined.to_json('processed_movie_data.json', orient='records', force_ascii=False)

    logger.info("Data berhasil diproses dan disimpan ke Neo4j serta file JSON.")

if __name__ == "__main__":
    main()
