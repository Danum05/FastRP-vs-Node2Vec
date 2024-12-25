import logging
from py2neo import Graph
import numpy as np
import pandas as pd

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Menghubungkan ke database Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "211524037"))

# Fungsi untuk menjalankan proses FastRP
def run_fastrp_process(graph):
    # Mengecek apakah graf dengan nama 'movieGraph' sudah ada
    logger.info("Mengecek apakah graf 'movieGraph' sudah ada...")
    graph_exists = graph.run("""CALL gds.graph.exists('movieGraph') YIELD exists RETURN exists""").data()[0]['exists']
    
    if graph_exists:
        # Menghapus graf yang ada jika sudah ada
        logger.info("Menghapus graf 'movieGraph' yang sudah ada...")
        graph.run("""CALL gds.graph.drop('movieGraph', false) YIELD graphName""")
    
    # Membuat graf baru tanpa menyebutkan 'embedding' secara eksplisit
    logger.info("Membuat graf baru tanpa 'embedding'...")
    graph.run("""
        CALL gds.graph.project(
            'movieGraph',
            ['Movie', 'Tag'],  
            {
                TAGGED: {
                    orientation: 'UNDIRECTED'
                }
            }
    )
    """)

    # Menjalankan FastRP untuk menghasilkan embeddings
    logger.info("Menjalankan FastRP...")
    result = graph.run("""
        CALL gds.fastRP.stream('movieGraph', {
            embeddingDimension: 64,          
            iterationWeights: [0.0, 1.0, 10.0],            
            propertyRatio: 0.0,  
            normalizationStrength: 0.8,  
            randomSeed: 42                
        })
        YIELD nodeId, embedding
        WITH nodeId, embedding
        MATCH (n:Movie) WHERE id(n) = nodeId
        SET n.embedding = embedding
        RETURN n.id AS id, n.embedding AS embedding
    """).data()

    # Periksa hasil dan normalisasi embedding
    embeddings_list = []
    for record in result:
        embedding = np.array(record['embedding'])

        # Mengecek apakah embedding mengandung NaN
        if np.isnan(embedding).any():
            logger.warning(f"Embedding kosong ditemukan untuk node ID {record['id']}.")
            continue  # Skip jika embedding invalid

        # Normalisasi embedding
        normalized_embedding = embedding / np.linalg.norm(embedding)
        
        # Menyimpan hasil embedding yang sudah dinormalisasi
        graph.run("MATCH (n:Movie {id: $id}) SET n.embedding = $embedding", id=record['id'], embedding=normalized_embedding.tolist())
        logger.info(f"Node ID: {record['id']}, Normalized Embedding: {normalized_embedding.tolist()}")

        # Menambahkan hasil embedding ke list untuk disimpan di CSV
        embeddings_list.append({'id': record['id'], 'embedding': normalized_embedding.tolist()})

    # Menyimpan embeddings ke file CSV
    if embeddings_list:
        df_embeddings = pd.DataFrame(embeddings_list)
        df_embeddings.to_csv("FastRP.csv", index=False)
        logger.info("Embeddings disimpan ke file 'FastRP.csv'.")
    else:
        logger.warning("Tidak ada embedding yang valid untuk disimpan.")

# Menjalankan proses FastRP
run_fastrp_process(graph)
