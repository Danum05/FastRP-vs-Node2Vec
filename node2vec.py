import logging
from py2neo import Graph
import numpy as np 
import pandas as pd

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Menghubungkan ke database Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "211524037"))

# Fungsi untuk menjalankan proses Node2Vec
def run_node2vec_process(graph):
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
            ['Movie', 'Tag', 'Genre', 'Actor'],  
            {
                TAGGED: {
                    orientation: 'UNDIRECTED'
                }
            }
    )
    """)

    # Menjalankan Node2Vec untuk menghasilkan embeddings
    logger.info("Menjalankan Node2Vec...")
    result = graph.run("""
        CALL gds.beta.node2vec.stream('movieGraph', {
            embeddingDimension: 256,       
            walkLength: 80,               
            iterations: 10,                    
            inOutFactor: 1.0,                      
            returnFactor: 2.0,                      
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
        df_embeddings.to_csv("Node2Vec.csv", index=False)
        logger.info("Embeddings disimpan ke file 'Node2Vec.csv'.")
    else:
        logger.warning("Tidak ada embedding yang valid untuk disimpan.")

# Menjalankan proses Node2Vec
run_node2vec_process(graph)

# import logging
# from py2neo import Graph
# import numpy as np
# import pandas as pd
# from datetime import datetime

# # Konfigurasi logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Menghubungkan ke database Neo4j
# graph = Graph("bolt://localhost:7687", auth=("neo4j", "211524037"))

# # Fungsi untuk menjalankan proses Node2Vec dengan parameter yang berbeda
# def run_node2vec_process(graph, embedding_dim, walk_length, iterations, inout_factor, return_factor, random_seed):
#     # Mengecek apakah graf dengan nama 'movieGraph' sudah ada
#     logger.info("Mengecek apakah graf 'movieGraph' sudah ada...")
#     graph_exists = graph.run("""CALL gds.graph.exists('movieGraph') YIELD exists RETURN exists""").data()[0]['exists']

#     if graph_exists:
#         # Menghapus graf yang ada jika sudah ada
#         logger.info("Menghapus graf 'movieGraph' yang sudah ada...")
#         graph.run("""CALL gds.graph.drop('movieGraph', false) YIELD graphName""")

#     # Membuat graf baru tanpa menyebutkan 'embedding' secara eksplisit
#     logger.info("Membuat graf baru tanpa 'embedding'...")
#     graph.run("""
#         CALL gds.graph.project(
#             'movieGraph',
#             ['Movie', 'Tag'],  
#             {
#                 TAGGED: {
#                     orientation: 'UNDIRECTED'
#                 }
#             }
#         )
#     """)

#     # Menjalankan Node2Vec untuk menghasilkan embeddings
#     logger.info(f"Menjalankan Node2Vec dengan dimensi {embedding_dim}, walk length {walk_length}...")
#     params = {
#         "dim": embedding_dim,
#         "length": walk_length,
#         "iters": iterations,
#         "inout": inout_factor,
#         "return": return_factor,  # Menggunakan 'return' untuk Neo4j
#         "seed": random_seed
#     }
    
#     result = graph.run("""
#         CALL gds.beta.node2vec.stream('movieGraph', {
#             embeddingDimension: $dim,
#             walkLength: $length,
#             iterations: $iters,
#             inOutFactor: $inout,
#             returnFactor: $return,
#             randomSeed: $seed
#         })
#         YIELD nodeId, embedding
#         WITH nodeId, embedding
#         MATCH (n:Movie) WHERE id(n) = nodeId
#         SET n.embedding = embedding
#         RETURN n.id AS id, n.embedding AS embedding
#     """, params).data()

#     # Periksa hasil dan normalisasi embedding
#     embeddings_list = []
#     for record in result:
#         embedding = np.array(record['embedding'])

#         # Mengecek apakah embedding mengandung NaN
#         if np.isnan(embedding).any():
#             logger.warning(f"Embedding kosong ditemukan untuk node ID {record['id']}.")
#             continue  # Skip jika embedding invalid

#         # Normalisasi embedding
#         normalized_embedding = embedding / np.linalg.norm(embedding)

#         # Menyimpan hasil embedding yang sudah dinormalisasi
#         graph.run("MATCH (n:Movie {id: $id}) SET n.embedding = $embedding", 
#                  id=record['id'], embedding=normalized_embedding.tolist())
#         logger.info(f"Node ID: {record['id']}, Normalized Embedding: {normalized_embedding.tolist()}")

#         # Menambahkan hasil embedding ke list untuk disimpan di CSV
#         embeddings_list.append({'id': record['id'], 'embedding': normalized_embedding.tolist()})

#     # Menyimpan embeddings ke file CSV dengan timestamp dan parameter
#     if embeddings_list:
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#         filename = f"Node2Vec_dim{embedding_dim}_walk{walk_length}_{timestamp}.csv"
#         df_embeddings = pd.DataFrame(embeddings_list)
#         df_embeddings.to_csv(filename, index=False)
#         logger.info(f"Embeddings disimpan ke file '{filename}'.")
#     else:
#         logger.warning("Tidak ada embedding yang valid untuk disimpan.")

# # Parameter eksperimen
# experiments = [
#     {
#         'embedding_dim': 128,
#         'walk_length': 40,
#         'iterations': 10,
#         'inout_factor': 1.0,
#         'return_factor': 1.0,
#         'random_seed': 42
#     },
#     {
#         'embedding_dim': 256,
#         'walk_length': 80,
#         'iterations': 10,
#         'inout_factor': 1.0,
#         'return_factor': 2.0,
#         'random_seed': 42
#     },
#     {
#         'embedding_dim': 512,
#         'walk_length': 100,
#         'iterations': 15,
#         'inout_factor': 2.0,
#         'return_factor': 2.0,
#         'random_seed': 42
#     }
# ]

# # Menjalankan eksperimen dengan berbagai parameter
# for i, params in enumerate(experiments, 1):
#     logger.info(f"\nMenjalankan eksperimen {i}/{len(experiments)}")
#     logger.info(f"Parameter: {params}")
#     run_node2vec_process(
#         graph=graph,
#         embedding_dim=params['embedding_dim'],
#         walk_length=params['walk_length'],
#         iterations=params['iterations'],
#         inout_factor=params['inout_factor'],
#         return_factor=params['return_factor'],
#         random_seed=params['random_seed']
#     )