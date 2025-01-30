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
            ['Movie', 'Tag', 'Actor', 'Genre'],  
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
            embeddingDimension: 256,          
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
        df_embeddings.to_csv("FastRp.csv", index=False)
        logger.info("Embeddings disimpan ke file 'fastRp.csv'.")
    else:
        logger.warning("Tidak ada embedding yang valid untuk disimpan.")

# Menjalankan proses FastRP
run_fastrp_process(graph)

# import logging
# from py2neo import Graph
# import numpy as np
# import pandas as pd
# from datetime import datetime

# # Konfigurasi logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(_name_)

# # Menghubungkan ke database Neo4j
# graph = Graph("bolt://localhost:7687", auth=("neo4j", "211524037"))

# # Fungsi untuk menjalankan proses FastRP dengan parameter yang berbeda
# def run_fastrp_process(graph, embedding_dim, iteration_weights, prop_ratio, norm_strength, random_seed, feature_properties=None):
#     # Mengecek apakah graf dengan nama 'movieGraph' sudah ada
#     logger.info("Mengecek apakah graf 'movieGraph' sudah ada...")
#     graph_exists = graph.run("""CALL gds.graph.exists('movieGraph') YIELD exists RETURN exists""").data()[0]['exists']
    
#     if graph_exists:
#         # Menghapus graf yang ada jika sudah ada
#         logger.info("Menghapus graf 'movieGraph' yang sudah ada...")
#         graph.run("""CALL gds.graph.drop('movieGraph', false) YIELD graphName""")
    
#     # Membuat graf baru dengan property yang akan digunakan
#     logger.info("Membuat graf baru...")
#     create_graph_query = """
#         CALL gds.graph.project(
#             'movieGraph',
#             ['Movie', 'Tag'],
#             {
#                 TAGGED: {
#                     orientation: 'UNDIRECTED'
#                 }
#             }
#             %s
#         )
#     """
    
#     # Menambahkan nodeProperties jika ada feature_properties
#     node_properties = ""
#     if feature_properties:
#         node_properties = f", {{ nodeProperties: {feature_properties} }}"
    
#     graph.run(create_graph_query % node_properties)
    
#     # Menyiapkan parameter FastRP
#     fastrp_params = {
#         'embeddingDimension': embedding_dim,
#         'iterationWeights': iteration_weights,
#         'normalizationStrength': norm_strength,
#         'randomSeed': random_seed
#     }
    
#     # Hanya menambahkan propertyRatio dan featureProperties jika propertyRatio > 0
#     if prop_ratio > 0 and feature_properties:
#         fastrp_params['propertyRatio'] = prop_ratio
#         fastrp_params['featureProperties'] = feature_properties
    
#     # Menjalankan FastRP untuk menghasilkan embeddings
#     logger.info(f"Menjalankan FastRP dengan parameter: {fastrp_params}")
#     result = graph.run("""
#         CALL gds.fastRP.stream('movieGraph', $params)
#         YIELD nodeId, embedding
#         WITH nodeId, embedding
#         MATCH (n:Movie) WHERE id(n) = nodeId
#         SET n.embedding = embedding
#         RETURN n.id AS id, n.embedding AS embedding
#     """, params=fastrp_params).data()
    
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
    
#     # Menyimpan embeddings ke file CSV dengan timestamp
#     if embeddings_list:
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#         filename = f"FastRP_dim{embedding_dim}norm{norm_strength}{timestamp}.csv"
#         df_embeddings = pd.DataFrame(embeddings_list)
#         df_embeddings.to_csv(filename, index=False)
#         logger.info(f"Embeddings disimpan ke file '{filename}'.")
#     else:
#         logger.warning("Tidak ada embedding yang valid untuk disimpan.")

# # Parameter eksperimen
# experiments = [
#     {
#         'embedding_dim': 128,
#         'iteration_weights': [0.0, 1.0, 5.0],
#         'prop_ratio': 0.0,  # Tidak memerlukan feature_properties
#         'norm_strength': 0.5,
#         'random_seed': 42,
#         'feature_properties': None
#     },
#     {
#         'embedding_dim': 256,
#         'iteration_weights': [0.0, 1.0, 10.0],
#         'prop_ratio': 0.0,  # Tidak memerlukan feature_properties
#         'norm_strength': 0.8,
#         'random_seed': 42,
#         'feature_properties': None
#     },
#     {
#         'embedding_dim': 512,
#         'iteration_weights': [0.0, 2.0, 8.0],
#         'prop_ratio': 0.3,
#         'norm_strength': 0.8,
#         'random_seed': 42,
#         'feature_properties': None
#     }
# ]

# # Menjalankan eksperimen dengan berbagai parameter
# for i, params in enumerate(experiments, 1):
#     logger.info(f"\nMenjalankan eksperimen {i}/{len(experiments)}")
#     logger.info(f"Parameter: {params}")
#     run_fastrp_process(
#         graph,
#         params['embedding_dim'],
#         params['iteration_weights'],
#         params['prop_ratio'],
#         params['norm_strength'],
#         params['random_seed'],
#         params['feature_properties']
#     )