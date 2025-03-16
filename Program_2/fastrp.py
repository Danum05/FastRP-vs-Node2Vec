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
    
    # Membuat graf baru dengan semua node (Movie, Genre, Actor, Overview)
    logger.info("Membuat graf baru...")
    graph.run("""
        CALL gds.graph.project(
            'movieGraph',
            ['Movie', 'Genre', 'Actor', 'Overview'],
            {
                HAS_GENRE: { orientation: 'UNDIRECTED' },
                FEATURES: { orientation: 'UNDIRECTED' },
                HAS_OVERVIEW: { orientation: 'UNDIRECTED' }
            }
        )
    """)

    # Menjalankan FastRP
    logger.info("Menjalankan FastRP...")
    result = graph.run("""
        CALL gds.fastRP.stream('movieGraph', {
            embeddingDimension: 256,
            iterationWeights: [0.0, 1.0, 10.0],
            propertyRatio: 0.0,
            normalizationStrength: 0.8,
            randomSeed: 1
        })
        YIELD nodeId, embedding
        WITH nodeId, embedding
        MATCH (n) WHERE id(n) = nodeId
        RETURN n.id AS id, embedding, labels(n)[0] as label
    """).data()

    # Periksa hasil dan simpan embedding ke Neo4j
    embeddings_list = []
    for record in result:
        if record['embedding'] is None:
            logger.warning(f"Embedding kosong ditemukan untuk node ID {record['id']}.")
            continue

        embedding = np.array(record['embedding'])
        
        # Cek apakah embedding valid
        if np.all(embedding == 0):
            logger.warning(f"Zero embedding ditemukan untuk node ID {record['id']}.")
            continue

        try:
            # Normalisasi embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                normalized_embedding = embedding / norm
            else:
                logger.warning(f"Norm adalah 0 untuk node ID {record['id']}.")
                continue

            # Menentukan label node dan menyimpan embedding ke dalam Neo4j
            label = record['label']
            graph.run(f"""
                MATCH (n:{label} {{id: $id}}) 
                SET n.fastrp_embedding = $embedding
            """, id=record['id'], embedding=normalized_embedding.tolist())

            embeddings_list.append({
                'id': record['id'],
                'embedding': normalized_embedding.tolist(),
                'type': label
            })

            logger.info(f"Berhasil menyimpan embedding untuk {label} ID: {record['id']}")

        except Exception as e:
            logger.error(f"Error saat memproses node ID {record['id']}: {str(e)}")
            continue

    # Menyimpan embeddings ke file CSV
    if embeddings_list:
        df_embeddings = pd.DataFrame(embeddings_list)
        df_embeddings.to_csv("FastRP_embeddings.csv", index=False)
        logger.info("Embeddings berhasil disimpan ke file 'FastRP_embeddings.csv'.")

        # Menghitung statistik embedding
        embeddings_array = np.array([e['embedding'] for e in embeddings_list])
        logger.info(f"Statistik embedding:")
        logger.info(f"Shape: {embeddings_array.shape}")
        logger.info(f"Mean: {np.mean(embeddings_array)}")
        logger.info(f"Std: {np.std(embeddings_array)}")
        logger.info(f"Min: {np.min(embeddings_array)}")
        logger.info(f"Max: {np.max(embeddings_array)}")
    else:
        logger.warning("Tidak ada embedding yang valid untuk disimpan.")

# Menjalankan proses FastRP
if __name__ == "__main__":
    run_fastrp_process(graph)
