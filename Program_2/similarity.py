import logging
import pandas as pd
from py2neo import Graph
import json

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Menghubungkan ke database Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "211524037"))

try:
    # Konversi embedding dari JSON string ke list menggunakan APOC
    logger.info("Mengonversi properti 'embedding' dari JSON string ke list...")
    graph.run("""
        MATCH (n:Movie)
        SET n.embedding = apoc.convert.fromJsonList(n.embedding)
    """)
    logger.info("Konversi embedding selesai.")

    # Cek dan hapus graph projection yang sudah ada
    graph_name = "movieGraph"
    logger.info(f"Memeriksa graph projection '{graph_name}'...")

    exists_query = f"CALL gds.graph.exists('{graph_name}') YIELD exists"
    graph_exists = graph.run(exists_query).evaluate("exists")

    if graph_exists:
        logger.info(f"Menghapus graph projection '{graph_name}' yang sudah ada...")
        graph.run(f"CALL gds.graph.drop('{graph_name}')")
        logger.info(f"Graph projection '{graph_name}' berhasil dihapus.")

    # Membuat graph projection baru untuk KNN
    logger.info("Membuat graph projection baru...")
    graph.run("""
    CALL gds.graph.project(
        'movieGraph',
        'Movie',
        '*',
        {
            nodeProperties: ['embedding', 'fastrp_embedding']
        }
    )
    """)
    logger.info("Graph projection baru berhasil dibuat.")

    # Menjalankan KNN 
    logger.info("Menjalankan KNN...")
    graph.run("""
    CALL gds.knn.write(
      'movieGraph',
      {
        nodeProperties: ['embedding', 'fastrp_embedding'],
        topK: 5,
        sampleRate: 1.0,
        deltaThreshold: 0.001,
        maxIterations: 10,
        writeRelationshipType: 'KNN',
        writeProperty: 'score',
        concurrency: 4
      }
    )
    """)
    logger.info("Proses KNN selesai.")

    # Mengambil hasil KNN dan menerapkan pembobotan manual dengan weight
    logger.info("Mengambil dan membobot hasil KNN dari Neo4j...")
    query = """
    MATCH (n1:Movie {type: 'history'})-[r:KNN]->(n2:Movie {type: 'movie'})
    RETURN 
        n1.id AS id1, 
        n1.title AS title1,
        n1.overview AS overview1,
        n1.type AS type1,
        n2.id AS id2,
        n2.title AS title2,
        n2.overview AS overview2,
        n2.type AS type2,
        gds.similarity.cosine(n1.embedding, n2.embedding) * 0.3 + 
        gds.similarity.cosine(n1.fastrp_embedding, n2.fastrp_embedding) * 0.7 AS similarity,
        [(n2)-[:HAS_GENRE]->(g) | g.name] AS genres,
        [(n2)-[:FEATURES]->(a) | a.name] AS actors
    ORDER BY similarity DESC
    """
    knn_results = graph.run(query).data()
    logger.info("Hasil KNN telah diambil dan dibobot.")

    # Menyaring hasil KNN agar hanya yang memiliki skor lebih besar dari 0 yang disimpan
    filtered_results = [result for result in knn_results if result['similarity'] > 0]

    if filtered_results:
        # Mengkonversi hasil ke DataFrame dan melihat nilai similarity asli
        df_knn = pd.DataFrame(filtered_results)
        
        # Periksa nilai asli similarity sebelum normalisasi
        logger.info("Nilai similarity yang diambil dari Neo4j:")
        logger.info(df_knn[['id1', 'id2', 'similarity']])

        # Normalisasi similarity berdasarkan nilai tertinggi
        df_knn['similarity'] = df_knn['similarity'].astype(float)
        max_similarity = df_knn['similarity'].max()
        df_knn['similarity_normalized'] = df_knn['similarity'] / max_similarity

        # Inisialisasi set untuk melacak ID film yang sudah digunakan
        used_ids = set()

        # Menyiapkan data JSON tanpa normalisasi
        json_data = []
        for _, group in df_knn.groupby('id1'):
            filtered_group = []
            for _, row in group.iterrows():
                if row['id2'] not in used_ids and row['type2'] == 'movie':
                    used_ids.add(row['id2'])
                    filtered_group.append({
                        "id": row["id2"],
                        "title": row["title2"],
                        "overview": row["overview2"],
                        "type": row["type2"],
                        "similarity_score": float(row["similarity"])
                    })
                if len(filtered_group) == 5:  # Membatasi 5 rekomendasi teratas
                    break
            if filtered_group:
                json_data.extend(filtered_group)

        # Menyimpan hasil ke file JSON
        with open("Rekomendasi.json", "w", encoding='utf-8') as file:
            json.dump(json_data, file, indent=4, ensure_ascii=False)

        logger.info("Hasil KNN 5 teratas berhasil disimpan ke file Rekomendasi.json")
    else:
        logger.warning("Tidak ada hasil KNN dengan skor similarity > 0")

except Exception as e:
    logger.error(f"Terjadi error: {e}", exc_info=True)
finally:
    # Membersihkan graph projection
    try:
        graph.run("CALL gds.graph.exists('movieGraph') YIELD exists")
        graph.run("CALL gds.graph.drop('movieGraph')")
        logger.info("Graph projection dibersihkan")
    except Exception as e:
        logger.error(f"Error saat membersihkan graph projection: {e}")