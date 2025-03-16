import logging
import pandas as pd
from py2neo import Graph
import json

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Koneksi ke database Neo4j
try:
    logger.info("Menghubungkan ke database Neo4j...")
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "211524037"))
    logger.info("Koneksi ke Neo4j berhasil.")
except Exception as e:
    logger.error(f"Gagal terhubung ke Neo4j: {e}")
    exit(1)  # Keluar dari program jika koneksi gagal

try:
    # Langsung melewatkan langkah konversi embedding
    logger.info("Melewatkan langkah konversi embedding karena sudah dalam format yang benar...")

    # Cek dan hapus graph projection jika sudah ada
    graph_name = "movieGraph"
    logger.info(f"Memeriksa apakah graph projection '{graph_name}' sudah ada...")
    graph_exists = graph.run("CALL gds.graph.exists($graphName) YIELD exists", graphName=graph_name).evaluate("exists")

    if graph_exists:
        logger.info(f"Menghapus graph projection '{graph_name}' yang sudah ada...")
        graph.run("CALL gds.graph.drop($graphName)", graphName=graph_name)
        logger.info(f"Graph projection '{graph_name}' berhasil dihapus.")

    # Membuat graph projection baru untuk KNN
    logger.info("Membuat graph projection baru...")
    graph.run("""
    CALL gds.graph.project(
        'movieGraph',
        'Movie',
        '*',
        { nodeProperties: ['fastrp_embedding'] }
    )
    """)
    logger.info("Graph projection baru berhasil dibuat.")

    # Menjalankan KNN dengan 'fastrp_embedding'
    logger.info("Menjalankan KNN...")
    graph.run("""
    CALL gds.knn.write(
      'movieGraph',
      {
        nodeProperties: ['fastrp_embedding'],
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

    # Mengambil hasil KNN berdasarkan history
    logger.info("Mengambil hasil KNN dari Neo4j...")
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
        gds.similarity.cosine(n1.fastrp_embedding, n2.fastrp_embedding) AS similarity,
        [(n2)-[:HAS_GENRE]->(g) | g.name] AS genres,
        [(n2)-[:FEATURES]->(a) | a.name] AS actors
    ORDER BY similarity DESC
    """
    knn_results = graph.run(query).data()
    logger.info(f"Jumlah hasil KNN: {len(knn_results)}")

    # Menyaring hasil KNN agar hanya yang memiliki skor similarity > 0 yang disimpan
    filtered_results = [result for result in knn_results if result['similarity'] is not None and result['similarity'] > 0]

    if filtered_results:
        df_knn = pd.DataFrame(filtered_results)

        # Periksa nilai asli similarity sebelum normalisasi
        logger.info("Nilai similarity yang diambil dari Neo4j:")
        logger.info(df_knn[['id1', 'id2', 'similarity']])

        # Normalisasi similarity berdasarkan nilai tertinggi
        df_knn['similarity'] = df_knn['similarity'].astype(float)
        max_similarity = df_knn['similarity'].max()
        df_knn['similarity_normalized'] = df_knn['similarity'] / max_similarity if max_similarity > 0 else 0

        # Inisialisasi set untuk melacak ID film yang sudah digunakan
        used_ids = set()

        # Menyiapkan data JSON tanpa duplikasi
        json_data = []
        for _, group in df_knn.groupby('id1'):
            recommendations = []
            for _, row in group.iterrows():
                if row['id2'] not in used_ids and row['type2'] == 'movie':
                    used_ids.add(row['id2'])
                    recommendations.append({
                        "id": row["id2"],
                        "title": row["title2"],
                        "overview": row["overview2"],
                        "type": row["type2"],
                        "similarity_score": float(row["similarity"])
                    })
                if len(recommendations) == 5:  # Batasi 5 rekomendasi per history
                    break
            if recommendations:
                json_data.extend(recommendations)

        # Menyimpan hasil ke file JSON
        output_file = "Rekomendasi.json"
        with open(output_file, "w", encoding='utf-8') as file:
            json.dump(json_data, file, indent=4, ensure_ascii=False)

        logger.info(f"Hasil KNN 5 teratas berhasil disimpan ke file {output_file}")
    else:
        logger.warning("Tidak ada hasil KNN dengan skor similarity > 0.")

except Exception as e:
    logger.error(f"Terjadi error: {e}", exc_info=True)

finally:
    # Membersihkan graph projection jika masih ada
    try:
        exists = graph.run("CALL gds.graph.exists($graphName) YIELD exists", graphName="movieGraph").evaluate("exists")
        if exists:
            graph.run("CALL gds.graph.drop($graphName)", graphName="movieGraph")
            logger.info("Graph projection berhasil dibersihkan.")
    except Exception as e:
        logger.error(f"Error saat membersihkan graph projection: {e}")