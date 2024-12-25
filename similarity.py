import logging
import pandas as pd
from py2neo import Graph
import json

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Neo4j Connection
graph = Graph("bolt://localhost:7687", auth=("neo4j", "211524037"))

try:
    # Check and drop existing graph projection if it exists
    graph_name = "movieGraph"
    logger.info(f"Checking if graph projection '{graph_name}' exists...")
    exists_query = f"CALL gds.graph.exists('{graph_name}') YIELD exists"
    graph_exists = graph.run(exists_query).evaluate("exists")
    
    if graph_exists:
        logger.info(f"Graph projection '{graph_name}' exists. Dropping it...")
        graph.run(f"CALL gds.graph.drop('{graph_name}') YIELD graphName")
        logger.info(f"Graph projection '{graph_name}' dropped successfully.")
    else:
        logger.info(f"No existing graph projection '{graph_name}' found.")

    # Create new graph projection
    timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
    graph_name = f"movieGraph_{timestamp}"
    logger.info(f"Creating graph projection: {graph_name}")

    graph.run(f"""
        CALL gds.graph.project(
            '{graph_name}',
            'Movie',
            '*',
            {{
                nodeProperties: ['embedding']
            }}
        )
    """)

    # Running KNN Algorithm
    logger.info("Running KNN...")
    graph.run(f"""
        CALL gds.knn.write(
            '{graph_name}', 
            {{
                nodeProperties: ['embedding'],
                topK: 1,
                sampleRate: 1.0,
                deltaThreshold: 0.001,
                maxIterations: 10,
                writeRelationshipType: 'KNN',
                writeProperty: 'score'
            }}
        )
    """)

    # Fetching KNN Results
    logger.info("Retrieving KNN results...")
    query = """ 
    MATCH (n1:Movie {type: 'history'})-[r:KNN]->(n2:Movie {type: 'movie'}) 
    RETURN 
        n1.id AS id1, 
        n1.title AS title1, 
        n1.type AS type1, 
        n2.id AS id2, 
        n2.title AS title2, 
        n2.type AS type2, 
        r.score AS similarity 
    ORDER BY id1, similarity DESC 
    """
    knn_results = graph.run(query).data()

    # Processing Results
    df_knn = pd.DataFrame(knn_results)

    if not df_knn.empty:
        df_knn['similarity'] = df_knn['similarity'].astype(float)
        max_similarity = df_knn['similarity'].max() or 1
        df_knn['similarity_normalized'] = df_knn['similarity'] / max_similarity

        # Generating Recommendations
        used_ids = set()
        json_data = []
        for _, group in df_knn.groupby('id1'):
            filtered_group = []
            for _, row in group.iterrows():
                if row['id2'] not in used_ids and row['type2'] == 'movie':
                    used_ids.add(row['id2'])
                    filtered_group.append({
                        "id": row["id2"],
                        "title": row["title2"],
                        "type": row["type2"],
                        "similarity_score": row["similarity_normalized"]
                    })
                if len(filtered_group) == 1:
                    break
            if filtered_group:
                json_data.extend(filtered_group)

        # Saving Results
        with open("Rekomendasi.json", "w", encoding='utf-8') as file:
            json.dump(json_data, file, indent=4, ensure_ascii=False)

        logger.info("Recommendations saved successfully")
    else:
        logger.warning("No KNN results found")

except Exception as e:
    logger.error(f"Error: {e}")
