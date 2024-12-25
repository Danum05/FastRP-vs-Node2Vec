# import json

# def load_data(file_path):
#     # Load data from a JSON file
#     with open(file_path, 'r') as file:
#         return json.load(file)

# def calculate_metrics(knn_results_path, data_uji_path):
#     # Load data from files
#     knn_results = load_data(knn_results_path)
#     data_uji = load_data(data_uji_path)

#     # Calculate True Positives and False Positives from knn_results
#     tp = sum(1 for result in knn_results if result['relevan'] == "1")
#     fp = sum(1 for result in knn_results if result['relevan'] == "0")

#     # Calculate False Negatives and True Negatives from data_uji
#     all_data = sum(1 for item in data_uji if item['relevan'] == "1")
#     fn = all_data - tp
#     tn = sum(1 for item in data_uji if item['relevan'] == "0")

#     # Calculate Precision and Recall
#     precision = 100 * tp / (tp + fp) if (tp + fp) > 0 else 0
#     recall = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0

#     # Calculate F1-Score
#     if precision + recall > 0:
#         f1_score = 2 * (precision * recall) / (precision + recall)
#     else:
#         f1_score = 0

#     # Return all calculated values including TN
#     return precision, recall, f1_score, tn

# # Define paths to the JSON files
# knn_results_path = 'Rekomendasi.json'
# data_uji_path = 'pengujian_Sport/groundTruth_Riwayat_Berita_Sport.json'

# # Call the function and print the results
# precision_value, recall_value, f1_score_value, tn_value = calculate_metrics(knn_results_path, data_uji_path)
# print(f"Precision: {precision_value:.2f}%")
# print(f"Recall: {recall_value:.2f}%")
# print(f"F1-Score: {f1_score_value:.2f}%")