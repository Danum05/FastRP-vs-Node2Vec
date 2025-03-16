import json
import os

# Load data JSON besar
with open("movies.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Tentukan jumlah split
num_splits = 100
chunk_size = len(data) // num_splits

# Pastikan folder penyimpanan ada
output_folder = "split_data"
os.makedirs(output_folder, exist_ok=True)

# Membagi dataset dan menyimpannya
for i in range(num_splits):
    start = i * chunk_size
    end = (i + 1) * chunk_size if i != num_splits - 1 else len(data)
    split_data = data[start:end]

    output_filename = os.path.join(output_folder, f"split_{i+1}.json")
    with open("output.json", "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=True)

print(f"Dataset telah dibagi menjadi {num_splits} file di folder '{output_folder}'")
