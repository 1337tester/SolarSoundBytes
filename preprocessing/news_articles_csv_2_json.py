import csv
import json

# Input CSV path
input_csv_path = "../data/csv/articles_cleantech/articles_cleantech.csv"

# Output JSON path
output_json_path = "../data/json/articles_cleantech_combined.json"

# Open the CSV file
with open(input_csv_path, "r") as csv_file:
    csv_reader = csv.DictReader(csv_file)
    data = [row for row in csv_reader]

# Convert the list of dictionaries to JSON format and write to a file
with open(output_json_path, "w") as json_file:
    json.dump(data, json_file, indent=4)

print(f"✅ JSON file created at: {output_json_path}")
