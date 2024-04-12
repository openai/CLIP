import os
import csv

# Input folder path containing the text files
folder_path = '/home/marya/Desktop/zero-shot-object-tracking/runs/detect/6-7-9-09/labels'

# Output CSV file path
output_csv = '/home/marya/Desktop/zero-shot-object-tracking/runs/detect/6-7-9-09/output.csv'

# Define the CSV header
csv_header = ['frame', 'track', 'class', 'bbox']

# Create or open the CSV file for writing in append mode
with open(output_csv, mode='a', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=csv_header)
    
    # Check if the file is empty, if so, write the header
    if os.path.getsize(output_csv) == 0:
        writer.writeheader()

    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            
            # Read each line in the text file and convert it to CSV
            with open(file_path, 'r') as text_file:
                for line in text_file:
                    data = {}
                    parts = line.strip().split('; ')
                    for part in parts:
                        key, value = part.split(': ')
                        data[key.strip()] = value.strip()
                    writer.writerow(data)

print("Conversion completed. Data has been written to", output_csv)
