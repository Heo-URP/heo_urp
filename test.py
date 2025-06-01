from main import full_pipe
from pathlib import Path
import os
import csv


cwd = Path.cwd()
input_csv = cwd/"test"/"prompt.csv"
output_path = cwd/"test"/"results"


with open(input_csv, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        image_path = row['image_path']
        responses = row['transform']

        path = Path(image_path)
        image_id = path.stem
        output_dir = os.path.join(output_path, image_id)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        full_pipe(image_path, responses, output_dir=output_dir)
