from main import full_pipe
from pathlib import Path
import os
import csv
import pytz
from datetime import datetime


cwd = Path.cwd()
input_csv = cwd/"test"/"prompt.csv"
image_dir = cwd/"test"/"images"
output_path = cwd/"test"/"results"
timezone = pytz.timezone('Asia/Seoul')#type_your_timezone
now = datetime.now(timezone)
time_f_name = now.strftime("%Y%m%d_%H%M%S")
output_dir = output_path / time_f_name
output_dir.mkdir(parents=True, exist_ok=True)


with open(input_csv, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        file_name = row['image_path']
        image_path = image_dir / file_name
        responses = row['transform']

        full_pipe(image_path, responses, output_dir=output_dir)
