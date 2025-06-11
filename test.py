import argparse
from main import full_pipe
from pathlib import Path
import os
import csv
import pytz
from datetime import datetime
import json



def run_test(input_csv, image_dir, output_dir, test_flag=False):

    if test_flag:
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / "predictions.jsonl"
        jsonl_file = open(jsonl_path, "w")
    else:
        jsonl_file = None
        timezone = pytz.timezone('Asia/Seoul')
        now = datetime.now(timezone)
        time_f_name = now.strftime("%Y%m%d_%H%M%S")
        output_dir = output_dir / time_f_name
        output_dir.mkdir(parents=True, exist_ok=True)

    # CSV 읽고 실행
    with open(input_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            file_name = row['image_path']
            image_path = image_dir / Path(file_name).name
            responses = row["transform"]
            image_id = Path(file_name).stem

            predicted_boxes = full_pipe(image_path, responses, output_dir=output_dir, test=test_flag)

            if test_flag and predicted_boxes:
                result = {
                    "image_id": image_id,
                    "predicted_boxes": predicted_boxes
                }
                jsonl_file.write(json.dumps(result) + "\n")

    if jsonl_file:
        jsonl_file.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true", help="Run on benchmark dataset")
    parser.add_argument("--input_csv", type=str, help="Path to CSV file")
    parser.add_argument("--image_dir", type=str, help="Path to image directory")
    parser.add_argument("--output_dir", type=str, default="test/results", help="Base output directory")
    args = parser.parse_args()

    if args.benchmark:
        base = Path.cwd()
        input_csv = base / "benchmark" / "data" / "benchmark_prompts.csv"
        image_dir = base / "benchmark" / "data" / "images"
        output_dir = base / "test" / "results" / "benchmark"
        run_test(input_csv, image_dir, output_dir, test_flag=True)
    else:
        if not args.input_csv or not args.image_dir:
            raise ValueError("For non-benchmark mode, provide --input_csv and --image_dir.")
        run_test(Path(args.input_csv), Path(args.image_dir), Path(args.output_dir), test_flag=False)

    print('test completed.')
