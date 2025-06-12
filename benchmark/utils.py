import json
import ast
import argparse
from pathlib import Path

def get_average_scores(file_path):
    """
    Compute dataset-level mean IoU and mean CLIP score from an evaluation file.
    Supports both JSON (single dict) and JSONL (one JSON per line).

    Returns:
      dataset_mean_iou: float
      dataset_mean_clip: float
    """
    file_path = Path(file_path)
    content = file_path.read_text(encoding='utf-8').strip()

    # Determine format: pure JSON dict vs JSONL
    if content.startswith('{') and content.endswith('}'):
        # Single JSON object
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # fallback to ast
            data = ast.literal_eval(content)
        records = []
        for img_id, metrics in data.items():
            records.append((img_id, metrics))
    else:
        # JSONL or mixed lines
        records = []
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                try:
                    rec = ast.literal_eval(line)
                except Exception:
                    continue
            if isinstance(rec, dict) and 'image_id' in rec:
                img_id = rec['image_id']
                metrics = rec
            elif isinstance(rec, dict) and len(rec)==1 and isinstance(list(rec.values())[0], dict):
                img_id, metrics = list(rec.items())[0]
            else:
                continue
            records.append((img_id, metrics))

    # Aggregate scores
    all_mean_ious = []
    all_mean_clips = []
    for img_id, metrics in records:
        ious = metrics.get('ious', metrics.get('iou', []))
        clips = metrics.get('clip_scores', metrics.get('clip', []))
        if not isinstance(ious, (list, tuple)) or not isinstance(clips, (list, tuple)):
            continue
        mean_iou = sum(ious)/len(ious) if ious else 0.0
        mean_clip = sum(clips)/len(clips) if clips else 0.0
        all_mean_ious.append(mean_iou)
        all_mean_clips.append(mean_clip)

    dataset_mean_iou = sum(all_mean_ious)/len(all_mean_ious) if all_mean_ious else 0.0
    dataset_mean_clip = sum(all_mean_clips)/len(all_mean_clips) if all_mean_clips else 0.0
    return dataset_mean_iou, dataset_mean_clip

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute dataset-level mean IoU and CLIP score')
    parser.add_argument('file_path', help='Evaluation file (.json or .jsonl)')
    args = parser.parse_args()
    miou, mclip = get_average_scores(args.file_path)
    print(f"Dataset mean IoU: {miou:.4f}")
    print(f"Dataset mean CLIP score: {mclip:.4f}")



# import json

# def get_average_scores(file_path):
#     """
#     1) 각 이미지에 대해 ious, clip_scores 리스트를 평균내고 (image_means)
#     2) 이미지별 평균을 모아 데이터셋 전체 평균을 계산합니다.
    
#     Returns:
#       dataset_mean_iou: float
#       dataset_mean_clip: float
#     """
#     all_mean_ious  = []
#     all_mean_clips = []

#     with open(file_path, 'r', encoding='utf-8') as f:
#         for raw_line in f:
#             line = raw_line.strip()
#             if not line:
#                 continue
#             # json.load → json.loads
#             rec = json.loads(line)

#             # { "image_id": {"ious":…, "clip_scores":…} } 형태 처리
#             img_id = rec.get("image_id", list(rec.keys())[0])
#             if "ious" not in rec:
#                 rec = rec[img_id]

#             ious        = rec.get("ious", [])
#             clip_scores = rec.get("clip_scores", rec.get("clip", []))

#             mean_iou  = sum(ious)  / len(ious)        if ious        else 0.0
#             mean_clip = sum(clip_scores) / len(clip_scores) if clip_scores else 0.0

#             all_mean_ious.append(mean_iou)
#             all_mean_clips.append(mean_clip)

#     dataset_mean_iou  = sum(all_mean_ious)  / len(all_mean_ious)  if all_mean_ious  else 0.0
#     dataset_mean_clip = sum(all_mean_clips) / len(all_mean_clips) if all_mean_clips else 0.0

#     return dataset_mean_iou, dataset_mean_clip

# if __name__ == '__main__':
#     file_path = 'results/ours/evaluations.jsonl'
#     get_average_scores(file_path)