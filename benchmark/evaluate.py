import json
import os
from PIL import Image
import torch
from pathlib import Path
# import clip

from transformers import CLIPProcessor, CLIPModel
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)



def compute_clip_scores(image_path, pred_boxes, texts):
    """
    Given an image path, list of predicted boxes, and corresponding texts,
    compute CLIP cosine similarity scores for each cropped region vs text.
    Returns a list of floats.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure model in eval mode
    model.eval()

    # Load image and prepare crops
    image = Image.open(image_path).convert("RGB")
    crops = [image.crop(box) for box in pred_boxes]

    # Preprocess crops
    pixel_values = torch.cat([
        processor(images=crop, return_tensors="pt")["pixel_values"] for crop in crops
    ], dim=0).to(device)

    # Tokenize texts (assume len(texts) == len(pred_boxes))
    text_inputs = processor(text=texts, padding=True, return_tensors="pt").to(device)

    # Encode features
    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=pixel_values)
        text_features = model.get_text_features(**text_inputs)

    # Normalize
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Compute cosine similarities (one-to-one by index)
    scores = (image_features * text_features).sum(dim=-1)
    return scores.cpu().tolist()


def compute_iou(boxA, boxB):
    """
    Compute Intersection over Union (IoU) of two bounding boxes.
    Boxes are in [x_min, y_min, x_max, y_max] format.
    """
    boxA = [float(x) for x in boxA]
    boxB = [float(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union_area = boxA_area + boxB_area - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def compute_iou_per_image(gt_boxes, pred_boxes):
    """
    Given lists of ground-truth and predicted boxes for one image,
    compute IoU for each matched pair (ordered) and return a list of IoUs.
    """
    assert len(gt_boxes) == len(pred_boxes), \
        "GT and prediction must have same number of boxes"
    return [compute_iou(gt, pred) for gt, pred in zip(gt_boxes, pred_boxes)]




def main(gt_path, pred_path, image_dir, output_path):
    """
    Load ground-truth and prediction JSONL files, compute IoU and CLIP scores per image,
    and print results.
    JSONL format per line:
      {"image_id": "image1", "boxes": [[x1,y1,x2,y2], ...], "texts": ["label1", ...]}
    Predictions file format:
      {"image_id": "image1", "boxes": [[x1,y1,x2,y2], ...]}

    Output jsonl format:
    {"image1.jpg": {"ious": [0.82, 0.76], "clip_scores": [0.45, 0.38]}, ...}
    """
    iou_results = {}
    clip_results = {}

    with open(gt_path, 'r') as f_gt, open(pred_path, 'r') as f_pred:
        for gt_line, pred_line in zip(f_gt, f_pred):
            gt = json.loads(gt_line)
            pred = json.loads(pred_line)
            image_id = gt['image_file']
            gt_boxes = gt['ground_truth_boxes']
            pred_boxes = pred['predicted_boxes']
            texts = gt.get('texts', [])

            # IoU
            ious = compute_iou_per_image(gt_boxes, pred_boxes)
            iou_results[image_id] = ious

            # CLIP
            img_path = os.path.join(image_dir, image_id+'.jpg')
            clip_scores = compute_clip_scores(img_path, pred_boxes, texts)
            clip_results[image_id] = clip_scores
    
    results = {}
    for image_id in iou_results:
        results[image_id] = {
            "ious": iou_results[image_id],
            "clip_scores": clip_results.get(image_id, [])
        }

    # save to JSON file
    os.makedirs(Path(output_path).parent, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # print("IoU Results per image:")
    # for img, vals in iou_results.items():
    #     print(f"{img}: {vals}")

    # print("\nCLIP Scores per image:")
    # for img, vals in clip_results.items():
    #     print(f"{img}: {vals}")




### run directory : heo_urp/benchmark
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate IoU and CLIP scores.")
    parser.add_argument('--gt', default='/data/GTBOX.jsonl', help='Path to ground-truth JSONL file')
    parser.add_argument('--pred', required=True, help='Path to prediction JSONL file')
    parser.add_argument('--images', default='/data/images', help='Directory containing images')
    parser.add_argument('--output', required=True, help='Path to output dir')
    args = parser.parse_args()
    main(args.gt, args.pred, args.images, args.output)
