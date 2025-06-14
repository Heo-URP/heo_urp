import json
import csv
import os
from PIL import Image
import torch
from pathlib import Path
from torchvision import transforms
# import clip


from transformers import CLIPModel, CLIPTokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
model.eval()

_clip_mean = (0.48145466, 0.4578275, 0.40821073)
_clip_std  = (0.26862954, 0.26130258, 0.27577711)
_clip_size = (224, 224)


def convert_xywh_to_xyxy(boxes, size):
    """
    Convert normalized or absolute xywh boxes to absolute xyxy.
    boxes: list of [cx, cy, w, h] normalized in [0,1]
    size: (width, height)
    """
    W, H = size
    abs_boxes = []
    for cx, cy, w, h in boxes:
        x0 = (cx - w/2) * W
        y0 = (cy - h/2) * H
        x1 = (cx + w/2) * W
        y1 = (cy + h/2) * H
        abs_boxes.append([x0, y0, x1, y1])
    return abs_boxes

def compute_clip_scores(image_path, prompt):
    img = Image.open(image_path).convert("RGB")

    # prepare transforms
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(_clip_mean, _clip_std)

    img_tensor = normalize(to_tensor(img.resize(_clip_size, Image.BICUBIC))).unsqueeze(0).to(device)
    inputs = tokenizer(text=[prompt], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        img_feat  = model.get_image_features(pixel_values=img_tensor)
        txt_feat  = model.get_text_features(**inputs)
    # normalize and cosine sim
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
    score = (img_feat * txt_feat).sum().item()
    return score




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


def main(gt_path, pred_path, image_dir, prompt_path, output_path):
    """
    Load ground-truth and prediction JSONL files, compute IoU and CLIP scores per image,
    and print results.
    GTBOX JSONL format:
      {"image_id": "image1", "boxes": [[x1,y1,x2,y2], ...], "texts": ["label1", ...]}
    Predictions JSONL format:
      {"image_id": "image1", "boxes": [[x1,y1,x2,y2], ...]}
    Prompt csv format per line:
      image_id.jpg, transform prompt 

    Output jsonl format:
    {"image1.jpg": {"ious": [0.82, 0.76], "clip_scores": [0.45, 0.38]}, ...}
    """

    iou_results = {}
    clip_results = {}

    with open(gt_path, 'r') as f_gt, open(pred_path, 'r') as f_pred, open(prompt_path, 'r') as p:
        reader = csv.DictReader(p)
        for gt_line, pred_line, p_line in zip(f_gt, f_pred, reader):
            gt = json.loads(gt_line)
            pred = json.loads(pred_line)
            image_id = gt['image_file']
            image_path = os.path.join(image_dir, image_id+'.jpg')
            image = Image.open(image_path).convert('RGB')
            size = image.size
            gt_boxes = gt['ground_truth_boxes']
            pred_boxes = pred['predicted_boxes']
            pred_boxes = convert_xywh_to_xyxy(pred_boxes, size)
            prompt = p_line['transform']

            while len(gt_boxes) > len(pred_boxes):
              pred_boxes.append([0.,0.,0.,0.])
              
            assert len(gt_boxes) == len(pred_boxes), \
        f"image id {image_id} GT and prediction must have same number of boxes"

            # IoU
            ious = compute_iou_per_image(gt_boxes, pred_boxes)
            iou_results[image_id] = ious

            # CLIP
            img_path = os.path.join(image_dir, image_id+'.jpg')
            clip_score = compute_clip_scores(img_path, prompt)
            clip_results[image_id] = clip_score
    
    results = {}
    for image_id in iou_results:
        results[image_id] = {
            "ious": iou_results[image_id],
            "clip_score": clip_results.get(image_id, None)
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
    parser.add_argument('--gt', default='./data/GTBOX.jsonl', help='Path to ground-truth JSONL file')
    parser.add_argument('--pred', required=True, help='Path to prediction JSONL file')
    parser.add_argument('--images', default='./data/images', help='Directory containing images')
    parser.add_argument('--prompt', default='./data/benchmark_prompt.csv',help='Path to transform prompt used in editing')
    parser.add_argument('--output', required=True, help='Path to output dir')
    args = parser.parse_args()
    main(args.gt, args.pred, args.images, args.prompt, args.output)
