import json
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

def compute_clip_scores(image_path, pred_boxes, texts):
    """
    Crop regions, resize to 224x224, normalize, and compute CLIP cosine scores.
    """
    model.eval()

    img = Image.open(image_path).convert("RGB")
    size = img.size

    # if max(max(pred_boxes)) <= 1.0: 
    pred_boxes = convert_xywh_to_xyxy(pred_boxes, size)

    # prepare transforms
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(_clip_mean, _clip_std)

    scores = []
    for box, text in zip(pred_boxes, texts):

        x0, y0, x1, y1 = map(float, box)
        # round/clip coordinates
        left   = int(max(0, x0))
        top    = int(max(0, y0))
        right  = int(min(img.width,  x1))
        bottom = int(min(img.height, y1))
        if right <= left or bottom <= top:
            scores.append(0.0)
            continue
        # crop and resize
        crop = img.crop((left, top, right, bottom)).resize(_clip_size, Image.BICUBIC)
        # tensor and normalize
        t = normalize(to_tensor(crop)).unsqueeze(0).to(device)
        # text preprocessing per region
        # lazy import to avoid overhead if not used
        # tokenize text
        inputs = tokenizer(texts=[text], return_tensors="pt", padding=True).to(device)
        # encode
        with torch.no_grad():
            img_feat  = model.get_image_features(pixel_values=t)
            txt_feat  = model.get_text_features(**inputs)
        # normalize and cosine
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        score = (img_feat * txt_feat).sum().item()
        scores.append(score)
    return scores




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

            while len(gt_boxes) > len(pred_boxes):
              pred_boxes.append([0.,0.,0.,0.])
              
            assert len(gt_boxes) == len(pred_boxes), \
        f"image id {image_id} GT and prediction must have same number of boxes"

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
    parser.add_argument('--gt', default='./data/GTBOX.jsonl', help='Path to ground-truth JSONL file')
    parser.add_argument('--pred', required=True, help='Path to prediction JSONL file')
    parser.add_argument('--images', default='./data/images', help='Directory containing images')
    parser.add_argument('--output', required=True, help='Path to output dir')
    args = parser.parse_args()
    main(args.gt, args.pred, args.images, args.output)
