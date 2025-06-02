
from transformers import AutoProcessor, AutoModel
import torch

import os
import re 
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import spacy
from pathlib import Path
from collections import defaultdict



def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    i= -1
    final_boxes = []
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        i+=1

        box2 = box.clone()
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]

        # from xywh to xyxy
        box2[:2] -= box2[2:] / 2
        box2[2:] += box2[:2]
        boxes[i] = box2
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)


        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)
    
    final_boxes = boxes

    return image_pil, mask ,final_boxes



def find_phrase_word_indices(sentence, phrases):
    # Tokenize the sentence and keep track of each word's start and end indices
    sentence = re.sub(r'[^\w\s]', '', sentence)
    phrases = [re.sub(r'[^\w\s]', '', phrase) for phrase in phrases]
    print("s",sentence)
    print("p",phrases)
    words_in_sentence = []
    start = 0
    for word in sentence.split():
        end = start + len(word)
        words_in_sentence.append((word, start, end))
        start = end + 1  # +1 for the space

    # Function to find all occurrences of a phrase and individual word indices within the phrase
    def find_occurrences(phrase):
        phrase_words = phrase.split()
        occurrences = []
        for i in range(len(words_in_sentence)):
            if words_in_sentence[i][0] == phrase_words[0]:
                match = True
                occurrence_indices = []
                for j in range(len(phrase_words)):
                    if i+j >= len(words_in_sentence) or words_in_sentence[i+j][0] != phrase_words[j]:
                        match = False
                        break
                    else:
                        occurrence_indices.append([words_in_sentence[i+j][1], words_in_sentence[i+j][2]])
                if match:
                    occurrences = occurrence_indices
        print(occurrences)
        
        return occurrences

    # Finding indices for each phrase
    results = [find_occurrences(phrase) for phrase in phrases]

    return results



def sort_groups(objects):

    grouped_indices = defaultdict(list)

    for idx, obj in enumerate(objects):
        grouped_indices[obj].append(idx)
    filtered = {k: v for k, v in grouped_indices.items() if len(v) >= 2}
    return filtered if filtered else False



def create_positive_map_from_span(tokenized, token_span, max_text_len=256):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j
    Input:
        - tokenized:
            - input_ids: Tensor[1, ntokens]
            - attention_mask: Tensor[1, ntokens]
        - token_span: list with length num_boxes.
            - each item: [start_idx, end_idx]
    """
    positive_map = torch.zeros((len(token_span), max_text_len), dtype=torch.float)
    for j, tok_list in enumerate(token_span):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            if os.environ.get("SHILONG_DEBUG_ONLY_ONE_POS", None) == "TRUE":
                positive_map[j, beg_pos] = 1
                break
            else:
                positive_map[j, beg_pos : end_pos + 1].fill_(1)

    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)




def get_grounding_box(source_sentence, image_path, output_dir, objects, box_threshold, 
              get_one_mask=True, with_logits=True, alpha=1.5, gamma=1.0):
    grounding_sentences = source_sentence
    grounding_sentences[-1] = ' and '.join(grounding_sentences[:-1])
    token_spans = find_phrase_word_indices(grounding_sentences[-1], grounding_sentences[:-1])
    overlap = sort_groups(objects)

    model_id = "IDEA-Research/grounding-dino-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)

    image = Image.open(image_path).convert("RGB")
    text = grounding_sentences[-1]
    text = text.lower()
    text = text.strip()
    if not text.endswith('.'):
      text = text+'.'

    inputs = processor(images=image, text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs["encoder_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["encoder_pred_boxes"][0]  # (nq, 4)

    # given-phrase mode
    positive_maps = create_positive_map_from_span(
        processor.tokenizer(text),
        token_span=token_spans
    ).to(device) # n_phrase, 256


    _, batch_scores = torch.max(logits, dim=-1)  # (nq,)
    threshold = 0.25
    keep = batch_scores > threshold
    # batch_scores = batch_scores * keep
    boxes = boxes[keep]
    logits = logits[keep]

    logits_for_phrases_ = positive_maps @ logits.T # n_phrase, nq

    if overlap != False:
        objects_token_spans = find_phrase_word_indices(grounding_sentences[-1], objects)
        object_positive_maps = create_positive_map_from_span(
            processor.tokenizer(text),
            token_span = objects_token_spans
        ).to(device)
        object_scores = object_positive_maps @ logits.T

        logits_for_phrases = logits_for_phrases_.clone()
        modified_positive_maps = (positive_maps - gamma * object_positive_maps).clamp(min=0)

        for _, phrase_indices in overlap.items():
            # threshold = 0.25
            # mask = logits_for_phrases[phrase_indices] > threshold  # shape: (n_subset, n_queries)
            # bbox_k = int(mask.sum(dim=1).float().mean().item())  # shape: (n_subset,)
            # topk_indices = torch.topk(object_scores[phrase_indices], k=bbox_k, dim=1).indices
            bbox_k = int(alpha * len(phrase_indices))
            topk_indices = torch.topk(object_scores[phrase_indices], k=bbox_k, dim=1).indices # (len(phrase_indices), bbox_k)

            logits_for_phrases[phrase_indices] = 0
            modified_logits_for_phrases = modified_positive_maps @ logits.T
            gathered_values = torch.gather(modified_logits_for_phrases[phrase_indices], dim=1, index=topk_indices)
            row_indices = torch.tensor(phrase_indices).unsqueeze(1).expand(-1, topk_indices.size(1))
            logits_for_phrases[row_indices, topk_indices] = gathered_values

    else:
        logits_for_phrases = logits_for_phrases_

    
    all_logits = []
    all_phrases = []
    all_boxes = []

    phrase_counts = {}
    for i, (token_span, logit_phr) in enumerate(zip(token_spans, logits_for_phrases)):
        # get phrase
        phrase = ' '.join([text[_s:_e] for (_s, _e) in token_span])
        phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        current_count = phrase_counts[phrase]
        sorted_indices = torch.argsort(logit_phr, descending=True)
        selected_index = sorted_indices[min(current_count - 1, len(sorted_indices) - 1)]
        # get mask
        if get_one_mask:
            filt_mask = torch.zeros_like(logit_phr, dtype=torch.bool)  
            filt_mask[selected_index] = True
        else:
            filt_mask = logit_phr > box_threshold
        #filt_mask = logit_phr > box_threshold

        # filt box
        all_boxes.append(boxes[filt_mask])
        # filt logits
        all_logits.append(logit_phr[filt_mask])
        if with_logits:
            logit_phr_num = logit_phr[filt_mask]
            all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
        else:
            all_phrases.extend([phrase for _ in range(len(filt_mask))])
    boxes_filt = torch.cat(all_boxes, dim=0).cpu()
    pred_phrases = all_phrases

    size = image.size
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
        "labels": pred_phrases,
    }
    image_with_box  , _ , final_boxes= plot_boxes_to_image(image, pred_dict)
    image_with_box.save(os.path.join(output_dir, f"{Path(image_path).stem}_pred.jpg"))
    
    return final_boxes