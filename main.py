import openai
from groundascore.groundascore import main
import base64
import requests
from openai import OpenAI
import os
from dotenv import load_dotenv

from transformers import AutoProcessor, AutoModel
import torch

import re
from datetime import datetime
import pytz
from pathlib import Path
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import base64
from io import BytesIO  

import spacy
from collections import defaultdict
import traceback


load_dotenv() 
api_key = os.getenv("OPENAI_API_KEY")




def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG") 
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def load_and_encode(image_path: str, left=0, right=0, top=0, bottom=0):
    image = np.array(Image.open(image_path))[:, :, :3]
    h, w, c = image.shape

    left = max(0, min(left, w-1))#0
    right = max(0, min(right, w-1))#0
    top = max(0, min(top, h-1))#0
    bottom = max(0, min(bottom, h-1))#0

    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape

    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w, :]

    image = Image.fromarray(image)
    return encode_image(image)


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


#######################_OURS_###########################
def sort_groups(phrases):
    nlp = spacy.load("en_core_web_sm")

    groups = defaultdict(list)
    for idx, text in enumerate(phrases):
        nouns = [token.text.lower() for token in nlp(text) if token.pos_ == "NOUN"]
        noun = nouns[0] #phrase 마다 객체 하나 
        groups[noun].append(idx)

    groups = {k: v for k, v in groups.items() if len(v) > 1}
    return groups if groups else False
########################################################


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




def get_grounding_box(source_sentence, image_path, output_dir, box_threshold, 
              get_one_mask=True, with_logits=True, alpha=1, beta=3, tok_k=1, gamma=0.5):
    grounding_sentences = source_sentence
    grounding_sentences[-1] = ' and '.join(grounding_sentences[:-1])
    token_spans = find_phrase_word_indices(grounding_sentences[-1], grounding_sentences[:-1])
    #######################_OURS_###########################
    overlap = sort_groups(grounding_sentences[:-1])
    ########################################################

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

    # _, batch_scores = torch.max(logits, dim=-1)  # (nq,)
    # threshold = 0.25
    # keep = batch_scores > threshold
    # # batch_scores = batch_scores * keep
    # boxes = boxes[keep]
    # logits = logits[keep]

    logits_for_phrases_ = positive_maps @ logits.T # n_phrase, nq
    #######################_OURS_###########################
    if overlap != False:
        logits_for_phrases = logits_for_phrases_.clone()
        modified_positive_maps = positive_maps.clone()
        for _, phrase_indices in overlap.items():
            phrase_indices = list(phrase_indices)
            bbox_k = int(round(alpha * len(phrase_indices)))  # 후보군 개수 alpha로 조정 가능
            selected_logits = logits_for_phrases[phrase_indices]  # (len(phrase_indices), nq)
            mean_logits = selected_logits.mean(dim=0) # 평균값 기준 (nq,)
            topk_indices = torch.topk(mean_logits, bbox_k).indices # 각 group 별 bbox 후보 index

            # 후보 bbox들에 대해서 공통으로 연관이 가장 높은 tok_k개의 token index 추출
            top_logits = logits[topk_indices] #(candidate_boxes, token_labels) 
            hist = torch.topk(top_logits, k=beta, dim=1).indices  # (candidate_boxes, beta)  
            counts = torch.bincount(hist.flatten(), minlength=top_logits.shape[1])
            top_tok_idx = torch.topk(counts, k=tok_k).indices 

            # 추출한 token index에 대한 positive map 값 gamma배
            row_idx = torch.tensor(phrase_indices).unsqueeze(1)  # (n, 1)
            modified_positive_maps[row_idx, top_tok_idx] *= gamma
            
            # 후보 bbox만 남기고 나머지 bbox의 score는 0으로
            # 후보 bbox에 대해서는 수정된 positive map에 대해서 계산된 최종 score 저장 
            logits_for_phrases[phrase_indices] = 0
            temp_logits_for_phrases = modified_positive_maps @ logits.T
            logits_for_phrases[phrase_indices][:,topk_indices] = temp_logits_for_phrases[phrase_indices][:,topk_indices]

    else:
        logits_for_phrases = logits_for_phrases_

    ########################################################
    
    all_logits = []
    all_phrases = []
    all_boxes = []

    phrase_counts = {}
    for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
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
    image_with_box.save(os.path.join(output_dir, "pred.jpg"))
    
    return final_boxes



RETRY_LIMIT = 20

print("""This is image editing interface. Please follow the instructions to edit the image.
image editing interface will ask you to input the image path and the transformation you want to make to the image.
You can make multiple transformations to the image.
For the transform command, enter it in the form change A into B.""")


image_path = input("Please enter image path: ")
# Initialize an empty string to store the transformations
responses = ""
# Ask the user for the initial transformation
response = input("How would you like to transform the image? Please enter in a sentence: ")

# Loop until the user says "no"
while response.lower() != "no":
    # Add the response to the string of transformations
    responses += response + " "
    # Ask if the user wants to make more transformations
    response = input("Do you have any other parts you would like to transform? If you have, please type it in. If not, please type no: ")

# Print the final string of transformations
print("Your requested transformations: ", responses.strip())



# Getting the base64 string
base64_image = load_and_encode(image_path)
attempt_count = 0

while attempt_count < RETRY_LIMIT:
  try:
    
    change = responses

    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}"
    }
    with open('groundascore/LLM_template/template2.txt', 'r') as file:
        template = file.read()
        template3 = template.format(responses=change)
    payload = {
      "model": "gpt-4.1-mini",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": template3
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
          ]
        }
      ],
      "max_tokens": 3000
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    print(response.json())

    input_str = response.json()['choices'][0]['message']['content']

    if "answer" in input_str:
      answer_index = input_str.find('answer') + len('answer')
      input_str = input_str[answer_index:]

    source_index = input_str.find('source_list') + len('source_list')
    
    source_index = input_str.find('source_list') + len('source_list')
    source_start = input_str.find('[', source_index)
    source_end = input_str.find(']', source_start) + 1

    target_index = input_str.find('target_list', source_end) + len('target_list')
    target_start = input_str.find('[', target_index)
    target_end = input_str.find(']', target_start) + 1

    preserve_form_index = input_str.find('preserve_form', target_end) + len('preserve_form')
    preserve_form_start = input_str.find('[', preserve_form_index)
    preserve_form_end = input_str.find(']', preserve_form_start) + 1


    source_sentence_str = input_str[source_start:source_end]
    target_sentence_str = input_str[target_start:target_end]
    preserve_form = input_str[preserve_form_start:preserve_form_end]
    source_sentence = eval(source_sentence_str)
    target_sentence = eval(target_sentence_str)
    preserve_form = eval(preserve_form)

    beta = [0.5,0.4,0.3,0.2,0.1]
    output_dir = Path.cwd()/"output"
    if not os.path.exists(output_dir):
      os.mkdir(output_dir)
    timezone = pytz.timezone('Asia/Seoul')#type_your_timezone
    now = datetime.now(timezone)
    time_f_name = now.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir,time_f_name)
    if not os.path.exists(output_dir):
      os.mkdir(output_dir)

    attempt_count =1000

    bbox = get_grounding_box(source_sentence, image_path, output_dir, box_threshold=0.3).tolist()
    bbox.append([0,0,1,1])
    main(source_sentence, target_sentence,image_path,num_iters = 500,beta = beta, 
        bbox = bbox, output_dir=output_dir,cutloss_flag = preserve_form)
    

  except Exception as e:
        # error_msg = f"Error: {str(e)}\n"
        # print(error_msg)
        traceback.print_exc()
        
        # Log the error message to error.txt
        #with open('error.txt', 'a') as error_file:
        #    error_file.write(error_msg)

        attempt_count += 1
        if attempt_count < RETRY_LIMIT:
            print(f"Retrying... Attempt {attempt_count + 1}/{RETRY_LIMIT}")
        else:
            print("Max retries reached. Moving on to the next file.")   