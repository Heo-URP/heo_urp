import openai
from groundascore.groundascore import main
import base64
import requests
from openai import OpenAI
import os
import traceback

import numpy as np
from PIL import Image
import base64
from io import BytesIO  

#enter your openai api key here
api_key = "put-your-api-key-here"



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



RETRY_LIMIT = 20

def origin(image_path, responses, output_dir = None, test = False):
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
      output_dir = output_dir
      attempt_count =1000
      bbox = main(source_sentence, target_sentence,image_path,num_iters = 500,beta = beta, output_dir=output_dir,cutloss_flag = preserve_form, test_flag=test)

      if bbox:
          return bbox
      

    except Exception as e:
          traceback.print_exc()
          # error_msg = f"Error: {str(e)}\n"
          # print(error_msg)
          
          # Log the error message to error.txt
          #with open('error.txt', 'a') as error_file:
          #    error_file.write(error_msg)

          attempt_count += 1
          if attempt_count < RETRY_LIMIT:
              print(f"Retrying... Attempt {attempt_count + 1}/{RETRY_LIMIT}")
          else:
              print("Max retries reached. Moving on to the next file.")

  print("The image has been edited successfully. check the output folder for the edited image.")    


if __name__ == "__main__":
        
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

    origin(image_path, responses)



