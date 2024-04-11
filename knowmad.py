import getpass
import os
import io
import openai
from llama_index.core import SimpleDirectoryReader,  ServiceContext,VectorStoreIndex, KnowledgeGraphIndex
from llama_index.embeddings.together import TogetherEmbedding
from llama_index.llms.together import TogetherLLM
import logging
import sys
import llama_index.core
from openai import OpenAI
from langchain import hub
from langchain_community.llms import OpenAI
from langchain.agents import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core.evaluation import RetrieverEvaluator
import PIL.Image
from googleapiclient.discovery import build
import google.generativeai as genai
import nest_asyncio
# from app import uploaded_image
# from pytube import YouTube
# from pydub import AudioSegment
# from IPython.display import Audio, display
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# import gradio as gr
# import os
# import cv2
# from imagehash import dhash
# from PIL import Image
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import numpy as np
# import csv
# import torch
# from pathlib import Path
# from torchvision import transforms
# from IPython.display import display
# from openai import OpenAI
# import assemblyai as aai
# import layoutparser as lp
# from llama_index.core import SimpleDirectoryReader
# from llama_index.multi_modal_llms.anthropic import AnthropicMultiModal
# from docx import Document
# from docx.shared import Inches
# import pytesseract
# from transformers import pipeline
# import pandas as pd
# from docx import Document
# from docx.shared import Inches
# from PIL import Image
# import os
# from docx import Document
# from docx.shared import Inches
# from flask import Flask

#history of chat
chat_history_context=5

nest_asyncio.apply()

genai.configure(api_key="AIzaSyBJMRsN0TS8TsUdynlPHgoHcLHlQVPtZmo")
gemini_model = genai.GenerativeModel('gemini-pro')
openai.api_key = "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"
youtube_api_key = "AIzaSyCZsOMZX_w0QVT2r0UcKTcqWbPgT04UJbk"
youtube = build("youtube", "v3", developerKey=youtube_api_key)
GOOGLE_API_KEY="AIzaSyBJMRsN0TS8TsUdynlPHgoHcLHlQVPtZmo"
genai.configure(api_key=GOOGLE_API_KEY)
timestamping_model = genai.GenerativeModel('gemini-pro')
vision_model = genai.GenerativeModel('gemini-pro-vision')

from langchain_community.llms import OpenAI
documents = SimpleDirectoryReader('training').load_data()
vector_index = VectorStoreIndex.from_documents(documents)
prompt = hub.pull("hwchase17/react")
model = OpenAI(api_key="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")
tools = [
    Tool(
      name = "LlamaIndex",
      func=lambda q: str(vector_index.as_query_engine().query(q)),
      description="useful for when you want to answer questions about the author. The input to this tool should be a complete english sentence.",
      return_direct=True
    ),
]
conversational_memory = ConversationBufferWindowMemory( memory_key='chat_history', k=5, return_messages=True )
# Create the react agent without using 'bind' method
agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

#document description from gemini
def document_description(document_text):
  response = gemini_model.generate_content("Write a very short 10-15 words description of these document and the info this contains"+document_text, stream=True)
  response.resolve()
  return response.text

#checking if the chatbot cananswer the query correctly
def can_chatbot_answer(input_message,description):
  from openai import OpenAI
  os.environ['OPENAI_API_KEY'] = "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"
  client = OpenAI()
  summar_text=input_message+"\nWill the following pipeline be able to answer the above question(write only yes/no)?\n"+"""prompt = hub.pull("hwchase17/react")
              model = OpenAI()
              documents = SimpleDirectoryReader('training').load_data()
              vector_index = VectorStoreIndex.from_documents(documents)
              tools = [
                  Tool(
                    name = "LlamaIndex",
                    func=lambda q: str(vector_index.as_query_engine().query(q)),
                    description="useful for when you want to answer questions about the author. The input to this tool should be a complete english sentence.",
                    return_direct=True
                  ),
              ]
              conversational_memory = ConversationBufferWindowMemory( memory_key='chat_history', k=5, return_messages=True )
              agent = create_react_agent(model, tools, prompt)
              agent_executor = AgentExecutor(agent=agent, tools=tools)

              def chatbot_interaction(input_message):
                prompt = input_message
                response = agent_executor.invoke({'input':prompt})
                return response['output']"""+"\n the folder test contains"+description
  summarized_text = client.chat.completions.create(model="gpt-3.5-turbo",messages=[{"role": "user", "content": summar_text}]).choices[0].message.content
  return summarized_text

#document addition to database and increasing history context
def document_addition(k):
  #loading more data
  #message to add a more relevant document in order to get better results
  documents = SimpleDirectoryReader('training').load_data()
  vector_index = VectorStoreIndex.from_documents(documents)
  tools = [
      Tool(
        name = "LlamaIndex",
        func=lambda q: str(vector_index.as_query_engine().query(q)),
        description="useful for when you want to answer questions about the author. The input to this tool should be a complete english sentence.",
        return_direct=True
      ),
  ]
  conversational_memory = ConversationBufferWindowMemory( memory_key='chat_history', k=k+2, return_messages=True )
  agent = create_react_agent(model, tools, prompt)
  agent_executor = AgentExecutor(agent=agent, tools=tools)
#document(chat_history_context)

#checking if content is safe to be viewed
def check_result_publishable(input_message,document_description):
  response = gemini_model.generate_content(input_message+"Is it safe to provide the answer of the above query based on the document below such that it does not violate any copyrights, safety of any country and ensures societal saftely and ethnicity\n"+document_description+"answer in yes or no", stream=True)
  response.resolve()
  if response.resolve() in ['No','no','NO']:
    return "Content is not safe to be viewed!"
  return response.text

def chatbot_interaction(input_message,agent_executor):
  prompt = input_message
  response = agent_executor.invoke({'input':prompt})
  return response['output']

def evaluation_check(text_content):
  from langchain_community.llms import OpenAI
  logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
  logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
  llama_index.core.set_global_handler("simple")
  llm = OpenAI(model="gpt-3.5-turbo", temperature=0.0)
  evaluator = FaithfulnessEvaluator(llm=llm)
  query_engine = vector_index.as_query_engine()
  response = query_engine.query("what is dk in Scaled Dot-Product Attention?")
  eval_result = evaluator.evaluate_response(response=response)
  retriever = vector_index.as_retriever(similarity_top_k=2)
  retriever_evaluator = RetrieverEvaluator.from_metric_names(
      ["mrr", "hit_rate"], retriever=retriever
  )
  print(
  retriever_evaluator.evaluate(
      query="what is dk in Scaled Dot-Product Attention?", expected_ids=[text_content]
  ))
  return str(eval_result.passing)

def tell_answer_from_uploaded_material(input_message):
  document_desc=document_description(documents[0].text)
  print('documents retrieved')
  if can_chatbot_answer(input_message,document_desc) in ['no','No','NO']:
    document_addition(chat_history_context)
  print('chatbot can answer this')
  if check_result_publishable(input_message,document_desc) in ['no','No','NO']:
    return "Sorry! This search is a violation of civil and constitutional harmony and can't be displayed..."
  print('result is publishable')
  text_result=chatbot_interaction(input_message,agent_executor)
  print("interacted with chatbot!")
  # if evaluation_check(documents[0].text) in ['False','FALSE','false']:
  #   document_addition(chat_history_context)
  # print('evaluation check completed')
  return text_result

"""**EXPLAIN NOTES**"""

def take_notes(img,user_query):
  response = vision_model.generate_content(["what is written in these notes, with what topic are they related to?, jus't write the name of the topic, the user query was "+user_query, img], stream=True)
  response.resolve()
  return response.text

def search_youtube_for_links(search_query):
  # Call the search.list method to retrieve results
  search_response = youtube.search().list(
      q=search_query,
      part="snippet",
      maxResults=10  # Adjust as needed
  ).execute()
  new_list=[]
  # Parse the response and extract video information
  for search_result in search_response.get("items", []):
      if search_result["id"]["kind"] == "youtube#video":
          video_title = search_result["snippet"]["title"]
          video_id = search_result["id"]["videoId"]
          video_url = f"https://www.youtube.com/watch?v={video_id}"
          new_list.append([video_title,video_url])
  return new_list

def timestamping(url,user_query):
  response = timestamping_model.generate_content(url+"you are a video timestamper ,see this and tell me what are the relevant timestamps for the below query:"+user_query)
  able_response = timestamping_model.generate_content(response.text+"this is the info from a timestamper, if the response doesn't contain any timestamps or info the he wasn't able to.was he able to time stamp the video?(answer in yes/no)")
  return [able_response.text,response.text]

def give_video_explanation(user_query,counter,img):
  topic=take_notes(img,user_query)
  links_list=search_youtube_for_links(topic)
  if counter<=len(links_list):
    timestamped_response=timestamping(links_list[counter][1],user_query)
    if timestamped_response[0] in ['no','No','NO']:
      return links_list[counter][1]
    return links_list[counter][1]+"&t=10s"
  else:
    return "No more videos to show..."

# """**GENERATE NOTES**"""

# !pip install selenium pytube torchaudio sentencepiece transformers pydub openai imagehash pytorch sentencepiece assemblyai llama-index-multi-modal-llms-anthropic llama-index-vector-stores-qdrant matplotlib python-docx opencv-python pytesseract llama-index-readers-file
# !pip install git+https://github.com/huggingface/transformers -q
# !pip install -r requirements.txt
# !pip freeze | grep transformers
# !pip install transformers -U -q
# !pip install layoutparser torchvision && pip install "detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.5#egg=detectron2"
# !sudo apt-get install tesseract-ocr



# aai.settings.api_key="6b56941c69c74347af5a94aca1d1a73f"
# client=OpenAI(api_key="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")
# os.environ["ANTHROPIC_API_KEY"] = "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"

# model = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.65],
#                                  label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
# image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

# def split_text_into_segments(text, max_tokens=16385):
#     segments = []
#     current_segment = ""
#     current_length = 0
#     for word in text.split():
#         word_length = len(word)
#         if current_length + word_length + 1 > max_tokens:
#             segments.append(current_segment)
#             current_segment = ""
#             current_length = 0
#         current_segment += word + " "
#         current_length += word_length + 1
#     if current_segment:
#         segments.append(current_segment)
#     return segments

# def calculate_hash(image_path):
#     image = cv2.imread(image_path)
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     pil_image = Image.fromarray(gray_image)
#     resized_image = pil_image.resize((9, 8), Image.ANTIALIAS)
#     return str(dhash(resized_image))

# def remove_similar_frames(input_folder, output_folder):
#     hashes = set()
#     for filename in os.listdir(input_folder):
#         if filename.endswith(".jpg"):
#             image_path = os.path.join(input_folder, filename)
#             image_hash = calculate_hash(image_path)
#             if image_hash not in hashes:
#                 hashes.add(image_hash)
#                 output_path = os.path.join(output_folder, filename)
#                 os.makedirs(output_folder, exist_ok=True)
#                 os.rename(image_path, output_path)

# def extract_diagram(image_path,output_folder,counter):
#   image = cv2.imread(image_path)
#   image = image[..., ::-1]
#   layout = model.detect(image)
#   figure_blocks = lp.Layout([b for b in layout if b.type == 'Figure'])
#   for idx, block in enumerate(figure_blocks):
#       x, y, w, h = map(int, block.coordinates)
#       figure_image = image[y:y + h, x:x + w]
#       figure_image_path = os.path.join(output_folder, f'Figure_{idx + 1+counter}.png')
#       cv2.imwrite(figure_image_path, figure_image)

# def apply_ocr(image_path):
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     text = pytesseract.image_to_string(thresh)
#     return text

# def extract_caption(image_path):
#   text=image_to_text(image_path)[0]['generated_text']
#   return text

# def apply_ocr_and_caption_to_folder(input_folder, output_csv):
#     with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
#         csv_writer = csv.writer(csvfile)
#         #csv_writer.writerow(['Image Name', 'OCR Text', 'Caption'])  # Write header
#         for filename in os.listdir(input_folder):
#             if filename.endswith('.jpg') or filename.endswith('.png'):
#                 image_path = os.path.join(input_folder, filename)
#                 ocr_text = apply_ocr(image_path)
#                 caption = extract_caption(image_path)
#                 csv_writer.writerow([filename, ocr_text, caption])
#     print("OCR and caption extraction completed. Results saved to", output_csv)

# def insert_image(doc, image_path):
#     doc.add_picture(image_path, width=Inches(3))  # Adjust width as needed
#     doc.add_paragraph()

# def create_word_document(notes, images_folder, output_filename):
#     doc = Document()
#     for line in notes.split('\n'):
#         if line.strip().startswith("Figure_"):
#             image_filename = line.strip().split(" ")[0]
#             image_path = os.path.join(images_folder, image_filename)
#             if os.path.exists(image_path):
#                 insert_image(doc, image_path)
#         else:
#             doc.add_paragraph(line.strip())

#     doc.save(output_filename)
#     print("Word document created successfully:", output_filename)

# def normal_tutorial(yt_url,thumbnail_title):
#   youtube = YouTube(yt_url)
#   video_stream = youtube.streams.filter(only_audio=False).first()
#   video_stream.download(output_path=".", filename="Downloaded-video.mp4")
#   print("Video downloaded")
#   audio = AudioSegment.from_file("Downloaded-video.mp4", format="mp4")
#   audio.export("output-audio.mp3", format="mp3")
#   audio_length = len(audio)/60000
#   print("Audio downloaded")
#   config=aai.TranscriptionConfig(language_code="hi")
#   transcriber=aai.Transcriber(config=config)
#   transcript=transcriber.transcribe("output-audio.mp3")
#   original_text = transcript.text
#   print("audio converted to text")
#   text_segments = split_text_into_segments(original_text)
#   headed_segments = []
#   for segment in text_segments:
#     translation_prompt = segment + "\nThis is what the narrator of a youtube video lecture said, give me the complete end to end english translation of the above hindi text, don't add anything else, just give me the translated text"
#     translated_text = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": translation_prompt}])
#     headed_prompt=translated_text.choices[0].message.content+"\nThis is the what the narrator said in a video lecture.give me the notes of this lecture labeled with heading and explanation under each heading for what the narrator said  in bullets, each bullet should have 20-30 words, make sure that u don't skip any information of the video and don't add any of your's and don't write anyother thing else, just what i said."
#     headed_segments.append(client.chat.completions.create(model="gpt-3.5-turbo",messages=[{"role": "user", "content":headed_prompt}]).choices[0].message.content)
#   headed_segments_combined = " ".join(headed_segments)
#   #diagram extraction
#   output_folder="/content/frame_contents"
#   os.makedirs(output_folder, exist_ok=True)
#   cam=cv2.VideoCapture("/content/Downloaded-video.mp4")
#   n=0
#   i=0
#   while True:
#     ret,frame=cam.read()
#     if ret==False:
#       break
#     if n%100==0:
#       cv2.imwrite(os.path.join(output_folder, "{}.jpg".format(n)), frame)
#     n+=1
#   if __name__ == "__main__":
#     input_folder = "/content/frame_contents"  # Update with your input folder
#     output_folder = "/content/unique_frames"  # Update with your output folder

#     remove_similar_frames(input_folder, output_folder)
#   os.makedirs("/content/extracted_unique_diagrams", exist_ok=True)
#   counter=0
#   for image in os.listdir("/content/unique_frames"):
#     extract_diagram(os.path.join("/content/unique_frames/",image),"/content/extracted_unique_diagrams",counter)
#     counter+=1
#   input_folder = '/content/extracted_unique_diagrams'
#   output_csv = 'images_data.csv'
#   apply_ocr_and_caption_to_folder(input_folder, output_csv)
#   df = pd.read_csv('images_data.csv')
#   allignment_prompt=headed_segments_combined+"these are the notes I made from a youtube video with the title of :"+thumbnail_title+"\n the following is the data of images along with their names and what are they about, if you anything relevant about the image with that particular part of notes, then place the name of the image in the notes at that place and leaving a line before and after the name, don't delete anything from the notes, just place the following pictures in the manner i told\n"+str(df)
#   updated_notes = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": allignment_prompt}]).choices[0].message.content
#   doc = Document()
#   doc.add_paragraph(headed_segments_combined)
#   for item in os.listdir('/content/extracted_unique_diagrams'):
#     image_path=os.path.join('/content/extracted_unique_diagrams/'+item)
#     doc.add_picture(image_path, width=Inches(4))
#   doc.save("Notes_by_noteTHAT.docx")
#   notes =updated_notes
#   images_folder = "/content/extracted_unique_diagrams"
#   output_filename = "notes_by_noteTHAT.docx"
#   create_word_document(notes, images_folder, output_filename)
#   return headed_segments_combined







