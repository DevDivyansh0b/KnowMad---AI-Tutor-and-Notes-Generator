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

