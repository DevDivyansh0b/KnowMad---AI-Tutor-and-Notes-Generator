
# KnowMad - AI tutor and Notes Generator

A Notes explanation Chatbot that can be provided with context uses GenAI and RAG using llama_index over provided documents and images. Provision of YouTube videos for explanation along with Diagram and Notes Retrieval over given videos.


## API References

#### Get all items

Get the following api keys from respective links and save them as environment variables.
| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `youtube_api_key` | `string` | **Get From:** https://developers.google.com/youtube/v3/getting-started |
| `openai_api_key` | `string` | **Get From:** https://platform.openai.com/account/organization |
| `gemini_api_key` | `string` | **Get From:** https://aistudio.google.com/app/prompts/new_chat?utm_source=agd&utm_medium=referral&utm_campaign=core-cta&utm_content= |

## Terminal setup
Open vs code and create a new virtual environment by following commands
```bash
  python -m venv myenv
```
```bash
  myenv\Scripts\activate
```
Install the required libraries for the project using the following pip install

```bash
pip install google-generativeai llama-index llama-index-embeddings-together llama-index-llms-together openai langchain langchainhub llama-index-llms-langchain streamlit google-api-python-client
pip install selenium pytube torchaudio sentencepiece transformers pydub openai imagehash pytorch sentencepiece assemblyai llama-index-multi-modal-llms-anthropic llama-index-vector-stores-qdrant matplotlib python-docx opencv-python pytesseract llama-index-readers-file
pip install git+https://github.com/huggingface/transformers -q
pip install -r requirements.txt
pip freeze | grep transformers
pip install transformers -U -q
pip install layoutparser torchvision && pip install "detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.5#egg=detectron2"
sudo apt-get install tesseract-ocr
```
## Python files
put the respective codes in knowmad.py and app.y files code in a file knowmad.py, This file contains all important functions necessary to perform the deep learning tasks of Notes generation, Youtube video extraction and Chatbot interaction.
Make a directory 'test' which contains images you want to give to the model and another directory 'training' which will contain documents your RAG model will be trained on.
## Deployment

To deploy this project using streamlit run the following command in your app.py file

```bash
  streamlit run app.py
```

