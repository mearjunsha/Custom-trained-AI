import streamlit as st
from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import os

secret = os.environ['msecret']
os.environ["OPENAI_API_KEY"]=secret

INDEX = None

def construct_index(directory_path):
    index_file_path = 'index.json'
    if os.path.exists(index_file_path):
        index = GPTSimpleVectorIndex.load_from_disk(index_file_path)
    else:
        max_input_size = 4096
        num_outputs = 512
        max_chunk_overlap = 20
        chunk_size_limit = 600

        prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

        llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="text-davinci-003", max_tokens=num_outputs))

        documents = SimpleDirectoryReader(directory_path).load_data()

        index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

        index.save_to_disk(index_file_path)

    return index


def chatbot(input_text):
    index = construct_index("docs")
    response = index.query(input_text, response_mode="compact")
    return response.response

def main():
    st.title("Custom-trained AI Chatbot")
    input_text = st.text_area(label="Enter your message", height=200)
    if st.button("Send"):
        response = chatbot(input_text)
        st.text_area(label="Response", value=response, height=200)

if __name__ == '__main__':
    main()
