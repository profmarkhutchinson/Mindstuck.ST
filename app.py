import streamlit as st
import openai
import toml
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
#from icecream import ic
import os

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
os.environ["PINECONE_INDEX_NAME"] = st.secrets["PINECONE_INDEX_NAME"]

openai.api_key = os.environ["OPENAI_API_KEY"]
INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]

def get_vectorstore_openAI():
    embeddings = OpenAIEmbeddings()
    vectorstore = Pinecone.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for message in st.session_state.chat_history:
        st.write(message.content)

def main():
    openai.api_key = os.environ["OPENAI_API_KEY"]

    # Set up page settings
    st.set_page_config(page_title="Chat With Mindstuck PDFs", page_icon=":books:")
    st.header("Chat based on the Mindstuck PDFs 📚")

    # Set up session states
    if "conversation" not in st.session_state:
        st.session_state.conversation = get_conversation_chain(get_vectorstore_openAI())
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Ask a question about the Mindstuck documents:")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Information")
        st.write("Here you can converse with the data extracted from the Mindstuck PDFs using OpenAI and Pinecone.")

if __name__ == '__main__':
    main()
