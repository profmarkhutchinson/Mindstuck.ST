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
os.environ["PINECONE_ENVIRONMENT"] = st.secrets["PINECONE_ENVIRONMENT"]
os.environ["PINECONE_NAME_SPACE"] = st.secrets["PINECONE_NAME_SPACE"]
openai.api_key = st.secrets["OPENAI_API_KEY"]
INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]

MODEL_TEMPERATURE = 0.8
MODEL_NAME = "gpt-4"
MAX_TOKENS = 500
SYSTEM_MESSAGE = ("You are a talented professional mentor and coach providing assistance to an executive "
                  "using the resources you have available. Please refer to the resource and respond with a "
                  "detailed and thoughtful response. Please cite the included work and quote it as needed. "
                  "Do not make up answers, but you can elaborate upon ideas and concepts in this resource. "
                  "And if the question is out of scope say sorry and suggest the reader consult the book itself.")


def get_vectorstore_openAI():
    embeddings = OpenAIEmbeddings(model_kwargs={"api_key": OPENAI_API_KEY})
    vectorstore = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore, system_message=SYSTEM_MESSAGE):
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=MODEL_TEMPERATURE, max_tokens=MAX_TOKENS)  
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True,
        initial_messages=[SystemMessage(content=system_message)]
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question, index):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    try:
        embeddings = OpenAIEmbeddings(model_kwargs={"api_key": OPENAI_API_KEY})
        sample_vector = embeddings.embed_documents([user_question])[0]
        
        pinecone_response = index.query(namespace=PINECONE_NAME_SPACE, vector=sample_vector, top_k=3, include_values=False, include_metadata=True)
        
        metadata_and_sources = [match['metadata'] for match in pinecone_response.get('matches', [])]
    except Exception as e:
        st.error(f"Error querying Pinecone index in userinput: {e}")
        metadata_and_sources = []

    # Simply write the message content without any prefixes
    for message in st.session_state.chat_history:
        st.write(message.content)

    if metadata_and_sources:
        for meta in metadata_and_sources:
            filename = meta.get('filename', 'Unknown filename')
            page_number = meta.get('page_number', 'N/A')
            text = meta.get('text', 'No text available')
            
            with st.expander(f"Source: {filename} | Page Number: {page_number}"):
                st.write(text)


def setup_pinecone():
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENVIRONMENT,
    )
    existing_indexes = pinecone.list_indexes()

setup_pinecone()

def main():
    openai.api_key = OPENAI_API_KEY

    st.set_page_config(page_title="Chat With Michael McQueen's Mindstuck", page_icon=":books:", layout="centered", initial_sidebar_state='collapsed')
    st.header("Chat with Mindstuck 📚")

    try:
        embeddings = OpenAIEmbeddings(model_kwargs={"api_key": OPENAI_API_KEY})
        index = pinecone.Index(index_name=PINECONE_INDEX_NAME)
    except Exception as e:
        st.error(f"Error connecting to Pinecone index: {e}")
        return

    if "conversation" not in st.session_state:
        st.session_state.conversation = get_conversation_chain(get_vectorstore_openAI(), SYSTEM_MESSAGE)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_area("Ask a question about how you can use Mindstuck:")

    if user_question:
        handle_userinput(user_question, index)

    with st.sidebar:
        st.subheader("Information")
        st.write("Here you can converse with the brilliance of the ideas created and curated in Michael McQueen's book Mindstuck.")
        st.write("Has this been useful? Maybe you want to get the book, audiobook or perhaps a subscription to this chat service. email AAA@BBBB.com to join the waitlist")

if __name__ == '__main__':
    main()
