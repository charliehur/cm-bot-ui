import os

import streamlit as st

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import PyPDFLoader
import pinecone as pinecone
import pandas as pd
import random


@st.cache_resource
def load_retriever():
    data = []

    data.extend(CSVLoader(file_path="FiguresOfAmericanWest.csv", encoding='ISO-8859-1').load())
    data.extend(CSVLoader(file_path="PBSDocumentaries.csv", encoding='ISO-8859-1').load())
    data.extend(CSVLoader(file_path="Debate_topics.csv", encoding='ISO-8859-1').load())
    data.extend(PyPDFLoader(file_path="lyrics.pdf").load_and_split())

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    # initialize pinecone
    pinecone.init(
        api_key=os.environ['PINECONE_API_KEY'],  # find at app.pinecone.io
        environment=os.environ['PINECONE_API_ENV']  # next to api key in console
    )
    index_name = os.environ['PINECONE_INDEX_NAME'] # put in the name of your pinecone index here

    metadata_field_info = []
    #vector_store = Pinecone.from_existing_index(index_name=os.environ["PINECONE_INDEX_NAME"], embedding=OpenAIEmbeddings())
    vector_store = Pinecone.from_documents(data, embeddings, index_name=index_name)
    llm = OpenAI(model_name="gpt-4", temperature=1.25, logit_bias={"8237":-100, "47553": -100,  "4462": -100})
    document_content_description = "lyric content and lyric style"

    return SelfQueryRetriever.from_llm(
        llm, vector_store, document_content_description, metadata_field_info
    )

def get_innovation():
    # Load the CSV into a DataFrame
    df = pd.read_csv("innovations.csv")

    # Extract topics from the "Debate_Topic" column into a list
    innovations = df["Innovation"].tolist()

    # Randomly select a topic
    selected_innovation = random.choice(innovations)

    return selected_innovation

def get_topic():
    # Load the CSV into a DataFrame
    df = pd.read_csv("Debate_topics.csv")

    # Extract topics from the "Debate_Topic" column into a list
    topics = df["Debate_Topic"].tolist()

    # Randomly select a topic
    selected_topic = random.choice(topics)

    return selected_topic

def get_story_structure():
    # Load the CSV into a DataFrame
    df = pd.read_csv("story_structures.csv")

    # Extract topics from the "Debate_Topic" column into a list
    story_structure = df["narrative"].tolist()

    # Randomly select a topic
    selected_story_structure = random.choice(story_structure)

    return selected_story_structure

def load_analogy_chain():
    
    template = """{innovation}:{topic} :: {question}:_______
    {context}"""
    
    prompt = PromptTemplate(
        input_variables=["question", "context", "topic", "innovation"], 
        template=template
    )
        
    llm = OpenAI(model_name="gpt-4", temperature=1)
        #llm.logit_bias = {"8237":-100, "47553": -100}
    
    return load_qa_chain(llm, verbose=True, chain_type="stuff", prompt=prompt)    

def load_chain():

    #Theme of song should be result of analogy: rocketship:{topic} = {question}:_______
    template = """write a song with folk and country chord progressions, 
    about {question}.
    The first verse should be about the {first_question}. The chorus should repeat and be about the {question}.
    The second verse should be about a specific character or object within the {context}.
    The bridge should be a philosophical statement about {question} in the style of philosophers like Nietzsche, Kant, Locke and Camus.
    The third verse should be about historical event that is related to {question}.
    Write the song for an audience with a grade 15 vocabulary. Use characters from the old west and pbs documentary topics
    as inspiration, symbols and metaphors. 
    {topic}
    
    Randomly include a major 9th chord. Use several homophonic words in response. 
    Don't use rhyming couplets. Create inner rhymes and weak rhymes. Not all lines need to rhyme.
    Results include chord progression and title. Never use "PBS" in the response.

    Additional context: {context}"""

    prompt = PromptTemplate(
        input_variables=["question", "context", "topic", "first_question"], 
        template=template
    )

    llm = OpenAI(model_name="gpt-4", temperature=1, logit_bias={"8237": -100, "47553": -100, "4462": -100})
    #llm.logit_bias = {"8237":-100, "47553": -100}

    return load_qa_chain(llm, verbose=True, chain_type="stuff", prompt=prompt)

context_search = load_retriever()
analogy_chain = load_analogy_chain()
chain = load_chain()
