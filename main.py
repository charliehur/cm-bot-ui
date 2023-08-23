import streamlit as st
from streamlit_chat import message

st.set_page_config(page_title="ucb songwriting bot", page_icon="🤖")

from models import chain, context_search, get_topic, get_innovation, get_story_structure, analogy_chain

st.header("Ugly Cousin Bot")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("Enter a topic for your song", placeholder="topic", key="input")
    return input_text


user_input = get_text()

if user_input:
    topic = get_topic()
    innovation = get_innovation()
    story_structure = get_story_structure()
    context_docs = context_search.get_relevant_documents(user_input)
    firstoutput = analogy_chain.run(question=user_input, input_documents="", topic=topic, innovation=innovation)
    #output = chain.run(input_documents=context, question=user_input, topic=topic)
    output = chain.run(question=firstoutput, input_documents=context_docs, topic=topic, first_question=user_input)


    #str1= "innovation: " + innovation
    #st.write(str1)
    #str2 = "topic: " + topic
    #st.write(str2)
    #str3 = "first output: " + firstoutput
    #st.write(str3)
    #st.write(context_docs)
    #str5 = "initial prompt: " + user_input
    #st.write(str5)
    
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for ind, output in enumerate(st.session_state["generated"]):
        message(st.session_state["past"][ind], is_user=True, avatar_style="bottts-neutral", seed="Mittens", key=str(ind) + "_user")
        #message(st.session_state["past"][ind], is_user=True, logo="UCBLogo_Tsprnt.png", key=str(ind) + "_user")
        message(output, avatar_style="bottts-neutral", seed="Bear", key=str(ind))
        #message(output, logo="UCBLogo_Tsprnt.png", key=str(ind))

