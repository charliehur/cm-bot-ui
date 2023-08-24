import streamlit as st
from streamlit_chat import message

st.set_page_config(page_title="CM Data Platform", page_icon="🤖")

from models import chain, context_search

st.header("CM Data Platform Bot")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("Enter a topic for your song", placeholder="topic", key="input")
    return input_text


user_input = get_text()

if user_input:

    context_docs = context_search.get_relevant_documents(user_input)
    
    output = chain.run(input_documents=context_docs, question=user_input)

    
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for ind, output in enumerate(st.session_state["generated"]):
        message(st.session_state["past"][ind], is_user=True, avatar_style="bottts-neutral", seed="Mittens", key=str(ind) + "_user")
        #message(st.session_state["past"][ind], is_user=True, logo="UCBLogo_Tsprnt.png", key=str(ind) + "_user")
        message(output, avatar_style="bottts-neutral", seed="Bear", key=str(ind))
        #message(output, logo="UCBLogo_Tsprnt.png", key=str(ind))

