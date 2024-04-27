import streamlit as st
import numpy as np
import pandas as pd
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from datetime import datetime

st.set_page_config(layout="wide")
base_path = os.path.abspath(os.path.dirname("app.py"))

def load_model():
    savedModelpath = os.path.join(base_path,"savedmodels/")
    format = "%Y-%m-%d-%H-%M-%S"
    model_name = savedModelpath + max([datetime.strptime(d,format) for d in os.listdir(savedModelpath)]).strftime(format)
    # st.toast("Latest Model (FAISS Index, huggingface) loaded successfully! ")
    # st.toast(model_name)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    model = FAISS.load_local(model_name, embeddings)
    r = model.similarity_search_with_score("Something")
    rDf = pd.read_csv(r[0][0].metadata["source"])
    return model,rDf

st.markdown(""" <style> div.stButton > button {
  background-color: #04AA6D; 
  border-radius: 8px;
  color: white;
  padding: 16px 32px;
  text-align: center;
  text-decoration: none;
  display: inline-block;
  font-size: 16px;
  margin: 4px 2px;
  transition-duration: 0.4s;
  cursor: pointer;
  background-color: green; 
  color: white; 
  border: 2px solid #04AA6D;
}

button:hover {
  background-color: white;
  color: black;
  box-shadow: 0 12px 16px 0 rgba(0,0,0,0.24),0 17px 50px 0 rgba(0,0,0,0.19);
}
            </style>""",unsafe_allow_html=True)

st.title("Search for Similar Tickets")
st.write("(Searching for issues only in the CR Project)")
st.markdown('<br><style>body{border-color: black; border-style: solid;}</style>',unsafe_allow_html=True)
ocol1,ocol2 = st.columns([10,1])
with ocol1:
    with st.form("InputForm"):
        st.markdown("""<style> [data-testid="stForm"] {padding: 20px; border: 3px solid black; border-radius: 50px; margin-top: 20px; border-style: solid;
            box-shadow: 5px 5px 15px rgba(0,0,0,0.1);}</style>""",unsafe_allow_html=True)
        col1, col2,_= st.columns([2,8,2])
        with col1:
            st.write("<br><h6>Enter Summary:</h6>",unsafe_allow_html=True)
        with col2:
            test_query = st.text_area(value="",label="")
        col1, col2,_= st.columns([2,8,2])
        with col2:
            with st.spinner("Loading model"):
                if 'gt' not in st.session_state:
                    gtPath = os.path.join(base_path,"datainput","jira_links.csv")
                    gt = pd.read_csv(gtPath)
                    st.session_state['gt'] = gt
                if 'model' not in st.session_state:
                    st.session_state["model"], st.session_state["source"] = load_model()
                else:
                    del st.session_state["model"]
                    st.session_state["model"], st.session_state["source"] = load_model()
                submit_button = st.form_submit_button(label="Search")

def model_predict( test_query):
    result_docs = st.session_state["model"].similarity_search_with_score(test_query,k=10)
    rows, scores = [], []
    for doc in result_docs:
        rows.append(doc[0].metadata["row"])
        scores.append(round(doc[1],2))
    recomsDf = st.session_state["source"].iloc[rows,:]
    recomsDf["score"] = scores
    if 'Description' in recomsDf.columns:
        recomsDf.drop('Description',axis=1,inplace=True)
    return recomsDf

def linkable(cr):
    link_cr = f"{cr}"
    return f"<a target='_blank' href='{link_cr}'>{cr}</a>"

if submit_button:
    gt = st.session_state["gt"]
    gt_out = gt[gt.iloc[:,1]==test_query]
    if len(gt_out) != 0:
        st.markdown("Ground Truth Data available")
        st.write(gt_out.iloc[:5,:])
    fullData = model_predict(test_query)
    fullData.iloc[:,0] = fullData.iloc[:,0].apply(linkable)
    result_df = fullData.style.set_properties(**{'text-align': 'left','selector':'tr-nth-of-type(odd)','props':'background-color: #f2f2f2; color: black;','selector':'td','props':'border: 4px solid black;'}).hide()
    st.header("Query Results:")
    
    st.markdown(fullData.to_html(escape=False, index=False), unsafe_allow_html=True)
    st.markdown("""<br><style> table { border: 3px solid black; border-collapse: collapse;} th {background-color: #3498db; padding: 10px; border: 1px solid black; border-collapse: collapse;} </style>""",unsafe_allow_html=True)
    st.markdown("<style>tr:nth-child(even) {background-color: #f2f2f2;}</style>",unsafe_allow_html=True)