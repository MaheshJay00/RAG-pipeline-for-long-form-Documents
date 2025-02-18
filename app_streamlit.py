import streamlit as st
import requests

API_URL = "UR_URL"

st.set_page_config(page_title="RAG-powered business docs Q&A",layout="wide")

st.title("AI-powered document Q&A")
st.write("Ask questions about business invoice.")
query=st.text_input("Enter your question:",placeholder="What is the invoice id?")

top_k = st.slider("Number of relevant documents to reqtrieve:",1,10,5)

if st.button("Get Answer"):
    if query.strip():
        with st.spinner("🔍 Searching relevant documents & generating answer..."):
            # API Request
            response = requests.post(API_URL, json={"query": query, "top_k": top_k})
            
            if response.status_code == 200:
                data = response.json()
                st.success("✅ Answer Generated!")
                st.markdown(f"**📖 Query:** {data['query']}")
                st.markdown(f"**📝 Answer:** {data['response']}")
            else:
                st.error("❌ Error fetching response. Please check the API.")
    else:
        st.warning("⚠️ Please enter a valid question.")

