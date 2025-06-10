import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from groq import Groq
import json

# Page Configuration 
st.set_page_config(
    page_title="Semantic Quote Retrieval",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_resources():
    print("Loading resources...")
    retriever_model = SentenceTransformer('./fine_tuned_quote_model')

    index = faiss.read_index('quotes.index')
    
    df = pd.read_csv('processed_quotes.csv').fillna('')
    corpus = df['combined_text'].tolist()
    llm_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    
    print("Resources loaded successfully.")
    return retriever_model, index, df, corpus, llm_client

def get_rag_response(query: str, k: int = 5):

    query_embedding = retriever_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    
    retrieved_contexts_for_llm = [corpus[i] for i in indices[0]]
    retrieved_data_for_display = [df.iloc[i].to_dict() for i in indices[0]]
    context_str = "\n\n".join(retrieved_contexts_for_llm)

    try:
        chat_completion = llm_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful quote assistant. Use the provided context to answer the user's query. Provide a concise summary and then list the most relevant quotes from the context."
                },
                {
                    "role": "user",
                    "content": f"CONTEXT:\n{context_str}\n\nQUERY:\n{query}",
                }
            ],
            model="llama3-8b-8192",
            temperature=0.7,
            max_tokens=350,
        )
        generated_answer = chat_completion.choices[0].message.content
        return generated_answer, retrieved_data_for_display

    except Exception as e:
        st.error(f"An error occurred with the LLM API: {e}")
        return "Error: Could not get a response from the LLM.", []


retriever_model, index, df, corpus, llm_client = load_resources()

st.sidebar.title("About")
st.sidebar.info(
    "This is a RAG-based Semantic Quote Retrieval system. "
    "It uses a fine-tuned sentence transformer for retrieval and a Large Language Model (Llama 3 via Groq) for generation. "
    "Enter a query in the main panel to get started."
)

st.sidebar.title("Dataset Visualizations")

if st.sidebar.checkbox("Show Top 20 Authors"):
    st.sidebar.subheader("Top Authors by Quote Count")
    author_counts = df['author'].value_counts().nlargest(20)
    st.sidebar.bar_chart(author_counts)

if st.sidebar.checkbox("Show Top 20 Tags"):
    st.sidebar.subheader("Top Tags by Frequency")

    try:
        tags_exploded = df['tags'].apply(eval).explode()
        tag_counts = tags_exploded.value_counts().nlargest(20)
        st.sidebar.bar_chart(tag_counts)
    except Exception as e:
        st.sidebar.error(f"Could not process tags: {e}")


st.title("📚 RAG-Based Semantic Quote Retrieval")
st.write("Enter a natural language query to find relevant quotes. For example: 'What are some quotes about life by Oscar Wilde?'")

query = st.text_input("Enter your query:", key="query_input")

if st.button("Search", key="search_button"):
    if query:
        with st.spinner("Retrieving relevant quotes and generating an answer..."):
          
            generated_answer, retrieved_data = get_rag_response(query)
            st.subheader("💡 Generated Answer")
            st.markdown(generated_answer)
            
            st.divider()
            
            st.subheader("🔍 Retrieved Source Quotes")
            
            if retrieved_data:
                download_data = {
                    "query": query,
                    "generated_answer": generated_answer,
                    "retrieved_quotes": retrieved_data
                }
                
               
                st.download_button(
                    label="Download Results as JSON",
                    data=json.dumps(download_data, indent=4),
                    file_name=f"quote_results_{query[:20].replace(' ', '_')}.json",
                    mime="application/json",
                )
                
                for item in retrieved_data:
                    with st.container(border=True):
                        st.markdown(f"> ### *“{item['quote']}”*")
                        st.caption(f"— {item['author']} | Tags: `{item['tags']}`")
            else:
                st.warning("No relevant quotes were retrieved.")

    else:
        st.warning("Please enter a query.")