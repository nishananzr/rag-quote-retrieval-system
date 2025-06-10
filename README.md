# RAG-Based Semantic Quote Retrieval & Structured QA

This project is a complete end-to-end implementation of a Retrieval-Augmented Generation (RAG) system for semantic quote retrieval. The system is built on a fine-tuned sentence-transformer model, uses a robust RAG pipeline with a vector database, and is deployed as an interactive web application using Streamlit.

[![Streamlit App](https://rag-quote-retrieval-system-hkt8j9zmasqzv9fpvusor6.streamlit.app/)](https://rag-quote-retrieval-system-hkt8j9zmasqzv9fpvusor6.streamlit.app/)

**Live Application URL:** [https://rag-quote-retrieval-system-hkt8j9zmasqzv9fpvusor6.streamlit.app/](https://rag-quote-retrieval-system-hkt8j9zmasqzv9fpvusor6.streamlit.app/)

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [System Architecture](#system-architecture)
4. [Methodology & Design Decisions](#methodology--design-decisions)
    - [Data Preparation & EDA](#1-data-preparation--exploratory-data-analysis)
    - [Model Fine-Tuning](#2-model-fine-tuning)
    - [RAG Pipeline Implementation](#3-rag-pipeline-implementation)
    - [RAG Evaluation](#4-rag-evaluation)
5. [Evaluation Results & Discussion](#evaluation-results--discussion)
6. [Challenges Faced & Debugging Journey](#challenges-faced--debugging-journey)
7. [Setup & How to Run](#setup--how-to-run)
8. [Deliverables](#deliverables)

---

## Project Overview

The objective of this project was to build a system that can understand natural language queries and retrieve relevant quotes from the **Abirate/english_quotes:** [https://huggingface.co/datasets/Abirate/english_quotes](https://huggingface.co/datasets/Abirate/english_quotes)dataset. The workflow involved the entire lifecycle of a modern NLP application: in-depth data analysis and cleaning, fine-tuning a sentence embedding model, building and evaluating a RAG pipeline, and deploying a user-friendly web application.

## Features

- **Semantic Search:** Understands the user's intent beyond keywords (e.g., "humorous quotes by Irish authors").
- **Structured JSON Output:** The system can provide structured responses containing quotes, authors, and tags.
- **Interactive UI:** A user-friendly Streamlit application for easy interaction.
- **Data Visualizations:** The app includes visualizations for author and tag distributions in the dataset.
- **Downloadable Results:** Users can download the retrieved results as a JSON file.

## System Architecture

The application follows a classic Retrieval-Augmented Generation (RAG) architecture:

1.  **User Query:** The user enters a natural language query into the Streamlit UI.
2.  **Encoding:** The query is converted into a vector embedding by the **fine-tuned `all-MiniLM-L6-v2` model**.
3.  **Retrieval:** The query vector is used to perform a similarity search against a **FAISS vector index** containing the pre-encoded quote documents. The top-k most relevant documents are retrieved.
4.  **Augmentation:** The retrieved documents (context) are combined with the original user query into a detailed prompt.
5.  **Generation:** The augmented prompt is sent to a Large Language Model (**Llama 3 via the Groq API**), which generates a coherent, context-aware answer.
6.  **Response:** The final answer is displayed to the user in the Streamlit app.

 

---

## Methodology & Design Decisions

### 1. Data Preparation & Exploratory Data Analysis

The quality of the RAG system is highly dependent on the quality of the data it retrieves from. Therefore, significant effort was dedicated to cleaning and normalizing the `Abirate/english_quotes` dataset.

- **Duplicate Quotes:** I discovered instances of the same quote attributed to different authors. To ensure index integrity, I made the design decision to remove duplicate quotes by keeping only the first occurrence (`df.drop_duplicates(subset=['quote'])`), creating a clean and unique set of documents.
- **Author Name Normalization:** The analysis revealed inconsistencies like "Oscar Wilde" and "Oscar Wilde,". Such issues degrade retrieval and analysis. I normalized the author names by stripping trailing punctuation and whitespace, consolidating entries for the same author.
- **Tag Normalization:** The tags were highly fragmented (e.g., 'inspiration', 'inspirational', 'inspirational-quotes'). A cleaning function was implemented to:
    1.  Convert all tags to lowercase.
    2.  Consolidate common variants into a single canonical tag (e.g., all variants of "inspiration" were mapped to `inspiration`).
    3.  Replace hyphens with spaces for more natural tokenization.
    4.  Ensured uniqueness using Python sets.
- **Combined Text Field:** For effective semantic retrieval, I created a `combined_text` field for each entry by concatenating the quote, author, and normalized tags. This creates a rich document for the retriever model to encode, allowing it to match queries based on any of these three attributes.

### 2. Model Fine-Tuning

To improve the retriever's ability to understand the specific semantics of the quotes dataset, a pre-trained model was fine-tuned.

-   **Base Model:** `all-MiniLM-L6-v2`. This model was chosen for its excellent balance of performance, speed, and small size, making it ideal for this application.
-   **Task & Loss Function:** The fine-tuning was framed as a contrastive learning task using **`MultipleNegativesRankingLoss`**. For each data point, the model was taught to pull the vector for the `quote` closer to the vector for its corresponding `combined_text` (quote + author + tags), while pushing it away from all other examples in the batch. This directly optimizes the model for the retrieval task.
-   **Outcome:** The fine-tuned model produces embeddings that are highly specialized for this dataset, leading to more relevant retrieval results compared to the generic base model.

### 3. RAG Pipeline Implementation

-   **Retriever:** The fine-tuned model was used to encode all 2,500+ documents into vectors. These vectors were indexed using **FAISS (`IndexFlatL2`)**, a library for efficient similarity search.
-   **Generator:** A Large Language Model generates the final human-readable answer. After encountering significant issues with other APIs (see [Challenges Faced](#challenges-faced--debugging-journey)), I chose **Llama 3 (via the Groq API)** for its high speed, excellent instruction-following capabilities, and reliable free tier.

### 4. RAG Evaluation

The performance of the entire pipeline was quantitatively measured using the **RAGAS** framework. A small, high-quality evaluation dataset was manually created with questions, ideal `ground_truth` answers, and the expected contexts.

-   **Metrics Used:**
    -   `faithfulness`: Measures if the answer is factually grounded in the provided context.
    -   `answer_relevancy`: Measures if the answer is relevant to the user's query.
    -   `context_precision`: Measures if the retrieved context is relevant (low noise).
    -   `context_recall`: Measures if all necessary context was retrieved.
-   **Configuration:** To ensure a fair evaluation, the RAGAS framework was configured to use a `ChatGroq` wrapper, using Llama 3 as the judge LLM. For embedding-based metrics, a local `HuggingFaceBgeEmbeddings` model was used to avoid reliance on external paid APIs.

---

## Evaluation Results & Discussion

The RAGAS evaluation provided the following scores:

| Metric              | Score  |
| ------------------- | ------ |
| **`answer_relevancy`**  | 0.8655 |
| **`context_recall`**    | 0.3333 |
| **`context_precision`** | 0.2500 |
| **`faithfulness`**      | 0.2000 |

-   **Key Insight:** The system is strong in **Generation** but weak in **Retrieval**. The high `answer_relevancy` score (0.87) shows that the LLM is excellent at understanding the query and formulating a relevant answer *if given the right context*.
-   **Area for Improvement:** The low `context_recall` (0.33) and `context_precision` (0.25) scores clearly indicate that the retriever (our fine-tuned model + FAISS) is the main bottleneck. It struggles to find all the correct documents while also pulling in irrelevant noise.
-   **Next Steps:** To improve the system, the primary focus should be on enhancing the retriever, potentially through more advanced fine-tuning techniques or by implementing a hybrid search system that combines semantic search with traditional keyword search.

---

## Challenges Faced & Debugging Journey

The development process was not without its challenges. The journey to a working LLM endpoint demonstrates a key real-world engineering skill: **adaptability and systematic problem-solving.**

1.  **Initial Plan (OpenAI):** The first attempt used the OpenAI API, which failed due to expired free trial credits (`RateLimitError`).
2.  **Pivot 1 (Hugging Face API):** I then pivoted to the free Hugging Face Inference API. This led to a persistent and difficult-to-diagnose `StopIteration` error. Through systematic isolation using dedicated test scripts (`test_client.py`, `test_requests.py`), I discovered the root cause: the Hugging Face API was returning `404 Not Found` errors for several models, including `gpt2`. This indicated a service-level issue, not a code issue.
3.  **Final Solution (Groq API):** With the HF API being unreliable, I made a final pivot to the **Groq API**. This service provides free, high-speed access to Llama 3 and uses an OpenAI-compatible SDK. This switch was successful immediately and proved to be a robust and highly performant solution.

This iterative debugging process was invaluable and highlights the importance of not being locked into a single provider and being able to adapt to technical roadblocks.

---

## Setup & How to Run

Follow these steps to run the application locally.

**1. Clone the Repository:**
```bash
git clone https://github.com/nishananzr/rag-quote-retrieval-system.git
cd rag-quote-retrieval-system
```

**2. Create and Activate a Virtual Environment:**
```bash
# Create the environment
python -m venv rag_env

# Activate it (on Windows)
.\rag_env\Scripts\activate
```

**3. Install Dependencies:**
All required packages are listed in `requirements.txt`.
```bash
python -m pip install -r requirements.txt
```

**4. Set Up API Key:**
Create a `.streamlit` folder and a `secrets.toml` file inside it.
```
mkdir .streamlit
```
Add your Groq API key to `.streamlit/secrets.toml`:
```toml
# .streamlit/secrets.toml
GROQ_API_KEY = "gsk_YOUR_API_KEY_HERE"
```

**5. Run the Streamlit App:**
```bash
python -m streamlit run app.py
```
The application should open automatically in your web browser at `http://localhost:8501`.

---

## Deliverables
- **Notebooks/Scripts:**
    - `dataprep_and_finetune.ipynb`: Covers data cleaning, EDA, and model fine-tuning.
    - `app.py`: The final Streamlit application.
- **Evaluation Results:** A detailed discussion is provided in this README.
