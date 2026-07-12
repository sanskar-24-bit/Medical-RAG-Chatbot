🩺 Medical RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers medical questions using a knowledge base built from The Gale Encyclopedia of Medicine. The system retrieves relevant context from a local vector store and generates grounded answers using Mistral AI — no hallucinated medical advice, only information found in the source document.

Available as both a Streamlit web app and a CLI chat interface.


📋 Features


✅ Document-grounded answers — the LLM is instructed to answer only from retrieved context, and explicitly says so when the answer isn't found
✅ MMR retrieval — Maximal Marginal Relevance search for diverse, non-redundant context chunks
✅ Persistent vector store — ChromaDB with on-disk persistence, so ingestion only needs to run once
✅ Two interfaces — a polished Streamlit UI and a lightweight terminal chat loop
✅ Transparent retrieval — the Streamlit app lets you expand and inspect the exact context used to generate each answer



🏗️ Architecture

┌─────────────────────┐
│  Gale Encyclopedia   │
│   of Medicine (PDF)  │
└──────────┬───────────┘
           │  PyPDFLoader
           ▼
┌───────────────────────┐
│ RecursiveCharacter     │
│ TextSplitter           │
│ (chunk_size=1000,      │
│  overlap=200)          │
└──────────┬─────────────┘
           │  OpenAI Embeddings
           ▼
┌───────────────────────┐
│      ChromaDB          │
│  (persisted locally)   │
└──────────┬─────────────┘
           │  MMR retriever (k=4, fetch_k=10)
           ▼
┌───────────────────────┐
│   Context + Question   │
│   → Prompt Template     │
└──────────┬─────────────┘
           │
           ▼
┌───────────────────────┐
│  Mistral AI             │
│  (mistral-small-2506)   │
└──────────┬─────────────┘
           │
           ▼
      Grounded Answer

Pipeline stages:


Ingestion (database.py) — loads the source PDF, splits it into overlapping chunks, embeds them with OpenAI embeddings, and persists them to a local ChromaDB store.
Retrieval + Generation (main.py / app.py) — embeds the user's query, retrieves the most relevant and diverse chunks via MMR search, and passes them to Mistral AI along with a strict "answer only from context" system prompt.



🛠️ Tech Stack

ComponentTechnologyOrchestrationLangChainLLMMistral AI (mistral-small-2506)EmbeddingsOpenAI EmbeddingsVector StoreChromaDBDocument LoaderPyPDFLoaderWeb UIStreamlitEnv Managementpython-dotenv


📂 Project Structure

medical-rag-chatbot/
├── database.py          # One-time ingestion script: PDF → chunks → embeddings → ChromaDB
├── main.py               # CLI chatbot loop
├── app.py                 # Streamlit web app
├── document loaders/      # Place source PDF(s) here
│   └── The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf
├── chroma_db/              # Generated vector store (created after running database.py)
├── .env                     # API keys (not committed)
└── README.md


⚙️ Setup & Installation

1. Clone the repository

bashgit clone https://github.com/sanskar-24-bit/medical-rag-chatbot.git
cd medical-rag-chatbot

2. Create a virtual environment

bashpython -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

3. Install dependencies

bashpip install streamlit langchain langchain-openai langchain-huggingface \
            langchain-mistralai langchain-community langchain-text-splitters \
            chromadb pypdf python-dotenv


💡 Tip: run pip freeze > requirements.txt after installation to lock versions for the repo.



4. Configure API keys

Create a .env file in the project root:

envOPENAI_API_KEY=your_openai_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here

5. Add the source document

Place your medical reference PDF inside a folder named document loaders/, matching the path referenced in database.py:

document loaders/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf

6. Build the vector store

Run the ingestion script once to chunk, embed, and persist the document:

bashpython database.py

This creates a chroma_db/ folder containing the persisted vector store.


🚀 Usage

Streamlit web app

bashstreamlit run app.py

Then open the local URL shown in your terminal, type a medical question, and click Get Answer. Expand Retrieved Context to see exactly which chunks informed the response.

CLI chatbot

bashpython main.py

Type your question at the You: prompt. Type exit to quit.


💬 Example

You: What are the symptoms of Diabetes?

AI: [Answer generated from the retrieved chunks of the Gale Encyclopedia of Medicine]

If the answer isn't present in the ingested document, the assistant responds:


"I could not find the answer in the document."
