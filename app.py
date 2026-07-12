import streamlit as st
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma

st.set_page_config(
    page_title = "Medical RAG Chatbot",
    page_icon = "🩺",
    layout = "wide"
)

load_dotenv()

@st.cache_resource
def load_models():

      embedding_model = OpenAIEmbeddings()

      vectorstore = Chroma(
        persist_directory = "chroma_db",
        embedding_function = embedding_model)

      retriever = vectorstore.as_retriever(
        search_type = "mmr",
        search_kwargs = {
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        })

      llm = ChatMistralAI(model = "mistral-small-2506")                                                                                                             # type: ignore

      prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a helpful Medical AI Assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
reply:

"I could not find the answer in the document."
"""
        ),
        (
            "human",
            """
Context:
{context}

Question:
{question}
"""
        )
    ])

      return retriever, llm, prompt


retriever, llm, prompt = load_models()

with st.sidebar:
      st.title("🩺 Medical RAG")
      st.image(
        "https://img.icons8.com/color/480/stethoscope.png",
        width = 150
    )

      st.markdown("---")

      st.header("📋 Features")

      st.write("✅ Medical Question Answering")
      st.write("✅ ChromaDB Retrieval")
      st.write("✅ OpenAI Embeddings")
      st.write("✅ Mistral AI")
      st.write("✅ LangChain")

      st.markdown("---")

      st.info("💡 Ask questions related to your uploaded medical documents.")

st.title("🩺 Medical AI Assistant")
st.subheader("💊 Retrieval Augmented Generation (RAG)  Chatbot")

st.image(
    "https://images.unsplash.com/photo-1584982751601-97dcc096659c",
    use_container_width = True
)

st.success("📚 Your medical knowledge base is ready!")

query = st.text_input(
    "🔍 Ask your medical question",
    placeholder="Example: What are the symptoms of Diabetes?"
)

if st.button("🚀 Get Answer"):

    if query.strip() == "":
        st.warning("⚠️ Please enter a question.")

    else:

        with st.spinner("🔍 Searching documents..."):

            docs = retriever.invoke(query)

            context = "\n\n".join(
                doc.page_content for doc in docs
            )

            final_prompt = prompt.invoke({
                "context": context,
                "question": query
            })

            response = llm.invoke(final_prompt)

        st.divider()

        st.header("🤖 AI Response")

        st.write(response.content)

        st.divider()

        with st.expander("📄 Retrieved Context"):
            st.write(context)

st.divider()

st.caption("❤️ Built using Streamlit • LangChain • ChromaDB • Mistral AI")
