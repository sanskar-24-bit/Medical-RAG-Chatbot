from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma

load_dotenv()

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
    }
)

llm = ChatMistralAI(model = "mistral-small-2506")                                                                                                                    # type: ignore

prompt = ChatPromptTemplate.from_messages([
      (
        "system",
        """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say:
"I could not find the answer in the document."
"""
    ),
    (
        "human",
        """Context:
{context}

Question:
{question}
"""
    )
])

print("RAG system is created")
print("Type 'exit' to quit.\n")

while True:
      query = input("You: ")

      if query.lower() == "exit":
        break

      docs = retriever.invoke(query)

      if not docs:
        print("No relevant documents found.\n")
        continue

      context = "\n\n".join(
        doc.page_content for doc in docs)
    

      final_prompt = prompt.invoke({
        "context": context,
        "question": query})

      response = llm.invoke(final_prompt)

      print("\nAI:", response.content)
      print("-" * 60)
      
      