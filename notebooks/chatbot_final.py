from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import TokenTextSplitter,RecursiveCharacterTextSplitter
import pandas as pd
import csv
from langchain_community.llms import Ollama
import os
import pickle
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from joblib import dump, load


def load_and_chunk_documents(csv_file, embedding_file, chunks_file, faiss_index_file):
    """
    Load documents from CSV file, chunk them, and create FAISS index.

    Args:
    - csv_file (str): Path to the CSV file containing documents.
    - embedding_file (str): Path to save/load embeddings
    - chunks_file (str): Path to save/load chunks
    - faiss_index_file (str): Path to save/load FAISS index

    Returns:
    - FAISS: FAISS index containing chunked documents.
    """
    print(f"📄 Processing CSV file: {csv_file}")

    # Check if files exist
    embeddings_saved = os.path.exists(embedding_file)
    chunks_saved = os.path.exists(chunks_file)
    faiss_saved = os.path.exists(faiss_index_file)

    # Load or create embeddings
    if embeddings_saved:
        print(f"📦 Loading embeddings from: {embedding_file}")
        embeddings = load(embedding_file)
    else:
        print("🔄 Creating new embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        dump(embeddings, embedding_file)
        print(f"💾 Embeddings saved to: {embedding_file}")

    # Load or create chunks
    if chunks_saved:
        print(f"📦 Loading chunks from: {chunks_file}")
        chunks = load(chunks_file)
    else:
        print("🔄 Loading and chunking documents...")
        loader = CSVLoader(file_path=csv_file)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=50)
        chunks = text_splitter.split_documents(data)
        dump(chunks, chunks_file)
        print(f"💾 Chunks saved to: {chunks_file}")

    # Load or create FAISS index
    if faiss_saved:
        print(f"📦 Loading FAISS index from: {faiss_index_file}")
        db = FAISS.load_local(faiss_index_file, embeddings=embeddings, allow_dangerous_deserialization=True)
    else:
        print("🔄 Building FAISS index...")
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(faiss_index_file)
        print(f"💾 FAISS index saved to: {faiss_index_file}")

    return db

def initialize_chatbot():
    """
    Initialize the Ollama model and load and chunk documents.

    Returns:
    - tuple: Tuple containing Ollama model instance and FAISS index.
    """
    # File paths - adjust these to match your backend structure
    csv_file = '../backend/data/llama3-medical.csv'
    embedding_file = '../backend/data/llama3-medical-embedding.pkl'
    chunks_file = '../backend/data/llama3-medical-chunks.pkl'
    faiss_index_file = '../backend/data/llama3-medical-index.faiss'

    print("🤖 Initializing HealthWave Chatbot...")
    print("=" * 50)

    # Check if CSV file exists
    if not os.path.exists(csv_file):
        print(f"❌ Error: CSV file not found at {csv_file}")
        print("Please ensure the medical data CSV file exists.")
        return None, None, None

    # Initialize Ollama model
    print("🦙 Connecting to Ollama (llama3)...")
    try:
        llm = Ollama(model="llama3")
        # Test the connection
        test_response = llm.invoke("Hello")
        print("✅ Ollama connection successful!")
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        print("Please ensure Ollama is running and llama3 model is available.")
        return None, None, None

    # Load/create FAISS database
    db = load_and_chunk_documents(csv_file, embedding_file, chunks_file, faiss_index_file)
    retriever = db.as_retriever()

    print("✅ Chatbot initialization complete!")
    print("=" * 50)

    return llm, retriever, db

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_chatbot_answer(llm, retriever, question, chat_history, rag_chain=None):
    """
    Get answer from chatbot with context-aware retrieval
    """
    try:
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        qa_system_prompt = """You are a clinical bot that explains medical jargons in simple words.
        Answer the given questions based on your knowledge and given context.
        {context}
        You are allowed to rephrase the answer based on the context.
        Explain it so that the normal person can understand it."""

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
        chat_history.extend([HumanMessage(content=question), ai_msg_1["answer"]])
        return ai_msg_1["answer"]

    except Exception as e:
        print(f"Error in chatbot response: {e}")
        return "I apologize, but I encountered an error processing your question. Please try again."



def main():
    """Main function to run the chatbot setup"""
    print("🏥 HealthWave Medical Chatbot Setup")
    print("This script will create the RAG system files needed for your app.")
    print("=" * 60)

    llm, retriever, db = initialize_chatbot()

    if llm is None:
        print("❌ Chatbot initialization failed. Please check the errors above.")
        return

    print("🎉 SUCCESS! RAG system files have been created.")
    print("\n📁 Generated files:")
    print("   • llama3-medical-embedding.pkl")
    print("   • llama3-medical-chunks.pkl")
    print("   • llama3-medical-index.faiss/")
    print("\n✅ Your HealthWave app is now ready to use!")
    print("\n💡 You can now:")
    print("   1. Start your backend server")
    print("   2. Start your frontend server")
    print("   3. Use all 4 modules including the Medical ChatBot")

    # Optional: Test the chatbot
    test_mode = input("\n🔍 Would you like to test the chatbot? (y/n): ").lower().strip()

    if test_mode == 'y':
        print("\n🚀 Starting interactive test mode...")
        print("Type 'exit' to quit the test.\n")

        chat_history = []

        while True:
            user_input = input("💬 You: ")

            if user_input.lower() == 'exit':
                print("👋 Exiting test mode...")
                break

            try:
                response = get_chatbot_answer(llm, retriever, user_input, chat_history, None)
                print(f"🤖 Bot: {response}\n")
            except Exception as e:
                print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    main()