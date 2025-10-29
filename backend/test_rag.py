#!/usr/bin/env python3
"""Test script to check if RAG system components exist and work"""

import os
from joblib import load

def check_rag_components():
    print("=== RAG System Component Check ===")

    # Check configuration
    try:
        from utils import variables
        print("✓ Configuration loaded successfully")
        print(f"  LLM model: {variables['model_name']}")
        print(f"  CSV file: {variables['model_embedding_csv_file']}")
        print(f"  Embedding file: {variables['model_embedding_file']}")
        print(f"  Chunks file: {variables['model_chunks_file']}")
        print(f"  FAISS index: {variables['model_faiss_index_file']}")
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        return False

    print("\n=== File Existence Check ===")

    # Check if files exist
    files_to_check = [
        variables['model_embedding_csv_file'],
        variables['model_embedding_file'],
        variables['model_chunks_file'],
        variables['model_faiss_index_file']
    ]

    all_files_exist = True
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")
            all_files_exist = False

    print("\n=== Component Loading Check ===")

    # Check if pickle files can be loaded
    pickle_files = [variables['model_embedding_file'], variables['model_chunks_file']]
    for file_path in pickle_files:
        if os.path.exists(file_path):
            try:
                data = load(file_path)
                print(f"✓ Successfully loaded {file_path}")
                print(f"  Type: {type(data)}")
                if hasattr(data, '__len__'):
                    print(f"  Length: {len(data)}")
            except Exception as e:
                print(f"✗ Error loading {file_path}: {e}")
                all_files_exist = False

    # Check FAISS index
    faiss_path = variables['model_faiss_index_file']
    if os.path.exists(faiss_path):
        faiss_files = os.listdir(faiss_path)
        print(f"✓ FAISS directory exists with files: {faiss_files}")
    else:
        print(f"✗ FAISS directory {faiss_path} missing")
        all_files_exist = False

    print("\n=== Chatbot Initialization Check ===")

    # Try to initialize chatbot
    try:
        from app.model import Chatbot
        chatbot = Chatbot(
            model_name=variables['model_name'],
            csv_file=variables['model_embedding_csv_file'],
            embedding_file=variables['model_embedding_file'],
            chunks_file=variables['model_chunks_file'],
            fass_index_file=variables['model_faiss_index_file']
        )
        print("✓ Chatbot initialized successfully")
        print(f"  Embeddings saved: {chatbot.embeddings_saved}")
        print(f"  Chunks saved: {chatbot.chunks_saved}")
        print(f"  FAISS index saved: {chatbot.faiss_index_saved}")

        # Try to access database
        print("  Testing database access...")
        db = chatbot.db
        print("✓ RAG database loaded successfully!")

    except Exception as e:
        print(f"✗ Error initializing chatbot: {e}")
        all_files_exist = False

    print("\n=== Summary ===")
    if all_files_exist:
        print("🎉 RAG system is fully operational!")
        return True
    else:
        print("⚠️  RAG system has missing or corrupted components")
        return False

if __name__ == "__main__":
    check_rag_components()