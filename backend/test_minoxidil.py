#!/usr/bin/env python3
"""Test script to diagnose the minoxidil response issue"""

def test_minoxidil_responses():
    print("=== Testing Minoxidil Responses ===")

    from utils import variables

    # Test 1: T5 Model (used by /simplify_data endpoint)
    print("\n1. Testing T5 Model (anishbasnet/t5-base-ft-medical-simplifier):")
    try:
        from app.model import Model
        model = Model(variables["t5_model_name"])
        t5_response = model.generate_response("minoxidil")
        print(f"T5 Response: {t5_response}")
    except Exception as e:
        print(f"Error with T5 model: {e}")

    # Test 2: RAG System with LLaMA (used by /simplify_text_llm endpoint)
    print("\n2. Testing RAG System (LLaMA 3.2 + Medical Database):")
    try:
        from app.model import Chatbot
        chatbot = Chatbot(
            model_name=variables["model_name"],
            csv_file=variables["model_embedding_csv_file"],
            embedding_file=variables["model_embedding_file"],
            chunks_file=variables["model_chunks_file"],
            fass_index_file=variables["model_faiss_index_file"]
        )
        rag_response = chatbot.get_chatbot_answer("minoxidil")
        print(f"RAG Response: {rag_response}")
    except Exception as e:
        print(f"Error with RAG system: {e}")

    # Test 3: Check if minoxidil is in the medical database
    print("\n3. Searching for 'minoxidil' in medical database:")
    try:
        import pandas as pd
        df = pd.read_csv(variables["model_embedding_csv_file"])
        minoxidil_entries = df[df['page_text'].str.contains('minoxidil', case=False, na=False)]
        print(f"Found {len(minoxidil_entries)} entries containing 'minoxidil' in the database")

        if len(minoxidil_entries) > 0:
            print("Sample entries:")
            for idx, row in minoxidil_entries.head(2).iterrows():
                print(f"  Title: {row['page_title']}")
                print(f"  Text excerpt: {row['page_text'][:200]}...")
                print()
    except Exception as e:
        print(f"Error searching database: {e}")

    # Test 4: Test alternative spellings
    print("\n4. Testing alternative spellings:")
    test_terms = ["minoxidil", "menoxidil", "Minoxidil", "MINOXIDIL"]

    for term in test_terms:
        try:
            response = model.generate_response(term)
            print(f"'{term}' -> {response}")
        except:
            print(f"'{term}' -> Error")

if __name__ == "__main__":
    test_minoxidil_responses()