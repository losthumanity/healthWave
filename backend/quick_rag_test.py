#!/usr/bin/env python3
"""
Simple RAG Inference Tester for MediClare
Quick script to test RAG system inference with medical terms
"""

import os
import sys
from pathlib import Path
import time

# Import the RAG system components
try:
    from app.model import Chatbot
    from utils import variables
    print("✅ Successfully imported RAG components")
except ImportError as e:
    print(f"❌ Error importing components: {e}")
    sys.exit(1)

class SimpleRAGTester:
    """Simple RAG inference tester"""

    def __init__(self):
        """Initialize the RAG system"""
        print("🏥 Initializing MediClare RAG System...")

        try:
            self.chatbot = Chatbot(
                model_name=variables["model_name"],
                csv_file=variables["model_embedding_csv_file"],
                embedding_file=variables["model_embedding_file"],
                chunks_file=variables["model_chunks_file"],
                fass_index_file=variables["model_faiss_index_file"],
            )
            print("✅ RAG system loaded successfully!")

            # Test if vector database is working
            print("🔍 Testing vector database connectivity...")
            db = self.chatbot.db
            print(f"✅ Vector database connected with {db.index.ntotal} vectors")

        except Exception as e:
            print(f"❌ Error initializing RAG system: {e}")
            raise e

    def test_medical_queries(self):
        """Test with predefined medical queries"""
        test_queries = [
            "What is hypertension?",
            "Explain diabetes",
            "What does pneumonia mean?",
            "Tell me about asthma",
            "What is myocardial infarction?",
            "Explain bradycardia",
            "What is chronic kidney disease?",
            "Tell me about migraine"
        ]

        print("\n🧪 Testing Medical Queries:")
        print("=" * 50)

        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Testing: '{query}'")
            print("-" * 30)

            start_time = time.time()
            try:
                response = self.chatbot.get_chatbot_answer(query)
                response_time = time.time() - start_time

                print(f"✅ Response ({response_time:.2f}s):")
                print(f"📝 {response}")
                print(f"📊 Length: {len(response)} characters")

                # Quick quality check
                if len(response) > 50 and query.split()[-1].rstrip('?').lower() in response.lower():
                    print("✅ Quality: Good (contains term, adequate length)")
                else:
                    print("⚠️  Quality: Needs review")

            except Exception as e:
                print(f"❌ Error: {e}")

    def test_vector_retrieval(self):
        """Test vector similarity search directly"""
        test_terms = ["diabetes", "heart attack", "pneumonia", "hypertension", "asthma"]

        print("\n🎯 Testing Vector Retrieval:")
        print("=" * 50)

        try:
            db = self.chatbot.db

            for term in test_terms:
                print(f"\n🔍 Searching for: '{term}'")
                docs = db.similarity_search(term, k=3)

                print(f"📄 Found {len(docs)} relevant documents:")
                for i, doc in enumerate(docs, 1):
                    content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                    print(f"  {i}. {content_preview}")

        except Exception as e:
            print(f"❌ Vector retrieval error: {e}")

    def interactive_test(self):
        """Interactive testing mode"""
        print("\n💬 Interactive RAG Testing Mode")
        print("Type 'quit' to exit")
        print("=" * 50)

        while True:
            try:
                user_input = input("\n🏥 Enter medical question: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break

                if not user_input:
                    continue

                print("🤖 MediClare is thinking...")
                start_time = time.time()

                response = self.chatbot.get_chatbot_answer(user_input)
                response_time = time.time() - start_time

                print(f"\n✅ Response ({response_time:.2f}s):")
                print(f"📝 {response}")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function"""
    print("🏥 MediClare Simple RAG Inference Tester")

    try:
        # Initialize tester
        tester = SimpleRAGTester()

        # Show menu
        print("\n📋 Choose testing mode:")
        print("1. Run predefined medical queries")
        print("2. Test vector retrieval only")
        print("3. Interactive testing")
        print("4. Run all tests")

        choice = input("\nEnter choice (1-4): ").strip()

        if choice == '1':
            tester.test_medical_queries()
        elif choice == '2':
            tester.test_vector_retrieval()
        elif choice == '3':
            tester.interactive_test()
        elif choice == '4':
            tester.test_vector_retrieval()
            tester.test_medical_queries()
            print("\n" + "=" * 50)
            print("All tests completed! Try interactive mode? (y/n)")
            if input().lower().startswith('y'):
                tester.interactive_test()
        else:
            print("Invalid choice. Running all tests...")
            tester.test_vector_retrieval()
            tester.test_medical_queries()

        print("\n✨ Testing completed!")

    except Exception as e:
        print(f"❌ Critical error: {e}")
        print("💡 Make sure:")
        print("  - Ollama is running (for LLM inference)")
        print("  - All required files are in the backend/data directory")
        print("  - Dependencies are installed")

if __name__ == "__main__":
    main()