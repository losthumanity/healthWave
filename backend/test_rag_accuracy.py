#!/usr/bin/env python3
"""
RAG System Accuracy Testing Script for MediClare
Tests the retrieval and accuracy of medical term explanations from the FAISS vector database
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time
import json
from typing import List, Dict, Tuple, Any
from datetime import datetime

# Import the RAG system components
from app.model import Chatbot
from utils import variables

class RAGAccuracyTester:
    """Test suite for evaluating RAG system accuracy with medical terms"""

    def __init__(self):
        """Initialize the RAG system and test framework"""
        print("🏥 Initializing MediClare RAG Accuracy Tester")
        print("=" * 60)

        # Initialize the chatbot with RAG system
        try:
            self.chatbot = Chatbot(
                model_name=variables["model_name"],
                csv_file=variables["model_embedding_csv_file"],
                embedding_file=variables["model_embedding_file"],
                chunks_file=variables["model_chunks_file"],
                fass_index_file=variables["model_faiss_index_file"],
            )
            print("✅ RAG system initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing RAG system: {e}")
            sys.exit(1)

        # Load the medical dataset for testing
        try:
            from utils import relative_path
            csv_path = Path(relative_path(f"/{variables['model_embedding_csv_file']}"))
            self.medical_data = pd.read_csv(csv_path)
            print(f"✅ Loaded medical dataset: {len(self.medical_data)} entries")
        except Exception as e:
            print(f"❌ Error loading medical dataset: {e}")
            sys.exit(1)

        # Test results storage
        self.test_results = []

    def extract_medical_terms(self, n_terms: int = 20) -> List[str]:
        """Extract medical terms from the dataset for testing"""
        print(f"\n🔍 Extracting {n_terms} medical terms for testing...")

        # Get unique medical terms from page titles
        medical_terms = []

        # Sample from page titles (these are typically medical conditions/terms)
        titles = self.medical_data['page_title'].dropna().unique()

        # Select a diverse set of medical terms
        selected_terms = np.random.choice(titles, min(n_terms, len(titles)), replace=False)

        for term in selected_terms:
            medical_terms.append(term.strip())

        print(f"✅ Selected medical terms: {medical_terms[:5]}... (showing first 5)")
        return medical_terms

    def test_single_query(self, medical_term: str) -> Dict[str, Any]:
        """Test a single medical term query against the RAG system"""
        print(f"\n🧪 Testing query: '{medical_term}'")

        start_time = time.time()

        try:
            # Test different query formats
            queries = [
                f"What is {medical_term}?",
                f"Explain {medical_term}",
                f"Tell me about {medical_term}",
                medical_term  # Direct term
            ]

            results = {}

            for i, query in enumerate(queries, 1):
                try:
                    print(f"  📝 Query {i}: {query}")
                    response = self.chatbot.get_chatbot_answer(query)
                    response_time = time.time() - start_time

                    results[f"query_{i}"] = {
                        "query": query,
                        "response": response,
                        "response_time": response_time,
                        "response_length": len(response) if response else 0,
                        "success": bool(response and len(response) > 10)
                    }

                    print(f"    ✅ Response length: {len(response) if response else 0} chars")
                    print(f"    ⏱️  Response time: {response_time:.2f}s")

                except Exception as e:
                    print(f"    ❌ Error: {e}")
                    results[f"query_{i}"] = {
                        "query": query,
                        "response": None,
                        "error": str(e),
                        "success": False
                    }

            return {
                "medical_term": medical_term,
                "timestamp": datetime.now().isoformat(),
                "total_time": time.time() - start_time,
                "results": results,
                "overall_success": any(r.get("success", False) for r in results.values())
            }

        except Exception as e:
            print(f"    ❌ Critical error: {e}")
            return {
                "medical_term": medical_term,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "overall_success": False
            }

    def test_vector_retrieval(self, medical_term: str, k: int = 3) -> Dict[str, Any]:
        """Test the vector similarity retrieval directly"""
        print(f"\n🎯 Testing vector retrieval for: '{medical_term}'")

        try:
            # Get the FAISS database
            db = self.chatbot.db

            # Perform similarity search
            start_time = time.time()
            docs = db.similarity_search(medical_term, k=k)
            retrieval_time = time.time() - start_time

            retrieved_docs = []
            for i, doc in enumerate(docs):
                doc_info = {
                    "rank": i + 1,
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "full_content_length": len(doc.page_content),
                    "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
                }
                retrieved_docs.append(doc_info)
                print(f"  📄 Rank {i+1}: {doc_info['content']}")

            return {
                "medical_term": medical_term,
                "retrieval_time": retrieval_time,
                "num_retrieved": len(docs),
                "retrieved_docs": retrieved_docs,
                "success": len(docs) > 0
            }

        except Exception as e:
            print(f"    ❌ Retrieval error: {e}")
            return {
                "medical_term": medical_term,
                "error": str(e),
                "success": False
            }

    def evaluate_response_quality(self, response: str, medical_term: str) -> Dict[str, Any]:
        """Evaluate the quality of RAG responses"""
        if not response:
            return {"score": 0, "issues": ["No response generated"]}

        score = 0
        issues = []
        positive_indicators = []

        # Check if the medical term appears in the response
        if medical_term.lower() in response.lower():
            score += 20
            positive_indicators.append("Contains medical term")
        else:
            issues.append("Medical term not mentioned in response")

        # Check response length (should be informative but not too short)
        if len(response) < 50:
            issues.append("Response too short (< 50 characters)")
        elif len(response) > 100:
            score += 15
            positive_indicators.append("Adequate response length")

        # Check for medical explanation indicators
        explanation_words = ["is", "means", "condition", "disease", "treatment", "symptoms", "causes"]
        if any(word in response.lower() for word in explanation_words):
            score += 20
            positive_indicators.append("Contains explanation keywords")

        # Check for structured information
        if any(char in response for char in ['-', '•', '\n']):
            score += 10
            positive_indicators.append("Well-structured response")

        # Check for error messages or non-medical responses
        error_indicators = ["error", "sorry", "cannot", "unable", "technical difficulties"]
        if any(indicator in response.lower() for indicator in error_indicators):
            score -= 30
            issues.append("Contains error indicators")

        # Bonus for comprehensive responses
        if len(response) > 300:
            score += 10
            positive_indicators.append("Comprehensive response")

        # Normalize score to 0-100
        score = max(0, min(100, score))

        return {
            "score": score,
            "issues": issues,
            "positive_indicators": positive_indicators,
            "response_length": len(response)
        }

    def run_comprehensive_test(self, n_terms: int = 15) -> Dict[str, Any]:
        """Run comprehensive accuracy testing"""
        print(f"\n🚀 Starting comprehensive RAG accuracy test with {n_terms} medical terms")
        print("=" * 60)

        # Extract medical terms
        medical_terms = self.extract_medical_terms(n_terms)

        # Test results
        query_results = []
        retrieval_results = []
        quality_scores = []

        # Test each medical term
        for i, term in enumerate(medical_terms, 1):
            print(f"\n🏥 Testing {i}/{len(medical_terms)}: {term}")
            print("-" * 40)

            # Test query processing
            query_result = self.test_single_query(term)
            query_results.append(query_result)

            # Test vector retrieval
            retrieval_result = self.test_vector_retrieval(term)
            retrieval_results.append(retrieval_result)

            # Evaluate response quality for the first successful query
            for query_key, query_data in query_result.get("results", {}).items():
                if query_data.get("success") and query_data.get("response"):
                    quality = self.evaluate_response_quality(query_data["response"], term)
                    quality["medical_term"] = term
                    quality["query"] = query_data["query"]
                    quality_scores.append(quality)
                    print(f"    📊 Quality score: {quality['score']}/100")
                    break

        # Calculate overall statistics
        successful_queries = sum(1 for r in query_results if r.get("overall_success"))
        successful_retrievals = sum(1 for r in retrieval_results if r.get("success"))
        avg_quality_score = np.mean([q["score"] for q in quality_scores]) if quality_scores else 0
        avg_response_time = np.mean([r.get("total_time", 0) for r in query_results if r.get("total_time")])

        summary = {
            "test_summary": {
                "total_terms_tested": len(medical_terms),
                "successful_queries": successful_queries,
                "successful_retrievals": successful_retrievals,
                "query_success_rate": (successful_queries / len(medical_terms)) * 100,
                "retrieval_success_rate": (successful_retrievals / len(medical_terms)) * 100,
                "average_quality_score": round(avg_quality_score, 2),
                "average_response_time": round(avg_response_time, 2),
                "test_timestamp": datetime.now().isoformat()
            },
            "detailed_results": {
                "query_results": query_results,
                "retrieval_results": retrieval_results,
                "quality_scores": quality_scores
            }
        }

        return summary

    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save test results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_accuracy_test_{timestamp}.json"

        filepath = Path(__file__).parent / filename

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to: {filepath}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")

    def print_summary_report(self, results: Dict[str, Any]):
        """Print a formatted summary report"""
        summary = results["test_summary"]

        print("\n" + "=" * 60)
        print("📋 RAG SYSTEM ACCURACY TEST REPORT")
        print("=" * 60)
        print(f"🕒 Test Date: {summary['test_timestamp']}")
        print(f"🧪 Total Terms Tested: {summary['total_terms_tested']}")
        print(f"✅ Successful Queries: {summary['successful_queries']}/{summary['total_terms_tested']} ({summary['query_success_rate']:.1f}%)")
        print(f"🎯 Successful Retrievals: {summary['successful_retrievals']}/{summary['total_terms_tested']} ({summary['retrieval_success_rate']:.1f}%)")
        print(f"📊 Average Quality Score: {summary['average_quality_score']}/100")
        print(f"⏱️  Average Response Time: {summary['average_response_time']:.2f}s")

        print("\n📈 PERFORMANCE RATING:")
        if summary['query_success_rate'] >= 90:
            print("🌟 EXCELLENT - RAG system is performing very well")
        elif summary['query_success_rate'] >= 75:
            print("✅ GOOD - RAG system is performing adequately")
        elif summary['query_success_rate'] >= 50:
            print("⚠️  FAIR - RAG system needs improvement")
        else:
            print("❌ POOR - RAG system requires significant improvements")

        # Top performing queries
        quality_scores = results["detailed_results"]["quality_scores"]
        if quality_scores:
            top_scores = sorted(quality_scores, key=lambda x: x["score"], reverse=True)[:3]
            print(f"\n🏆 TOP PERFORMING QUERIES:")
            for i, score in enumerate(top_scores, 1):
                print(f"  {i}. {score['medical_term']} - Score: {score['score']}/100")

        print("=" * 60)

def main():
    """Main function to run the RAG accuracy tests"""
    print("🏥 MediClare RAG System Accuracy Tester")
    print("Testing the accuracy of medical term retrieval and explanation")

    # Initialize tester
    tester = RAGAccuracyTester()

    # Ask user for number of terms to test
    try:
        n_terms = int(input("\nEnter number of medical terms to test (default: 15): ") or 15)
        n_terms = max(5, min(50, n_terms))  # Limit between 5-50
    except ValueError:
        n_terms = 15
        print("Using default: 15 terms")

    print(f"\n🚀 Running accuracy test with {n_terms} medical terms...")

    # Run the test
    results = tester.run_comprehensive_test(n_terms)

    # Print summary
    tester.print_summary_report(results)

    # Ask to save results
    save_choice = input("\n💾 Save detailed results to file? (y/n): ").lower().strip()
    if save_choice in ['y', 'yes']:
        tester.save_results(results)

    print("\n✨ Test completed successfully!")

if __name__ == "__main__":
    main()