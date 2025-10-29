#!/usr/bin/env python3
"""
RAG Storage Analyzer for MediClare
Analyzes the RAG system components and data structure
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime

try:
    from utils import variables, relative_path
    print("✅ Successfully loaded configuration")
except ImportError as e:
    print(f"❌ Error importing configuration: {e}")
    sys.exit(1)

class RAGStorageAnalyzer:
    """Analyze RAG storage components"""

    def __init__(self):
        """Initialize the analyzer"""
        self.backend_path = Path(__file__).parent
        self.data_path = self.backend_path / "data"

    def analyze_csv_data(self):
        """Analyze the medical CSV dataset"""
        print("📊 ANALYZING MEDICAL DATASET")
        print("=" * 50)

        # Use relative_path to get correct path
        csv_file = Path(relative_path(f"/{variables['model_embedding_csv_file']}")).resolve()

        if not csv_file.exists():
            print(f"❌ CSV file not found: {csv_file}")
            return

        try:
            # Load and analyze CSV
            df = pd.read_csv(csv_file)

            print(f"📁 File: {csv_file.name}")
            print(f"📈 Total records: {len(df):,}")
            print(f"🏷️  Columns: {list(df.columns)}")
            print(f"💾 File size: {csv_file.stat().st_size / (1024*1024):.1f} MB")

            # Analyze page titles (medical terms)
            if 'page_title' in df.columns:
                unique_titles = df['page_title'].nunique()
                print(f"🏥 Unique medical terms: {unique_titles:,}")

                # Show sample terms
                sample_titles = df['page_title'].dropna().sample(min(10, unique_titles)).tolist()
                print(f"📋 Sample medical terms:")
                for i, term in enumerate(sample_titles, 1):
                    print(f"  {i:2d}. {term}")

            # Analyze text content
            if 'page_text' in df.columns:
                avg_text_length = df['page_text'].str.len().mean()
                max_text_length = df['page_text'].str.len().max()
                min_text_length = df['page_text'].str.len().min()

                print(f"\n📝 Text Analysis:")
                print(f"   Average length: {avg_text_length:.0f} characters")
                print(f"   Maximum length: {max_text_length:,} characters")
                print(f"   Minimum length: {min_text_length:,} characters")

                # Count total words
                total_words = df['page_text'].str.split().str.len().sum()
                print(f"   Total words: {total_words:,}")

        except Exception as e:
            print(f"❌ Error analyzing CSV: {e}")

    def analyze_faiss_index(self):
        """Analyze FAISS vector index"""
        print("\n\n🎯 ANALYZING FAISS VECTOR INDEX")
        print("=" * 50)

        faiss_dir = Path(relative_path(f"/{variables['model_faiss_index_file']}")).resolve()

        if not faiss_dir.exists():
            print(f"❌ FAISS directory not found: {faiss_dir}")
            return

        try:
            # Check FAISS files
            index_file = faiss_dir / "index.faiss"
            pkl_file = faiss_dir / "index.pkl"

            print(f"📁 FAISS Directory: {faiss_dir.name}")

            if index_file.exists():
                size_mb = index_file.stat().st_size / (1024*1024)
                print(f"📈 index.faiss size: {size_mb:.1f} MB")
            else:
                print("❌ index.faiss not found")

            if pkl_file.exists():
                size_mb = pkl_file.stat().st_size / (1024*1024)
                print(f"📈 index.pkl size: {size_mb:.1f} MB")

                # Try to load and analyze the pickle file
                try:
                    with open(pkl_file, 'rb') as f:
                        pkl_data = pickle.load(f)

                    print(f"🏷️  Pickle data type: {type(pkl_data)}")
                    if hasattr(pkl_data, '__len__'):
                        print(f"📊 Pickle data length: {len(pkl_data)}")

                    # If it's a dictionary, show keys
                    if isinstance(pkl_data, dict):
                        print(f"🔑 Pickle keys: {list(pkl_data.keys())}")

                except Exception as e:
                    print(f"⚠️  Could not analyze pickle data: {e}")
            else:
                print("❌ index.pkl not found")

            # Try to load with FAISS to get vector count
            try:
                from langchain_community.vectorstores import FAISS
                from langchain_community.embeddings import HuggingFaceEmbeddings

                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                db = FAISS.load_local(str(faiss_dir), embeddings=embeddings, allow_dangerous_deserialization=True)

                print(f"✅ Successfully loaded FAISS index")
                print(f"🎯 Total vectors: {db.index.ntotal:,}")
                print(f"📐 Vector dimension: {db.index.d}")

            except Exception as e:
                print(f"⚠️  Could not load FAISS index: {e}")

        except Exception as e:
            print(f"❌ Error analyzing FAISS index: {e}")

    def analyze_embeddings(self):
        """Analyze stored embeddings"""
        print("\n\n🧠 ANALYZING EMBEDDINGS")
        print("=" * 50)

        embedding_file = Path(relative_path(f"/{variables['model_embedding_file']}")).resolve()

        if not embedding_file.exists():
            print(f"❌ Embeddings file not found: {embedding_file}")
            return

        try:
            size_mb = embedding_file.stat().st_size / (1024*1024)
            print(f"📁 File: {embedding_file.name}")
            print(f"💾 Size: {size_mb:.1f} MB")

            # Try to load embeddings
            with open(embedding_file, 'rb') as f:
                embeddings = pickle.load(f)

            print(f"🏷️  Type: {type(embeddings)}")

            # If it's HuggingFace embeddings, get model info
            if hasattr(embeddings, 'model_name'):
                print(f"🤖 Model: {embeddings.model_name}")

            if hasattr(embeddings, 'model_kwargs'):
                print(f"⚙️  Model kwargs: {embeddings.model_kwargs}")

        except Exception as e:
            print(f"❌ Error analyzing embeddings: {e}")

    def analyze_chunks(self):
        """Analyze document chunks"""
        print("\n\n📄 ANALYZING DOCUMENT CHUNKS")
        print("=" * 50)

        chunks_file = Path(relative_path(f"/{variables['model_chunks_file']}")).resolve()

        if not chunks_file.exists():
            print(f"❌ Chunks file not found: {chunks_file}")
            return

        try:
            size_mb = chunks_file.stat().st_size / (1024*1024)
            print(f"📁 File: {chunks_file.name}")
            print(f"💾 Size: {size_mb:.1f} MB")

            # Load chunks
            with open(chunks_file, 'rb') as f:
                chunks = pickle.load(f)

            print(f"📊 Total chunks: {len(chunks):,}")

            if chunks:
                # Analyze chunk properties
                chunk_lengths = [len(chunk.page_content) for chunk in chunks]
                avg_length = np.mean(chunk_lengths)
                max_length = max(chunk_lengths)
                min_length = min(chunk_lengths)

                print(f"📝 Chunk Analysis:")
                print(f"   Average length: {avg_length:.0f} characters")
                print(f"   Maximum length: {max_length:,} characters")
                print(f"   Minimum length: {min_length:,} characters")

                # Show sample chunks
                print(f"\n📋 Sample chunks:")
                for i, chunk in enumerate(chunks[:3], 1):
                    content_preview = chunk.page_content[:100] + "..." if len(chunk.page_content) > 100 else chunk.page_content
                    print(f"  {i}. {content_preview}")
                    if hasattr(chunk, 'metadata'):
                        print(f"     Metadata: {chunk.metadata}")

        except Exception as e:
            print(f"❌ Error analyzing chunks: {e}")

    def analyze_configuration(self):
        """Analyze RAG system configuration"""
        print("\n\n⚙️  RAG SYSTEM CONFIGURATION")
        print("=" * 50)

        print(f"🤖 LLM Model: {variables.get('model_name', 'Not set')}")
        print(f"🎯 T5 Model: {variables.get('t5_model_name', 'Not set')}")
        print(f"💻 Device: {variables.get('device', 'Not set')}")
        print(f"🔧 Run Mode: {variables.get('run_mode', 'Not set')}")

        print(f"\n📁 Data Files:")
        print(f"   CSV: {variables.get('model_embedding_csv_file', 'Not set')}")
        print(f"   Embeddings: {variables.get('model_embedding_file', 'Not set')}")
        print(f"   Chunks: {variables.get('model_chunks_file', 'Not set')}")
        print(f"   FAISS Index: {variables.get('model_faiss_index_file', 'Not set')}")

        print(f"\n🗄️  Database Configuration:")
        print(f"   Type: {variables.get('database_type', 'Not set')}")
        print(f"   Host: {variables.get('database_host', 'Not set')}")
        print(f"   Port: {variables.get('database_port', 'Not set')}")
        print(f"   Database: {variables.get('database_name', 'Not set')}")

    def check_file_integrity(self):
        """Check integrity of all RAG files"""
        print("\n\n🔍 FILE INTEGRITY CHECK")
        print("=" * 50)

        files_to_check = [
            ("CSV Dataset", Path(relative_path(f"/{variables['model_embedding_csv_file']}"))),
            ("Embeddings", Path(relative_path(f"/{variables['model_embedding_file']}"))),
            ("Chunks", Path(relative_path(f"/{variables['model_chunks_file']}"))),
            ("FAISS Index Dir", Path(relative_path(f"/{variables['model_faiss_index_file']}"))),
            ("FAISS Index", Path(relative_path(f"/{variables['model_faiss_index_file']}")) / "index.faiss"),
            ("FAISS Metadata", Path(relative_path(f"/{variables['model_faiss_index_file']}")) / "index.pkl"),
        ]

        all_present = True

        for name, path in files_to_check:
            if path.exists():
                size = "N/A"
                if path.is_file():
                    size_bytes = path.stat().st_size
                    if size_bytes > 1024*1024:
                        size = f"{size_bytes / (1024*1024):.1f} MB"
                    else:
                        size = f"{size_bytes / 1024:.1f} KB"

                print(f"✅ {name}: Present ({size})")
            else:
                print(f"❌ {name}: Missing")
                all_present = False

        print(f"\n{'✅' if all_present else '❌'} Overall integrity: {'GOOD' if all_present else 'ISSUES FOUND'}")

        return all_present

    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "=" * 60)
        print("📋 RAG STORAGE ANALYSIS SUMMARY REPORT")
        print("=" * 60)
        print(f"🕒 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📂 Backend Path: {self.backend_path}")
        print(f"🗄️  Data Path: {self.data_path}")

        # Run all analyses
        self.analyze_configuration()
        integrity_ok = self.check_file_integrity()
        self.analyze_csv_data()
        self.analyze_faiss_index()
        self.analyze_embeddings()
        self.analyze_chunks()

        print("\n" + "=" * 60)
        print("🎯 ANALYSIS COMPLETE")
        print("=" * 60)

        if integrity_ok:
            print("✅ RAG system appears to be properly configured")
            print("💡 You can now run inference tests with quick_rag_test.py")
        else:
            print("❌ Some RAG components are missing")
            print("💡 Please check the setup and regenerate missing files")

def main():
    """Main function"""
    print("🏥 MediClare RAG Storage Analyzer")
    print("Analyzing RAG system components and data structure")

    try:
        analyzer = RAGStorageAnalyzer()
        analyzer.generate_summary_report()

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()