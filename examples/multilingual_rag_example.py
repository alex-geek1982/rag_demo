"""
Multilingual RAG Example

This example demonstrates true multilingual capabilities:
1. Document processing in multiple languages
2. Cross-lingual retrieval
3. Language detection
4. Multilingual query support
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_engine import RAGEngine
from rag_engine.config import RAGEngineConfig
from rag_engine.types import Document, ContentBlock, ContentType, ModalityType
from rag_engine.i18n import detect_language, is_multilingual_content, LanguageDetector


def example_1_language_detection():
    """Example 1: Language detection for content"""
    print("\n" + "="*60)
    print("Example 1: Language Detection")
    print("="*60)
    
    test_texts = {
        "English": "The quick brown fox jumps over the lazy dog.",
        "Chinese": "快速褐色的狐狸跳过懒狗。",
        "Japanese": "速い茶色のキツネが怠け者の犬をジャンプします。",
        "Spanish": "El rápido zorro marrón salta sobre el perro perezoso.",
        "German": "Der schnelle braune Fuchs springt über den faulen Hund.",
        "French": "Le rapide renard brun saute par-dessus le chien paresseux.",
        "Korean": "빠른 갈색 여우가 게으른 개를 뛰어넘습니다.",
    }
    
    detector = LanguageDetector()
    
    print("\nDetecting language for different texts:")
    for language_name, text in test_texts.items():
        detected = detector.detect(text)
        language_map = {
            "en": "English",
            "zh": "Chinese",
            "ja": "Japanese",
            "es": "Spanish",
            "de": "German",
            "fr": "French",
            "ko": "Korean",
        }
        detected_name = language_map.get(detected, detected)
        print(f"  {language_name:12} → Detected: {detected_name:12} ({detected})")


def example_2_multilingual_content_detection():
    """Example 2: Detect if content contains multiple languages"""
    print("\n" + "="*60)
    print("Example 2: Multilingual Content Detection")
    print("="*60)
    
    test_sets = {
        "Single Language (English)": [
            "The quick brown fox jumps over the lazy dog.",
            "This is another English text.",
        ],
        "Mixed Languages": [
            "The quick brown fox",
            "快速褐色的狐狸",
            "速い茶色のキツネ",
        ],
        "Bilingual (EN-ZH)": [
            "English and 中文 mixed content",
            "Another 例子 with both languages",
        ],
    }
    
    for description, texts in test_sets.items():
        is_multi = is_multilingual_content(texts)
        status = "✓ Multilingual" if is_multi else "✗ Single Language"
        print(f"\n  {description}: {status}")
        for text in texts:
            lang_code = detect_language(text)
            print(f"    - {text[:40]:40} → {lang_code}")


def example_3_create_multilingual_documents():
    """Example 3: Create and process multilingual documents"""
    print("\n" + "="*60)
    print("Example 3: Creating Multilingual Documents")
    print("="*60)
    
    try:
        config = RAGEngineConfig.from_env()
        engine = RAGEngine(config)
        
        # Create multilingual content blocks
        multilingual_content = [
            {
                "language": "en",
                "title": "English Document",
                "content": """
                Artificial Intelligence (AI) is transforming the world.
                Machine learning enables computers to learn from data.
                Deep learning powers modern AI applications.
                Natural language processing helps machines understand human language.
                """
            },
            {
                "language": "zh",
                "title": "中文文档",
                "content": """
                人工智能正在改变世界。
                机器学习使计算机能够从数据中学习。
                深度学习推动现代AI应用。
                自然语言处理帮助机器理解人类语言。
                """
            },
            {
                "language": "ja",
                "title": "日本語の文書",
                "content": """
                人工知能は世界を変えています。
                機械学習により、コンピューターはデータから学習できます。
                ディープラーニングは最新のAIアプリケーションを強化しています。
                自然言語処理は、機械が人間の言語を理解するのに役立ちます。
                """
            }
        ]
        
        # Process each language version
        documents = []
        for item in multilingual_content:
            # Create document
            doc = Document(
                id=f"doc_{item['language']}",
                title=item['title'],
                source_path=f"multilingual_{item['language']}.txt",
                language=item['language']
            )
            
            # Create content block
            block = ContentBlock(
                id=f"block_{item['language']}_1",
                type=ContentType.TEXT,
                content=item['content'],
                modality=ModalityType.TEXT,
                language=item['language'],
                metadata={"source": f"{item['title']}"}
            )
            
            doc.add_content_block(block)
            documents.append(doc)
            
            print(f"\n  ✓ Created {item['language'].upper()} document: {item['title']}")
            print(f"    Content blocks: {len(doc.content_blocks)}")
            print(f"    Language: {doc.language}")
        
        print(f"\n✓ Successfully created {len(documents)} multilingual documents")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_4_multilingual_query_demonstration():
    """Example 4: Demonstrate multilingual query capabilities"""
    print("\n" + "="*60)
    print("Example 4: Multilingual Query Capabilities")
    print("="*60)
    
    print("\nSupported multilingual features:")
    
    features = {
        "1. Language Detection": "Automatic detection of query language (7+ languages)",
        "2. Cross-lingual Retrieval": "Query in one language, find matches in others",
        "3. Language Affinity": "Higher relevance for same-language matches",
        "4. Multilingual Indexing": "Support for documents in any supported language",
        "5. Language-aware Ranking": "Results ranked considering query and document languages",
        "6. Metadata Tracking": "Language information preserved in query results",
    }
    
    for feature, description in features.items():
        print(f"\n  {feature}")
        print(f"    → {description}")


def example_5_supported_languages():
    """Example 5: Display supported languages and capabilities"""
    print("\n" + "="*60)
    print("Example 5: Supported Languages and Capabilities")
    print("="*60)
    
    from rag_engine.i18n import SUPPORTED_LANGUAGES
    
    print("\nSupported languages:")
    for lang_code, lang_info in SUPPORTED_LANGUAGES.items():
        print(f"\n  {lang_code.upper()}: {lang_info['name']}")
        print(f"    - Embedding Model: {lang_info['embedding_model']}")
        print(f"    - Cross-lingual Support: ✓ Yes")
        print(f"    - Language Detection: ✓ Yes")


def example_6_language_affinity_scoring():
    """Example 6: Understand language affinity scoring"""
    print("\n" + "="*60)
    print("Example 6: Language Affinity and Cross-lingual Scoring")
    print("="*60)
    
    from rag_engine.i18n import CrosslingualRetrieval
    
    crosslingual = CrosslingualRetrieval()
    
    print("\nLanguage affinity scores for retrieval:")
    print("(Higher = better cross-lingual match)")
    
    # Display some example affinities
    examples = [
        ("en", "zh", "English → Chinese"),
        ("en", "ja", "English → Japanese"),
        ("en", "ko", "English → Korean"),
        ("es", "fr", "Spanish → French"),
        ("zh", "ja", "Chinese → Japanese"),
        ("en", "en", "English ↔ English (same language boost)"),
    ]
    
    for query_lang, doc_lang, description in examples:
        base_sim = 0.75  # Example base similarity
        adjusted_sim = crosslingual.get_cross_lingual_score(query_lang, doc_lang, base_sim)
        boost = crosslingual.get_same_language_boost(query_lang, doc_lang)
        
        print(f"\n  {description}")
        print(f"    Base similarity: {base_sim:.3f}")
        print(f"    Cross-lingual adjusted: {adjusted_sim:.3f}")
        print(f"    Same-language boost: {boost:.2f}x")


def example_7_multilingual_workflow():
    """Example 7: Complete multilingual RAG workflow"""
    print("\n" + "="*60)
    print("Example 7: Complete Multilingual RAG Workflow")
    print("="*60)
    
    print("\nMultilingual RAG workflow steps:")
    
    steps = [
        "1. Document Ingestion",
        "   - Accept documents in multiple languages",
        "   - Auto-detect language for each document",
        
        "2. Content Processing",
        "   - Process content based on detected language",
        "   - Maintain language information in metadata",
        
        "3. Multi-lingual Embedding",
        "   - Generate embeddings with language awareness",
        "   - Tag embeddings with source language",
        
        "4. Query Processing",
        "   - Detect query language automatically",
        "   - Log query language for analytics",
        
        "5. Cross-lingual Retrieval",
        "   - Search across documents in all languages",
        "   - Apply language affinity scoring",
        "   - Boost same-language matches (15% boost)",
        
        "6. Result Ranking",
        "   - Rank results by adjusted similarity score",
        "   - Include language metadata in results",
        "   - Flag cross-lingual matches",
        
        "7. Answer Generation",
        "   - Generate answers considering multilingual context",
        "   - Preserve language evidence in sources",
    ]
    
    for step in steps:
        print(f"  {step}")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("Multilingual RAG Engine - Feature Demonstration")
    print("="*60)
    
    examples = [
        example_1_language_detection,
        example_2_multilingual_content_detection,
        example_3_create_multilingual_documents,
        example_4_multilingual_query_demonstration,
        example_5_supported_languages,
        example_6_language_affinity_scoring,
        example_7_multilingual_workflow,
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n✗ Error in {example_func.__name__}: {e}")
    
    print("\n" + "="*60)
    print("Multilingual Examples Complete")
    print("="*60)
    print("\nKey Takeaways:")
    print("  ✓ The RAG engine supports 7+ languages natively")
    print("  ✓ Language is detected automatically for all content")
    print("  ✓ Cross-lingual retrieval finds relevant results across languages")
    print("  ✓ Same-language matches receive a 15% relevance boost")
    print("  ✓ Language metadata is preserved throughout the pipeline")
    print("  ✓ This enables true multilingual information retrieval")


if __name__ == "__main__":
    main()
