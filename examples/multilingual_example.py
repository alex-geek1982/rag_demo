"""
RAG Engine - Multilingual support example
"""
from rag_engine.config import RAGEngineConfig, LanguageConfig
from rag_engine.core import create_engine
from rag_engine.i18n import set_language

def multilingual_example():
    """Example showing multilingual support"""
    
    # Create config with multilingual support
    config = RAGEngineConfig.from_env()
    
    print("=" * 60)
    print("MULTILINGUAL SUPPORT EXAMPLE")
    print("=" * 60)
    
    # Demonstrate language switching
    languages = ["en", "zh", "ja", "ko"]
    
    for lang in languages:
        set_language(lang)
        
        # Create engine for this language
        config.language.default_language = lang
        engine = create_engine(config)
        
        print(f"\\n🌍 Language: {lang.upper()}")
        print(f"   Supported languages: {', '.join(engine.i18n.get_supported_languages())}")
    
    # Example: Process documents in different languages
    print("\\n" + "=" * 60)
    print("PROCESSING DOCUMENTS IN DIFFERENT LANGUAGES")
    print("=" * 60)
    
    from pathlib import Path
    
    # English document
    en_doc_path = Path(config.output_dir) / "english_doc.txt"
    en_doc_path.parent.mkdir(parents=True, exist_ok=True)
    with open(en_doc_path, 'w', encoding='utf-8') as f:
        f.write("English document about artificial intelligence and machine learning.")
    
    # Chinese document
    zh_doc_path = Path(config.output_dir) / "chinese_doc.txt"
    with open(zh_doc_path, 'w', encoding='utf-8') as f:
        f.write("这是一份关于人工智能和机器学习的中文文档。")
    
    # Process documents with appropriate language settings
    set_language("en")
    en_config = RAGEngineConfig.from_env()
    en_config.language.default_language = "en"
    en_engine = create_engine(en_config)
    
    print("\\n📄 Processing English document...")
    en_doc = en_engine.process_document(str(en_doc_path), language="en")
    print(f"✅ Processed: {len(en_doc.content_blocks)} blocks")
    
    print("\\n📄 Processing Chinese document...")
    zh_config = RAGEngineConfig.from_env()
    zh_config.language.default_language = "zh"
    set_language("zh")
    zh_engine = create_engine(zh_config)
    zh_doc = zh_engine.process_document(str(zh_doc_path), language="zh")
    print(f"✅ Processed: {len(zh_doc.content_blocks)} blocks")


if __name__ == "__main__":
    try:
        multilingual_example()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
