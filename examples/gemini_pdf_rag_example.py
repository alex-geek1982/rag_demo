#!/usr/bin/env python3
"""
简化版 PDF 混合 RAG 示例 - 使用 Gemini Vision API

特性：
1. 使用 rag_engine 解析 PDF 并构建知识图谱
2. 使用 Chroma 作为向量知识库
3. 使用 Google Gemini 2.5 Flash 做图像分析（Vision API）
4. 本地 Ollama 的 bge-m3 做 embedding
5. 本地 Ollama 的 bge-reranker 做 rerank
6. GPT-4.1-mini 做最终答案生成

准备工作：
1. 设置环境变量：
   $env:GEMINI_API_KEY = "your-gemini-api-key"
   $env:VISION_PROVIDER = "gemini"

2. 安装依赖：
   pip install google-generativeai

运行方式（PowerShell）：
$env:GEMINI_API_KEY = "<your-key>"
$env:VISION_PROVIDER = "gemini"
d:/workspace/llamaindex_demo/.venv/Scripts/python.exe examples/gemini_pdf_rag_example.py
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag_engine.config import (
    EmbeddingConfig,
    LLMConfig,
    PDFProcessingConfig,
    RAGEngineConfig,
)
from rag_engine.core import create_engine


def main():
    """Run PDF processing with Gemini Vision API"""
    
    # Configuration
    pdf_path = PROJECT_ROOT / "examples" / "京东订单多维度调度系统PRD1.0.pdf"
    output_dir = PROJECT_ROOT / "output" / "gemini_demo"
    
    # Environment
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_openai_base_url = os.getenv("OLLAMA_OPENAI_BASE_URL", f"{ollama_base_url}/v1")
    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "qllama/bge-m3:q4_k_m")
    gen_model = os.getenv("GEN_MODEL", "gpt-4.1-mini")
    deepbricks_base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepbricks.ai/v1/")
    deepbricks_api_key = os.getenv("DEEPBRICKS_API_KEY") or os.getenv("OPENAI_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    # Validate prerequisites
    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path}")
        return 1
    
    if not gemini_api_key:
        print("❌ GEMINI_API_KEY environment variable not set")
        print("   Set it with: $env:GEMINI_API_KEY = 'your-key'")
        return 1
    
    if not deepbricks_api_key:
        print("❌ DEEPBRICKS_API_KEY or OPENAI_API_KEY not set")
        return 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Gemini Vision PDF RAG Demo")
    print("=" * 80)
    print(f"📄 PDF: {pdf_path.name}")
    print(f"🔬 Vision Provider: Gemini 2.5 Flash")
    print(f"🎯 Embedding Model: {embed_model}")
    print(f"🤖 Generation Model: {gen_model}")
    
    try:
        # Create RAG engine with Gemini Vision
        config = RAGEngineConfig(
            working_dir=str(output_dir),
            output_dir=str(output_dir),
            embedding=EmbeddingConfig(
                model=embed_model,
                dimension=1024,
                api_key="ollama",
                base_url=ollama_openai_base_url,
            ),
            llm=LLMConfig(
                model=gen_model,
                api_key=deepbricks_api_key,
                base_url=deepbricks_base_url,
                temperature=0.2,
                max_tokens=900,
            ),
            pdf_processing=PDFProcessingConfig(
                use_advanced_layout=True,
                extract_images=True,
                extract_tables=True,
                extract_text=True,
                use_vision_api=True,
                vision_api_key=gemini_api_key,
                vision_provider="gemini",
                vision_model="gemini-2.5-flash",
            ),
            log_level="INFO",
        )
        
        engine = create_engine(config)
        
        print("\n📖 Processing PDF with Gemini Vision API...")
        document = engine.process_document(
            str(pdf_path),
            doc_title=pdf_path.stem,
            language="zh"
        )
        
        print(f"✅ Processing complete!")
        print(f"   📦 Content blocks: {len(document.content_blocks)}")
        print(f"   🔗 Entities: {len(engine.kg.entities)}")
        print(f"   ➡️  Relationships: {len(engine.kg.relationships)}")
        
        # Display sample content blocks
        print("\n📋 Sample extracted content:")
        for idx, block in enumerate(document.content_blocks[:3], 1):
            preview = block.content[:100].replace("\n", " ")
            print(f"   [{idx}] {block.type.value}: {preview}...")
        
        print("\n✅ Gemini Vision API integration successful!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
