"""
Chroma到Kuzu图数据库的完整管道示例
展示rebuild_from_chroma_chunks使用LightRAG pipeline构造高质量KG
"""
import asyncio
import logging
from pathlib import Path

from rag_engine.storage.kuzu_graph import KuzuGraphStore
from rag_engine.config import LLMConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """展示Chroma chunks → LightRAG KG → Kuzu的完整流程"""
    
    # 模拟Chroma知识库数据（来自向量数据库的chunks）
    chunk_ids = [
        "chunk_001",
        "chunk_002", 
        "chunk_003",
        "chunk_004",
    ]
    
    documents = [
        """
        Apple Inc. is a multinational technology company that designs, develops, and sells 
        consumer electronics, software, and online services. The company was founded in 1976 
        by Steve Jobs, Steve Wozniak, and Ronald Wayne in the garage of Jobs' parents' house 
        in Los Altos, California.
        """,
        """
        Steve Jobs served as Apple's CEO for much of its modern history. He was known for his 
        visionary leadership, focus on design excellence, and ability to create products that 
        revolutionized entire industries. Jobs passed away in 2011, leaving a lasting legacy.
        """,
        """
        Tim Cook joined Apple in 1998 as Senior Vice President of Worldwide Operations. 
        Following Steve Jobs' death, Cook became CEO in 2011. Cook has led Apple to become 
        the world's most valuable company, with a market capitalization exceeding $3 trillion.
        """,
        """
        Apple's headquarters is located in Cupertino, California. The campus features state-of-the-art 
        facilities and is known for its innovative design. Apple's supply chain operations span globally, 
        with manufacturing facilities and partnerships across multiple countries.
        """
    ]
    
    metadatas = [
        {
            "content_type": "text",
            "language": "en",
            "source_file": "apple_overview.txt",
            "chunk_index": 0
        },
        {
            "content_type": "text",
            "language": "en",
            "source_file": "steve_jobs_biography.txt",
            "chunk_index": 0
        },
        {
            "content_type": "text",
            "language": "en",
            "source_file": "apple_ceo_succession.txt",
            "chunk_index": 0
        },
        {
            "content_type": "text",
            "language": "en",
            "source_file": "apple_operations.txt",
            "chunk_index": 0
        }
    ]
    
    # 模拟LLM函数
    async def mock_llm_function(prompt: str) -> str:
        """模拟LLM调用 - 在实际应用中应该是真实的API"""
        if "entity" in prompt.lower() and "continue" in prompt.lower():
            return "entity<|#|>Steve Jobs<|#|>Person<|#|>Co-founder, former CEO<|COMPLETE|>"
        else:
            return """
entity<|#|>Apple Inc.<|#|>Organization<|#|>Multinational technology company
entity<|#|>Steve Jobs<|#|>Person<|#|>Co-founder and former CEO, visionary leader
entity<|#|>Tim Cook<|#|>Person<|#|>Current CEO, led Apple to $3 trillion market cap
entity<|#|>Cupertino<|#|>Location<|#|>Apple headquarters in California
relation<|#|>Apple Inc.<|#|>Steve Jobs<|#|>founded_by,co_founder<|#|>Steve Jobs co-founded Apple
relation<|#|>Apple Inc.<|#|>Tim Cook<|#|>led_by,managed_by<|#|>Tim Cook is current CEO
relation<|#|>Steve Jobs<|#|>Tim Cook<|#|>succeeded_by<|#|>Tim Cook succeeded Jobs as CEO
relation<|#|>Apple Inc.<|#|>Cupertino<|#|>headquartered_in<|#|>Apple HQ in Cupertino
<|COMPLETE|>
            """
    
    # 配置KG构造器
    global_config = {
        "max_gleaning_rounds": 1,
        "max_summarization_descriptions": 10,
        "force_llm_summary": False,
        "tuple_delimiter": "<|#|>",
        "completion_delimiter": "<|COMPLETE|>",
    }
    
    # 初始化Kuzu图数据库
    db_path = Path("output/chroma_to_kuzu_demo")
    kuzu_store = KuzuGraphStore(db_path)
    
    print("\n" + "="*80)
    print("利用LightRAG管道从Chroma chunks重建知识图谱")
    print("="*80)
    
    # 使用rebuild_from_chroma_chunks
    # 这会自动：
    # 1. 将chunks转换为ContentBlock
    # 2. 调用build_from_content_blocks执行完整管道（gleaning + merging）
    # 3. 将结果保存到Kuzu
    
    # 配置 LLM（使用默认配置或自定义）
    llm_config = LLMConfig(
        model="gpt-4.1-mini",
        # 或使用 Ollama：
        # model="qllama/bge-m3:q4_k_m",
        # base_url="http://localhost:11434/v1"
    )
    
    kuzu_store.rebuild_from_chroma_chunks(
        chunk_ids=chunk_ids,
        documents=documents,
        metadatas=metadatas,
        llm_config=llm_config,
        global_config=global_config
    )
    
    print("\n" + "="*80)
    print("完成！知识图谱已从Chroma重建")
    print("="*80)
    print(f"数据库位置: {db_path}")
    
    # 验证结果
    print("\n" + "-"*80)
    print("查询验证")
    print("-"*80)
    
    try:
        # 查询entities数量
        result = kuzu_store.conn.execute(
            "MATCH (e:entities) RETURN count(e) as entity_count"
        ).get_as_df()
        entity_count = result.iloc[0]["entity_count"] if not result.empty else 0
        print(f"提取Entity数: {entity_count}")
        
        # 查询relationships数量
        result = kuzu_store.conn.execute(
            "MATCH (a:entities)-[r:related]-(b:entities) RETURN count(r) as rel_count"
        ).get_as_df()
        rel_count = result.iloc[0]["rel_count"] if not result.empty else 0
        print(f"提取Relationship数: {rel_count}")
        
        # 显示entities
        print("\nEntities:")
        result = kuzu_store.conn.execute(
            "MATCH (e:entities) RETURN e.id, e.entity_type, e.description LIMIT 5"
        ).get_as_df()
        for _, row in result.iterrows():
            print(f"  - {row['e.id']} ({row['e.entity_type']})")
        
    except Exception as e:
        logger.warning(f"Query verification failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
