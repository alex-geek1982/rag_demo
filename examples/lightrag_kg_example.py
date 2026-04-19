"""
LightRAG知识图谱构造完整示例
展示gleaning、summarization和merging的完整管道
"""
import asyncio
import logging
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_engine.core.knowledge_graph import (
    LightRAGKnowledgeGraphBuilder,
)
from rag_engine.types import ContentBlock, ContentType, ModalityType
from rag_engine.storage.kuzu_graph import KuzuGraphStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """完整端到端知识图谱构造示例"""
    
    # 1. 准备示例内容块
    content_blocks = [
        ContentBlock(
            id="block_1",
            type=ContentType.TEXT,
            modality=ModalityType.TEXT,
            content="""
            Apple Inc. is a technology company founded by Steve Jobs in 1976.
            The company is headquartered in Cupertino, California.
            Apple develops and sells personal computers, mobile devices, and software.
            Tim Cook is the current CEO of Apple.
            Apple has partnerships with various suppliers worldwide.
            """,
            source_file="example_doc.txt"
        ),
        ContentBlock(
            id="block_2",
            type=ContentType.TEXT,
            modality=ModalityType.TEXT,
            content="""
            Steve Jobs co-founded Apple with Steve Wozniak and Ronald Wayne.
            Jobs served as CEO until 2011 when Tim Cook took over.
            The first Apple computer was the Apple I, released in 1976.
            Apple went public in December 1980.
            """,
            source_file="example_doc.txt"
        ),
        ContentBlock(
            id="block_3",
            type=ContentType.TEXT,
            modality=ModalityType.TEXT,
            content="""
            Tim Cook joined Apple in 1998 as Senior Vice President.
            He manages Apple's worldwide operations and supply chain.
            Cook is known for his focus on operational excellence.
            Under Cook's leadership, Apple became the first trillion-dollar company.
            """,
            source_file="example_doc.txt"
        ),
    ]
    
    # 2. 配置LightRAG KG构造器
    global_config = {
        # Gleaning配置 - 可选的refinement轮数
        "max_gleaning_rounds": 1,
        
        # 描述合成配置
        "max_summarization_descriptions": 10,
        "force_llm_summary": False,  # False时仅在多个描述时才调用LLM
        
        # 分隔符（LightRAG风格）
        "tuple_delimiter": "<|#|>",
        "completion_delimiter": "<|COMPLETE|>",
    }
    
    # 3. 创建LLM函数（这是一个模拟版本）
    async def mock_llm_function(prompt: str) -> str:
        """
        模拟LLM调用
        在实际使用中，这应该是真实的LLM API调用（如OpenAI, Claude等）
        """
        # 模拟LLM返回
        if "entity" in prompt.lower() and "continue" in prompt.lower():
            # 模拟gleaning补充提取
            return """
entity<|#|>Steve Jobs<|#|>Person<|#|>Co-founder of Apple, served as CEO, visionary leader<|COMPLETE|>
            """
        else:
            # 模拟标准提取
            return """
entity<|#|>Apple Inc.<|#|>Organization<|#|>Technology company headquartered in Cupertino
entity<|#|>Steve Jobs<|#|>Person<|#|>Co-founder and former CEO
entity<|#|>Tim Cook<|#|>Person<|#|>Current CEO of Apple
entity<|#|>Cupertino<|#|>Location<|#|>Headquarters of Apple in California
relation<|#|>Apple Inc.<|#|>Steve Jobs<|#|>founded_by,co_founder<|#|>Steve Jobs co-founded Apple
relation<|#|>Apple Inc.<|#|>Tim Cook<|#|>managed_by,led_by<|#|>Tim Cook is the current CEO
relation<|#|>Steve Jobs<|#|>Tim Cook<|#|>succeeded_by<|#|>Tim Cook succeeded Steve Jobs as CEO
relation<|#|>Apple Inc.<|#|>Cupertino<|#|>headquartered_in<|#|>Apple is headquartered in Cupertino<|COMPLETE|>
            """
    
    # 4. 初始化KG构造器
    kg_builder = LightRAGKnowledgeGraphBuilder(
        language="en",
        llm_func=mock_llm_function,
        global_config=global_config
    )
    
    # 5. 构造知识图谱 - 这是核心方法！
    # 包含以下阶段：
    # a. Entity提取 (可选gleaning轮数refinement)
    # b. Entity合并 (去重 + 描述合成)
    # c. Relationship提取
    # d. Relationship合并 (权重聚合 + 描述合成)
    print("\n" + "="*80)
    print("开始构造知识图谱...")
    print("="*80)
    
    entities, relationships = await kg_builder.build_from_content_blocks(content_blocks)
    
    print(f"\n知识图谱构造完成！")
    print(f"   - 提取实体数: {len(entities)}")
    print(f"   - 提取关系数: {len(relationships)}")
    
    # 6. 显示提取结果
    print("\n" + "-"*80)
    print("提取的实体:")
    print("-"*80)
    for entity_name, entity in list(entities.items())[:5]:  # 显示前5个
        print(f"  - {entity.entity_name}")
        print(f"    类型: {entity.entity_type}")
        print(f"    描述: {entity.description[:60]}...")
        print()
    
    print("-"*80)
    print("提取的关系 (前3个):")
    print("-"*80)
    for rel in relationships[:3]:
        print(f"  - {rel.src_id} --[{rel.keywords}]--> {rel.tgt_id}")
        print(f"    权重: {rel.weight}")
        print(f"    描述: {rel.description[:60]}...")
        print()
    
    # 7. 持久化到Kuzu数据库
    print("\n" + "="*80)
    print("持久化到Kuzu数据库...")
    print("="*80)
    
    db_path = Path("output/kg_demo")
    try:
        db_path.mkdir(parents=True, exist_ok=True)
        kuzu_store = KuzuGraphStore(db_path)
        
        # 使用提取的实体和关系数据重建图
        kuzu_store.rebuild_from_extracted_data(entities, relationships)
        
        print(f"知识图谱已保存到: {db_path}")
    except Exception as e:
        print(f"Kuzu数据库初始化失败: {e}")
        print("   这不影响知识图谱的构造结果。")
    
    print("\n" + "="*80)
    print("知识图谱构造完成！")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
