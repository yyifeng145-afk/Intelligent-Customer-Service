# 定义状态图
from dotenv import load_dotenv

from app.import_process.agent.nodes.node_item_name_recognition import node_item_name_recognition
from app.import_process.agent.nodes.node_bge_embedding import node_bge_embedding
from app.import_process.agent.nodes.node_document_split import node_document_split
from app.import_process.agent.nodes.node_entry import node_entry
from app.import_process.agent.nodes.node_import_milvus import node_import_milvus
from app.import_process.agent.nodes.node_md_img import node_md_img
from app.import_process.agent.nodes.node_pdf_to_md import node_pdf_to_md
from app.import_process.agent.state import ImportGraphState
from langgraph.graph import StateGraph, END

load_dotenv()
graph = StateGraph(ImportGraphState)

# 添加节点
graph.add_node("node_entry",node_entry)
graph.add_node("node_pdf_to_md",node_pdf_to_md)
graph.add_node("node_md_img",node_md_img)
graph.add_node("node_document_split",node_document_split)
graph.add_node("node_item_name_recognition",node_item_name_recognition)
graph.add_node("node_bge_embedding",node_bge_embedding)
graph.add_node("node_import_milvus",node_import_milvus)

# 添加开始节点
graph.set_entry_point("node_entry")

# 添加边
def route_after_entry(state: ImportGraphState):
    if state['is_pdf_read_enabled']:
        return 'node_pdf_to_md'
    elif state['is_md_read_enabled']:
        return 'node_md_img'
    else:
        return END

graph.add_conditional_edges("node_entry",route_after_entry,{"node_pdf_to_md":"node_pdf_to_md","node_md_img":"node_md_img",END:END})

graph.add_edge("node_pdf_to_md","node_md_img")
graph.add_edge("node_md_img","node_document_split")
graph.add_edge("node_document_split","node_item_name_recognition")
graph.add_edge("node_item_name_recognition","node_bge_embedding")
graph.add_edge("node_bge_embedding","node_import_milvus")
graph.add_edge("node_import_milvus",END)

# 编译
kb_import_app = graph.compile()
