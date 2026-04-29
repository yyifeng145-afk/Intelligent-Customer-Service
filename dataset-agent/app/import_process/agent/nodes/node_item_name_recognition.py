import os
import sys

from langchain_core.messages import HumanMessage, SystemMessage
from pyarrow import DataType

from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.import_process.agent.state import ImportGraphState
from app.core.logger import logger
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.utils.task_utils import add_done_task, add_running_task
DEFAULT_ITEM_NAME_CHUNK_K = 5
# 单个切片内容截断长度：防止单切片内容过长，占满大模型上下文
SINGLE_CHUNK_CONTENT_MAX_LEN = 800
# 大模型上下文总字符数上限：适配主流大模型输入限制，默认2500
CONTEXT_TOTAL_MAX_CHARS = 2500

def step1_get_params(state):
    file_title = state['file_title']
    chunks = state['states']
    return file_title,chunks


def step2_cut_chunks(chunks):
    """
        截取内容处理：
          切片：{1}，标题:{title},内容：{content} \n\n
          切片：{2}，标题:{title},内容：{content} \n\n
          切片：{3}，标题:{title},内容：{content} \n\n
          切片：{4}，标题:{title},内容：{content} \n\n
          切片：{5}，标题:{title},内容：{content} \n\n
    """
    last_context = []
    str_length = 0
    for i,chunk in enumerate(chunks[:DEFAULT_ITEM_NAME_CHUNK_K],start=1):
        chunk_title = chunk['title']
        chunk_content = chunk['content']
        data = f"切片:{i},标题:{chunk_title}，内容:{chunk_content}"
        last_context.append(data)
        str_length = str_length + len(data)
        if str_length > CONTEXT_TOTAL_MAX_CHARS:
            break
    context = '\n\n'.join(last_context)
    return context[:CONTEXT_TOTAL_MAX_CHARS]


def step3_get_item_name_by_llm(file_title, context):
    #借助大模型得到chunks的item_name
    human_prompt = load_prompt("item_name_recognition",file_title=file_title,context=context)
    system_prompt = load_prompt("product_recognition_system")

    messages = [
        HumanMessage(content=human_prompt),
        SystemMessage(content=system_prompt)
    ]
    llm = get_llm_client()

    response = llm.invoke(messages)

    item_name = response.content
    if not item_name:
        item_name = file_title
    return item_name


def step4_update_state_and_chunks(state, item_name, chunks):
    state['item_name'] = item_name
    for chunk in chunks:
        chunk['item_name'] = item_name
    state['chunks'] = chunks
    
    logger.info(f"state的状态更新完成!!!")


def step5_generate_vector_by_embedding_model(item_name):
    result = generate_embeddings([item_name])
    dense_vector,sparse_vector = result['dense'][0],result['sparse'][0]
    return dense_vector,sparse_vector


def step6_save_to_db(file_title, dense_vector, sparse_vector, item_name):
    #1.创建milvus客户端
    client = get_milvus_client()
    #2.判断是否存在collection 若不存在则创建collection集合
    if not client.has_collection(collection_name=milvus_config.item_name_collection):
        #创建集合
        schema = client.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )

        # Add fields to schema
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 添加索引参数
        index_params = client.prepare_index_params()

        index_params.add_index(
            field_name="dense_vector",
            index_type="HNSW",  # 精确可控，生产常用；也可换 HNSW
            index_name="dense_vector_index",
            metric_type="COSINE",
            params={
                "M": 16,  # Maximum number of neighbors each node can connect to in the graph
                "efConstruction": 200
            }
        )

        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            index_name="sparse_inverted_index",
            metric_type="IP",
            params={"inverted_index_algo": "DAAT_MAXSCORE"}
        )

        client.create_collection(
            collection_name=milvus_config.item_name_collection,
            schema=schema,
            index_params=index_params
        )

        client.get_load_state(
            collection_name=milvus_config.item_name_collection
        )

    # 3.若collection存在 删除collection里面的内容
    client.delete(
        collection_name=milvus_config.item_name_collection,
        filter=f"item_name == '{item_name}'"
    )
    #4.向数据库插入最新的item_name
    data = {
        "file_title": file_title,
        "item_name": item_name,
        "sparse_vector": sparse_vector,
        "dense_vector": dense_vector
    }
    client.insert(
        collection_name=milvus_config.item_name_collection,
        data=[data]
    )
    client.get_load_state(
        collection_name=milvus_config.item_name_collection
    )
    logger.info(f"{item_name}已经保存到milvus向量数据库中了...")


def node_item_name_recognition(state: ImportGraphState)->ImportGraphState:

    # 1.记录当前节点日志(同时将当前节点推送到前端)
    node_name = sys._getframe().f_code.co_name
    logger.info(f"当前执行的节点是{node_name},当前的状态是{state}")
    add_running_task(state['task_id'],node_name)

    try:
        #1首先进行参数校验(file_title 以及chunks)
        file_title,chunks = step1_get_params(state)

        #2对chunks进行截取前5个同时保证字符长度不超过最大字符长度2500
        context = step2_cut_chunks(chunks)

        #3将得到的上下文结合提示词借助大模型得到item_name 若得不到item_name则借助file_title进行兜底
        item_name = step3_get_item_name_by_llm(file_title,context)

        #4将item_name存储到chunks中 同时更新state
        step4_update_state_and_chunks(state,item_name,chunks)
        
        #5调用嵌入模型 将item_name生成稀疏向量以及稠密向量
        dense_vector,sparse_vector = step5_generate_vector_by_embedding_model(item_name)

        #6将向量存到数据库中(传入的参数:file_title dense_vector sparse_vector item_name)
        step6_save_to_db(file_title,dense_vector,sparse_vector,item_name)

    except Exception as e:

        logger.info(f"使用mineru解析出现错误 其中错误是{e}")
        raise RuntimeError(f"结束！！！")
    finally:
        # 5.记录结束节点日志(同时将结束节点推送到前端)
        logger.info(f"当前结束节点的名称是,当前的状态是{state}")
        add_done_task(state['task_id'], node_name)


    return state



# ===================== 本地测试方法（直接运行调试，无需启动LangGraph） =====================
def test_node_item_name_recognition():
    """
    商品名称识别节点本地测试方法
    功能：模拟LangGraph流程输入，独立测试node_item_name_recognition节点全链路逻辑
    适用场景：本地开发、调试、单节点功能验证，无需启动整个LangGraph流程
    测试前准备：
        1. 确保项目环境变量配置完成（MILVUS_URL/ITEM_NAME_COLLECTION等）
        2. 确保大模型、Milvus、BGE-M3服务均可正常访问
        3. 确保prompt模板（item_name_recognition/product_recognition_system）已存在
    使用方法：
        直接运行该函数：if __name__ == "__main__": test_node_item_name_recognition()
    """
    logger.info("=== 开始执行商品名称识别节点本地测试 ===")
    try:
        # 1. 构造模拟的ImportGraphState状态（模拟上游节点产出数据）
        mock_state = ImportGraphState({
            "task_id": "test_task_123456",  # 测试任务ID
            "file_title": "华为Mate60 Pro手机使用说明书",  # 模拟文件标题
            "file_name": "华为Mate60Pro说明书.pdf",  # 模拟原始文件名（兜底用）
            # 模拟文本切片列表（上游切片节点产出，含title/content字段）
            "chunks": [
                {
                    "title": "产品简介",
                    "content": "华为Mate60 Pro是华为公司2023年发布的旗舰智能手机，搭载麒麟9000S芯片，支持卫星通话功能，屏幕尺寸6.82英寸，分辨率2700×1224。"
                },
                {
                    "title": "拍照功能",
                    "content": "华为Mate60 Pro后置5000万像素超光变摄像头+1200万像素超广角摄像头+4800万像素长焦摄像头，支持5倍光学变焦，100倍数字变焦。"
                },
                {
                    "title": "电池参数",
                    "content": "电池容量5000mAh，支持88W有线超级快充，50W无线超级快充，反向无线充电功能。"
                }
            ]
        })

        # 2. 调用商品名称识别核心节点
        result_state = node_item_name_recognition(mock_state)

        # 3. 打印测试结果（调试用）
        logger.info("=== 商品名称识别节点本地测试完成 ===")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"最终识别商品名称：{result_state.get('item_name')}")
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")
        logger.info(f"第一个切片商品名称：{result_state.get('chunks', [{}])[0].get('item_name')}")

        # 4. 验证Milvus存储（可选）
        milvus_client = get_milvus_client()
        collection_name = os.environ.get("ITEM_NAME_COLLECTION")
        if milvus_client and collection_name:
            milvus_client.load_collection(collection_name)
            # 检索测试结果
            item_name = result_state.get('item_name')
            safe_name = escape_milvus_string(item_name)
            res = milvus_client.query(
                collection_name=collection_name,
                filter=f'item_name=="{safe_name}"',
                output_fields=["file_title", "item_name"]
            )
            logger.info(f"Milvus中检索到的数据：{res}")

    except Exception as e:
        logger.error(f"商品名称识别节点本地测试失败，原因：{str(e)}", exc_info=True)


# 测试方法运行入口：直接执行该文件即可触发测试
if __name__ == "__main__":
    # 执行本地测试
    test_node_item_name_recognition()