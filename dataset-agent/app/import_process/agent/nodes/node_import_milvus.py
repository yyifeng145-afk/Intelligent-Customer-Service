import sys

from pyarrow import DataType

from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.import_process.agent.state import ImportGraphState
from app.core.logger import logger
from app.utils.task_utils import add_done_task, add_running_task


def step2_create_collections(state):
    client = get_milvus_client()
    if not client.has_collection(collection_name=milvus_config.chunks_collection):
        schema = client.create_schema(
            auto_id=True,
            enable_dynamic_fields=True,
        )

        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="part", datatype=DataType.INT64, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        index_params = client.prepare_index_params()

        index_params.add_index(
            field_name="dense_vector",  # Name of the vector field to be indexed
            index_type="HNSW",  # Type of the index to create
            index_name="dense_vector_index",  # Name of the index to create
            metric_type="COSINE",  # Metric type used to measure similarity
            params={
                "M": 16,  # Maximum number of neighbors each node can connect to in the graph
                "efConstruction": 200
                # Number of candidate neighbors considered for connection during index construction
            }  # Index building params
        )

        index_params.add_index(
            field_name="sparse_vector",  # Name of the vector field to be indexed
            index_type="SPARSE_INVERTED_INDEX",  # Type of the index to create
            index_name="sparse_inverted_index",  # Name of the index to create
            metric_type="IP",  # Metric type used to measure similarity
            params={"inverted_index_algo": "DAAT_MAXSCORE"},  # Algorithm used for building and querying the index
        )

        client.create_collection(
            collection_name=milvus_config.chunks_collection,
            schema=schema,
            index_params=index_params
        )
        client.get_load_state(
            collection_name=milvus_config.chunks_collection
        )
    return client


def step3_clear_collection_data(client, item_name):
    client.delete(
        collection_name=milvus_config.chunks_collection,
        filter=f"item_name == '{item_name}'"
    )
    client.load_collection(collection_name=milvus_config.chunks_collection)


def node_import_milvus(state: ImportGraphState)->ImportGraphState:
    # 先拿到chunks 在判断存不存在collection集合 删除collection集合的元素 在将数据插入到collection中 最后更新state

    # 1.记录当前节点日志(同时将当前节点推送到前端)
    node_name = sys._getframe().f_code.co_name
    logger.info(f"当前执行的节点是{node_name},当前的状态是{state}")
    add_running_task(state['task_id'], node_name)

    try:
        # 1首先进行参数校验(chunks)
        chunks = state['chunks']

        # 2创建collection集合
        client = step2_create_collections(state)

        # 3.清空collection集合中的数据
        step3_clear_collection_data(client,chunks[0]['item_name'])

        # 4.插入数据
        client.insert(
            collection_name=milvus_config.chunks_collection,
            data=chunks
        )


    except Exception as e:

        logger.info(f"使用mineru解析出现错误 其中错误是{e}")
        raise RuntimeError(f"结束！！！")
    finally:
        # 5.记录结束节点日志(同时将结束节点推送到前端)
        logger.info(f"当前结束节点的名称是,当前的状态是{state}")
        add_done_task(state['task_id'], node_name)


    return state



if __name__ == '__main__':
    # --- 单元测试 ---
    # 目的：验证 Milvus 导入节点的完整流程，包括连接、创建集合、清理旧数据和插入新数据。
    import sys
    import os
    from dotenv import load_dotenv

    # 加载环境变量 (自动寻找项目根目录的 .env)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    # 构造测试数据
    dim = 1024
    test_state = {
        "task_id": "test_milvus_task",
        "chunks": [
            {
                "content": "Milvus 测试文本 1",
                "title": "测试标题",
                "item_name": "测试项目_Milvus",  # 必须有 item_name，用于幂等清理
                "parent_title":"test.pdf",
                "part":1,
                "file_title": "test.pdf",
                "dense_vector": [0.1] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            }
,
            {
                "content": "Milvus 测试文本 2",
                "title": "测试标题2",
                "item_name": "测试项目_Milvus2",  # 必须有 item_name，用于幂等清理
                "parent_title": "test.pdf2",
                "part": 1,
                "file_title": "test.pdf2",
                "dense_vector": [0.1] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            }
        ]
    }

    print("正在执行 Milvus 导入节点测试...")
    try:
        # 检查必要的环境变量
        if not os.getenv("MILVUS_URL"):
            print("❌ 未设置 MILVUS_URL，无法连接 Milvus")
        elif not os.getenv("CHUNKS_COLLECTION"):
            print("❌ 未设置 CHUNKS_COLLECTION")
        else:
            # 执行节点函数
            result_state = node_import_milvus(test_state)

            # 验证结果
            chunks = result_state.get("chunks", [])
            if chunks and chunks[0].get("chunk_id"):
                print(f"✅ Milvus 导入测试通过，生成 ID: {chunks[0]['chunk_id']}")
            else:
                print("❌ 测试失败：未能获取 chunk_id")

    except Exception as e:
        print(f"❌ 测试失败: {e}")