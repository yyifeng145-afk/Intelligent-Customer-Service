import sys

from app.import_process.agent.state import ImportGraphState
from app.core.logger import logger
from app.lm.embedding_utils import generate_embeddings
from app.utils.task_utils import add_running_task, add_done_task


def step1_get_chunks(state):
    chunks = state['chunks']
    return chunks


def node_bge_embedding(state: ImportGraphState)->ImportGraphState:
    # 具体的流程：首先得到chunks 分批次处理chunks 当使用嵌入模型进行embedding时，没有主体可能导致后续的查询匹配不上 这里我们采用拼接字符串（f"商品名:{item_name},内容介绍:{content}"）来更好地识别
    # 得到的结果先存到currant_text 在调用嵌入模型进行embedding 嵌入模型返回的result有两种向量(dense_vector以及sparse_vector) 再将dense_vector存到chunk中



    # 1. 进入的日志和任务状态的配置
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}]开始执行了！现在的状态为：{state}")
    add_running_task(state['task_id'], function_name)

    try:
        # 1.首先获取到需要embedding的chunks
        chunks = step1_get_chunks(state)

        # 2.获取embedding模型 调用embedding批量生成向量
        final_chunks = []
        batch_size = 5
        for i in range(0,len(chunks),batch_size):
            batch_chunks = chunks[i:i+batch_size]

            currant_text = []

            for batch_chunk in batch_chunks:
                batch_chunk_title_name = batch_chunk.get('title_name')
                batch_chunk_content = batch_chunk.get('content')
                batch_chunk_text = f"标题名:{batch_chunk_title_name},文章内容是:{batch_chunk_content}"
                currant_text.append(batch_chunk_text)
            result = generate_embeddings(currant_text)
            for i,item in enumerate(batch_chunks):
                chunk_item = item.copy()
                chunk_item['dense_vector'] = result['dense'][i]
                chunk_item['sparse_vector'] = result['sparse'][i]
                final_chunks.append(chunk_item)
        # 3.将生成的chunks回传到state中
        state['chunks'] = final_chunks

        logger.info(f">>> [{function_name}]开始执行了！现在的状态为：{state}")
        add_done_task(state['task_id'], function_name)
    except Exception as e:
        logger.info(f"BGE向量化失败!!!")
    return state