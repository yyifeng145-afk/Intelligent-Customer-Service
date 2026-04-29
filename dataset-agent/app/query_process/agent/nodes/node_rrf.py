import time
import sys
from app.core.logger import logger
from onnxruntime.transformers.models.gpt2.gpt2_parity import score
from sympy.codegen.fnodes import cmplx

from app.utils.task_utils import add_running_task, add_done_task


def step2_cu_merge(cu_documents):
    score_dict = {}
    chunk_dict = {}
    #便利向量检索的结果和假设性搜索的结果 按照得分进行排名
    for vector_list,weight in cu_documents:     #分别遍历向量搜索和Hyde假设性搜索得到的chunks
        for rank,vector in enumerate(vector_list,start=1):      #vector的格式: 【{id: 实体的主键,distance:得分 0.8,entity:{chunk_id,content,title..}} ,{} ,{}】
            id = vector.get("id")
            score_dict[id] = score_dict.get(id,0.0)+1/(60+rank)*weight
            chunk_dict[id] = vector
    #将得到的score_dict和chunk_dict进行合并后排序
    merged_list = []
    for id,score in score_dict.items():         #这里的score_list是一个字典，使用items可以及拿到key也能拿到value
        chunk = chunk_dict.get(id)
        merged_list.append((chunk,score))
    merged_list.sort(key=lambda x:x[1],reverse=True)
    merged_list = merged_list[:5]
    last_result = [chunk for chunk,score in merged_list]        #我最终只想要chunk 不需要score
    return last_result


def node_rrf(state):

    """
        这段代码将向量搜索以及HYDE搜索得到的结果进行粗排
        其中他们的数据格式是: [[向量1]，[向量2]] [向量]的格式是:[{id:id,distance:0.8,entity:{}}]
        构造一个数组放置[vector,Hyde]
        遍历数组得到vector 计算vector的每一个id的得分 同时得到每一个chunk
        再将chunk以及score放到一起，进行排名后取topk个返回
    """

    # 记录任务开始
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 1获取粗排的对象
    embedding_chunks = state["embedding_chunks"]
    hyde_embedding_chunks = state["hyde_embedding_chunks"]

    # embedding_chunks的数据类型 [[向量1],[向量2]]    [向量1]  =》 【{id: 实体的主键,distance:得分 0.8,entity:{chunk_id,content,title..}} ,{} ,{}】
    # hyde_embedding_chunks的数据类型

    cu_documents = [(embedding_chunks,1.0),
                    (hyde_embedding_chunks,1.0)]

    # 2对粗排的对象进行合并
    rrf_chunks = step2_cu_merge(cu_documents)

    state["rrf_chunks"] = rrf_chunks

    # 记录任务结束
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))

    return state


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_rrf 本地测试")
    print("=" * 50)

    mock_state = {
        "session_id": "test_rrf_session",
        "is_stream": False,
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤是什么？",
        "item_names": ["HAK 180 烫金机"]
    }

    try:
        from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
        from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde

        emb_res = node_search_embedding(mock_state)
        hyde_res = node_search_embedding_hyde(mock_state)
        mock_state['embedding_chunks'] = emb_res.get("embedding_chunks") or []
        mock_state['hyde_embedding_chunks'] = hyde_res.get("hyde_embedding_chunks") or []

        result = node_rrf(mock_state)
        rrf_chunks = result.get("rrf_chunks", [])

        emb_cnt = len(mock_state.get("embedding_chunks") or [])
        hyde_cnt = len(mock_state.get("hyde_embedding_chunks") or [])

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"输入数量: Embedding={emb_cnt}, HyDE={hyde_cnt}")
        print(f"输出数量: {len(rrf_chunks)}")
        print("-" * 30)

        print("最终排名:")
        for i, doc in enumerate(rrf_chunks, 1):
            doc_id = doc.get("chunk_id") or doc.get("id")
            content = (doc.get("content") or "")[:20]
            print(f"Rank {i}: ID={doc_id}, Content={content}...")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")