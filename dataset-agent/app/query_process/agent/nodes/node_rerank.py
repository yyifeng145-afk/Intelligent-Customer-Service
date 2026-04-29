import time
import sys
from locale import normalize

from markdown_it.rules_inline import entity

from app.lm.reranker_utils import get_reranker_model
from app.utils.task_utils import add_running_task, add_done_task

DEFAULT_MAX_TOPK = 10
DEFAULT_MIN_TOPK = 1
DEFAULT_TOPK_REL = 0.25
DEFAULT_TOPK_ABS = 0.5



def step2_rerank_doc_list(doc_list, state):
    #将问题与答案进行组装送给rerank模型
    rewritten_query = state['rewritten_query']
    #获取所有答案
    text_list = [doc["text"] for doc in doc_list]
    #将问题与答案放在一起


    questions_query = [(text,rewritten_query) for text in text_list]
    rerank = get_reranker_model()
    scores = rerank.compute_score(questions_query,normalize=True)

    for score,doc in zip(scores,doc_list):
        doc["score"] = score

    doc_list.sort(key=lambda x:x["score"],reverse=True)
    return doc_list

    #这个返回的数据类型  [{"text": text, "id":None,"title": title, "url": url, "source": "web","score":score}]


def step3_topk_fangduanya(score_list, logger=None):
    #实现防断崖取值(设置最大topk值 最小topk值 绝对topk 相对topk )
    #其中取值是在min_topk到max_topk之间
    # 还需要判断相邻的值之间是否大于绝对topk 若大于肯定不相关
    # 判断相邻的值之间((score_list[i]-score_list[i+1])/score_list[i])是否大于相关topk，若大于肯定不相关
    topk = min(len(score_list),DEFAULT_MAX_TOPK)
    for i in range(DEFAULT_MIN_TOPK-1,topk-1):
        score_1 = score_list[i].get("score",0.0)
        score_2 = score_list[i+1].get("score",0.0)
        if((score_1-score_2)>DEFAULT_TOPK_ABS or (score_1-score_2)/(score_1 + 1e-5)>DEFAULT_TOPK_REL):
            topk = i
            logger.info(f"当前index{i}与index{i+1}发生了断崖!!!")
            break

    topk_list = score_list[:topk+1]
    return topk_list

def node_rerank(state):
    """
        对粗排后的结果与MCP联网搜索的结果进行细排
        1、将这两个数组调整好类型放到一起
        2、通过ReRank进行排序(将问题和答案放在一起进行rerank排序)

    """

    # 记录任务开始
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    reranked_docs = state["rrf_chunks"]
    # reranked_docs的数据格式 [{id,distance:0.8,entity:{chunk_id,content,title..}}}]

    web_search_docs = state["web_search_docs"]
    # web_search_docs的格式 [{title,snippet,url}]
    doc_list = []
    for doc in reranked_docs:
        entity = doc.get("entity")
        text = entity.get("content")
        id = doc.get("id")
        title = entity.get("title")
        doc_list.append({"text": text, "id": id, "title": title, "url": None, "source": "local"})

    for doc in web_search_docs:
        title = doc.get("title")
        text = doc.get("snippet")
        url = doc.get("url")
        doc_list.append({"text": text, "id":None,"title": title, "url": url, "source": "web"})

    score_list = step2_rerank_doc_list(doc_list,state)

    topk_list = step3_topk_fangduanya(score_list)

    state["reranked_docs"] = topk_list

    # 记录任务结束
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))

    return state