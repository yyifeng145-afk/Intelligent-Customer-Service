import re
import time
import sys

from sympy.abc import delta
from sympy.multipledispatch.dispatcher import source

from app.clients.mongo_history_utils import save_chat_message
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.lm.lm_utils import get_llm_client
from app.utils.sse_utils import push_to_session, SSEEvent
from app.utils.task_utils import add_running_task, add_done_task, set_task_result

MAX_LENGTH = 2400

def step1_get_answer_params(state):
    #调用llm生成答案
    # reranked_docs的数据格式     [{"text": text, "id":None,"title": title, "url": url, "source": "web","score":score}]
    reranked_docs = state["reranked_docs"]
    history = state["history"]
    item_names = state["item_names"]
    question = state["rewritten_query"]

    docs = []
    length_context = 0

    for i,doc in enumerate(reranked_docs,start=1):
        text = doc["text"]
        title = doc["title"]
        source = doc["source"]
        context = f"[{i}][{text}][{title}][{source}]"
        if len(context) + length_context > MAX_LENGTH:
            break
        docs.append(context)



    context_last = "\n\n".join(docs)

    history_last = ""
    for i,his in enumerate(history,start=1):
        text = his["text"]
        role = his["role"]
        if role == "user":
            history_currant = f"当前的用户是{role}，文本内容是{text}"
        else:
            history_currant = f"当前的用户是{role}，文本内容是{text}"

        if len(history_last) > MAX_LENGTH:
            break
        history_last = history_last + history_currant
    history_last = "\n\n".join(history_last)

    item_name_str = ",".join(item_names)

    prompt = load_prompt("answer_out",context = context_last,history = history_last,item_names = item_name_str,question = question)

    return prompt


def step2_get_answer(state, prompt):
    #得到Prompt 判断当前是流式输出还是非流式输出 返回答案
    is_stream = state["is_stream"]
    llm = get_llm_client()
    answer = ""
    if is_stream:
        for chunk in llm.stream(prompt):
            delta = chunk.content
            push_to_session(state["session_id"],SSEEvent.DELTA,{delta:delta})
            answer = answer+delta
    else:
        chunk = llm.invoke(prompt)
        answer = chunk.content
        set_task_result(state["session_id"],"answer",answer)
    state["answer"] = answer
    return answer


def step3_get_image_url(state):
    #需要将mcp得到的url地址与text中的url地址进行返回
    #reranked数据类型   [{"text": text, "id":None,"title": title, "url": url, "source": "web","score":score}]
    reranked_docs = state["reranked_docs"]
    image_reg = re.compile(r"!\[.*?\]\((.*?)\)")
    images_save = []
    for doc in reranked_docs:
        url = doc["url"]
        if url.endswith((".jpg", ".jpeg", ".png")):
            if url not in images_save:
                images_save.append(url)

        text = doc["text"]
        match_urls = image_reg.findall(text)
        for match_url in match_urls:
            if match_url not in images_save:
                images_save.append(match_url)

    return images_save                  #将url以及text中的url保存起来


def step4_save_history(state):
    session_id = state["session_id"]
    rewritten_query = state["rewritten_query"]
    item_names = state["item_names"]
    answer = state["answer"]

    if rewritten_query:
        save_chat_message(  session_id = session_id,
                            role = "user",
                            text = rewritten_query,
                            rewritten_query = rewritten_query,
                            item_names= item_names)
    if answer:
        save_chat_message(  session_id = session_id,
                            role = "assistant",
                            text = answer,
                            rewritten_query = rewritten_query,
                            item_names= item_names)

    logger.info(f"完成了本次的历史消息存储!!!")


def node_answer_output(state):
    """
        调用llm对答案进行输出
        需要参考{context}   {history}   {item_names}    {question}
    """
    print("---node_answer_output 节点处理开始---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    
    
    prompt = step1_get_answer_params(state)
    
    answer = step2_get_answer(state,prompt)

    image_urls = step3_get_image_url(state)

    # 将得到的图片传到前端
    if image_urls:
        push_to_session(state["session_id"],SSEEvent.FINAL,{"answer":answer,
                                                            "image_urls":image_urls,
                                                            "status":"completed"})

    step4_save_history(state)


    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    print("---node_answer_output 节点处理结束---")
    return {"answer": state}