import asyncio
import json
import time
import sys

from agents.mcp import MCPServerStreamableHttp
from dotenv import load_dotenv

from app.conf.bailian_mcp_config import mcp_config
from app.utils.task_utils import add_done_task,add_running_task


async def mcp_streamable(query):
    mcp_server = MCPServerStreamableHttp(
        name="mcp_server",
        params={
            "url": mcp_config.mcp_base_url,
            "headers": {"Authorization": f"Bearer {mcp_config.api_key}"},
            "timeout": 10,
        },
        cache_tools_list=True,
        max_retry_attempts=3,
    )
    try:
        await mcp_server.connect()
        result = await mcp_server.call_tool(
            tool_name="bailian_web_search",
            arguments={
                "query":query,
                "count":5
            }
        )
        return result

    finally:
        await mcp_server.cleanup()

def node_web_search_mcp(state):
    """
    节点功能，调用外部搜索引擎补充信息
    :param state:
    :return:
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])

    # 1.首先获取查询的问题
    query = state['rewritten_query']

    # 2.调用MCPServer
    result = asyncio.run(mcp_streamable(query))
    result = json.loads(result.content[0].text).get("pages")



    add_done_task(state["session_id"],sys._getframe().f_code.co_name,state["is_stream"])

    print("---node-web-search-mcp处理结束---")
    return {"web_search_docs":result}


if __name__ == '__main__':
    load_dotenv()
    test_state = {
        "session_id":"mcp_01",
        "rewritten_query": "HAK 180 在出厂默认状态下，若想在纸张上只把烫金膜转印到顶部 50 mm–170 mm 的局部区域，应在操作面板上如何设置",
        "is_stream":True
    }

    # 调用 websearch_node 函数
    result_state = node_web_search_mcp(test_state)

    # 验证结果
    print("测试结果:")
    print(f"查询内容: {test_state.get('rewritten_query')}")

    # 输出搜索结果
    search_results = result_state.get('web_search_docs', [])
    print(f"搜索结果数量: {len(search_results)}")
    print("search_results", search_results)