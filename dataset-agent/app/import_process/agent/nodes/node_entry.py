import os.path
import sys

from app.import_process.agent.state import ImportGraphState
from app.core.logger import logger
from app.utils.task_utils import add_running_task, add_done_task

"""
1.进入节点的日志输出(同时将当前节点推送给前端)
2.确定入口文件
3.判断当前文件格式
4.将state[pdf|md]=True
5.从文件名中提取file_title 作为后续的元数据
6.结束节点的日志输出(将当前节点推送给前端)
"""

def node_entry(state:ImportGraphState) -> ImportGraphState:
    # 1.当前节点的日志输出 同时将当前节点推送给前端
    node_name = sys._getframe().f_code.co_name
    logger.info(f"当前入口节点是{node_name},现在的状态是{state}")
    add_running_task(state['task_id'],node_name)

    # 2.获取入口文件
    if state['local_file_path'] is None:
        logger.error(f"{node_name}检查没有输入文件,请输入文件!!!")
        return state

    if state['local_file_path']:
        if state['local_file_path'].endswith('.pdf'):
            state['is_pdf_read_enabled'] = True
            state['pdf_path'] = state['local_file_path']
        elif state['local_file_path'].endswith('.md'):
            state['is_md_read_enabled'] = True
            state['md_path'] = state['local_file_path']
        else:
            logger.error(f"这不是需要处理的文件类型")

    # 3.从文件名中提取file_title
    file_title = os.path.splitext(os.path.basename(state['local_file_path']))[0]
    state['file_title'] = file_title

    # 4.结束节点的日志输出
    logger.info(f"当前结束节点是{node_name},现在的状态是{state}")
    add_done_task(state['task_id'],node_name)

    return state