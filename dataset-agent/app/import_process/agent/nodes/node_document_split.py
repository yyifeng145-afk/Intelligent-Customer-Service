import re
import sys

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.import_process.agent.state import ImportGraphState
from app.core.logger import logger
from app.utils.task_utils import add_running_task, add_done_task

"""
1.校验参数
2.粗粒度切割(按照标题进行切割)
3.特殊情况(没有标题添加标题)
4.细粒度切割(对粗粒度切割后的文档进行二次切割 对长的chunks进行切割 对短的chunks进行合并)
5.chunks的备份
6.更新state
"""
MAX_LENGTH = 2000
MIN_LENMGTH = 500

def step1_get_params(state):
    md_content = state['md_content']
    if not md_content:
        logger.info(f"文档内容为空,{md_content}不存在！！！")
        raise FileNotFoundError(f"{md_content}不存在!!!")
    md_content.replace('\r\n','\n').replace('\r','\n')

    file_title = state['file_title']
    return md_content,file_title


def step2_cu_split(md_content, file_title):
    # 对md_content进行切分 先按照标题进行切分 将切分后的md_content存到sections中 sections包含(content,title,file_title)
    line_count = 0
    title_count = 0
    is_code_block = False
    currant_lines = []
    currant_title = None
    sections = []

    #1.首先将文档按照\n进行划分 ->将md_content划分成很多行 在进行判断是不是代码块以及是不是标题(不是标题直接将line塞到currant_lines中
    # 如果是标题 需要先判断当前是不是第一个标题 只有第一个标题执行完成才能塞进最终的列表 就是先从第二个标题开始 因为执行到第一个标题时，第一个标题还没有塞进currant_lines中)
    for line in md_content.split('\n'):
        line = line.strip()
        line_count = line_count + 1

        # 判断是不是代码块
        if line.startswith('```') or line.startswith('~~~'):
            is_code_block = not is_code_block
            currant_lines.append(line)
            continue
        # 判断是不是标题->需要判断先正则判断当前标题能不能匹配上 在判断当前是不是代码块
        is_title = bool(re.compile(r'^#{1,6}\s+.+').match(line)) and not is_code_block
        if is_title:
            if currant_title:
                sections.append({"title":currant_title,
                                 "content": "\n".join(currant_lines),
                                 "file_title":file_title
                                 }
                                )
            currant_title = line
            currant_lines = [currant_title]
            title_count = title_count +1
        else:
            currant_lines.append(line)

    # 因为在第一次执行到标题的时候，当前标题的内容没有完全识别 不能直接将第一次识别到的标题以及内容直接插入到sections中 只有第二次检测到标题时，第一次检测的内容才可以督导section中 但是这样会导致最后一次
    # 最后一次没有被识别到 因此需要在for循环结束的时候将当前的title content读取到sections
    # 最终返回每一个段落的sections以及标题的数量以及md文档的行数
    if currant_title:
        sections.append({"title":currant_title,
                         "content": "\n".join(currant_lines),
                         "file_title":file_title})

    return sections,title_count,line_count


def merge_content(split_sections, MIN_LENMGTH):
    merged_sections = []
    pre_section = None
    # 使用双指针
    for section in split_sections:
        if pre_section is None:
            pre_section = section
            continue
        if section['parent_title'] == pre_section['parent_title'] and len(pre_section['content'])<MIN_LENMGTH:
            pre_section['content'] = pre_section['content']+"\n\n"+section['content']
            pre_section['part'] = section['part']
        else:
            merged_sections.append(pre_section)
            pre_section = section

    if pre_section:
        merged_sections.append(pre_section)

    return merged_sections







def step3_fine_split(sections, MAX_LENGTH, MIN_LENMGTH):
    #对过长的文档进行划分 对过长文档进行二次划分后变短进行合并
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_LENGTH,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )

    split_sections = []

    # 对于每一个段落 便利每一个段落下的section
    for section in sections:
        content = section['content']

        if len(content)<=MAX_LENGTH:
            split_sections.append(section)

        if len(content)>MAX_LENGTH:
            # 大于最大长度对content进行划分 划分后的content包含多个subcontent
            sub_content = splitter.split_text(content)
            for i,chunk in enumerate(sub_content,start=1):
                split_sections.append({"title":f"{section['title']}_{i}",
                                       "content":chunk,
                                       "file_title":section['file_title'],
                                       "parent_title":section['title'],
                                       "part":i})

    #对小的进行合并
    merged_sections = merge_content(split_sections,MIN_LENMGTH)












def node_document_split(state:ImportGraphState)->ImportGraphState:
    # 这段代码的作用是将文档进行划分
    #1.进行参数校验
    #2.将md_content进行切片
    #3.对切片后的大文本进行二次切片 对二次切片的小文本进行合并

    # 1.记录当前节点日志(同时将当前节点推送到前端)
    node_name = sys._getframe().f_code.co_name
    logger.info(f"当前执行的节点是{node_name},当前的状态是{state}")
    add_running_task(state['task_id'], node_name)

    #2.对参数进行校验(得到md_content的内容 以及文件的标题名(hak180产品安全手册)
    md_content,file_title = step1_get_params(state)

    #3.对md_content进行分割(划分后得到)
    sections,title_count,line_count = step2_cu_split(md_content,file_title)

    #4.对按照标题切割后的文档进行二次划分 过长进行划分 过短进行合并
    sections = step3_fine_split(sections,MAX_LENGTH,MIN_LENMGTH)



    # 7.记录当前节点日志(同时将当前节点推送到前端)
    logger.info(f"当前执行的节点是{node_name},当前的状态是{state}")
    add_done_task(state['task_id'], node_name)





    return state