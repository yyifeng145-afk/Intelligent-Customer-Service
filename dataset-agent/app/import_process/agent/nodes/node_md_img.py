import base64
import os
import re
import sys
from pathlib import Path

from app.clients.minio_utils import get_minio_client
from app.conf.lm_config import lm_config
from app.conf.minio_config import minio_config
from app.core.load_prompt import load_prompt
from app.import_process.agent.state import ImportGraphState
from app.core.logger import logger
from app.lm.lm_utils import get_llm_client
from app.utils.task_utils import add_running_task, add_done_task


def step1_get_params(state):
    md_path = state['md_path']
    if not md_path:
        raise ValueError(f"当前字符串不存在!!!")
    md_path = Path(md_path)
    if not md_path.exists():
        raise FileNotFoundError(f"当前文件不存在！！！")

    #还需要获取md_content(存在两种情况 第一种是来自pdf转成的md直接执行state 第二种是传来的就是md文件还没有将md文件读取的md_content中)
    # md_content = state['md_content']
    if not state['md_content']:
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        state['md_content'] = md_content

    #还需要获取images_dir(这个目录在md_path的上一级的images下)
    images_dir = md_path.parent / 'images'
    return md_path,md_content,images_dir


def get_image_context(md_content, image_name):
    #用于得到md中图片的上下文
    #首先需要定位到图片在md_content中的位置(使用正则表达式)
    reg = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")
    #在md_content可能会有多张图片匹配上
    result = []
    for match in reg.finditer(md_content):  #有很多匹配上的但是我只拿一个
        start,end = match.span()    #获得图片的起始地址和结束地址
        pre_context = md_content[max(start-100,0):start]
        post_context = md_content[end:min(end+100,len(md_content))]
        result.append((pre_context,post_context))

    if result:
        return result[0]






def step2_get_images_infos(md_content, images_dir):
    #便利每一个图片
    images_infos = []
    for image_name in os.listdir(images_dir):
        image_content = get_image_context(md_content, image_name)
        images_infos.append((image_name, str(images_dir / image_name), image_content))
    return images_infos


def step3_get_images_description(images_infos, stem):
    #通过大模型生成图片的描述信息
    sumaries = {}
    for image_name,image_path,image_content in images_infos:
        if image_content is None:
            logger.warning(f"图片 {image_name} 在md中未找到上下文，跳过")
            continue
        #1.初始化大模型
        vm_model = get_llm_client(model=lm_config.lv_model)
        #2.初始化提示词
        prompt = load_prompt(name="image_summary",root_folder=stem,image_content=image_content)
        #3.初始化messages
        # 3. 读取图片转base64(在url不能直接传入图片的地址 需要传入Url 这里使用base64将图片转成字节，再将字节转成string字符串处理)
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        response = vm_model.invoke(messages)
        sumary = response.content.strip()
        sumaries[image_name] = sumary
        logger.info(f"图片总结的结果是{sumary}")
    logger.info(f"图片总结的全部结果是{sumaries}")
    return sumaries





def step4_upload_img_replace(md_content, images_description, images_infos, stem):
    client = get_minio_client()
    bucket_name = minio_config.bucket_name

    image_urls = {}

    for image_name,image_path,image_content in images_infos:

        try:
            # object_name是相对桶内文件的地址
            object_name = f"{minio_config.minio_img_dir}/{stem}/{image_name}"
            client.fput_object(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=image_path,
            )
            image_urls[image_name] = f"{minio_config.endpoint}/{bucket_name}/{minio_config.minio_img_dir[1:]}/{stem}/{image_name}"
            logger.info(f"图片:{image_name}已经上传，对应的url:{image_urls[image_name]}")
        except Exception as e:
            logger.error(f"图片上传失败：{image_name}，错误：{str(e)}")
    #图片上传成功后 将图片的描述和图片的url放到一起(遍历每一个url 取出images_description的图片描述 按照正则表达式进行替换)
    new_md_content = md_content
    for image_name,image_url in image_urls.items():
        summary = images_description.get(image_name)
        reg = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")
        new_md_content = reg.sub(f"![{summary}]({image_url})", new_md_content)
        logger.info(f"图片链接已替换：{image_name} -> {image_url}")

    return new_md_content


def step6_get_new_mdpath(new_md_content, md_path):
    #将new_md_content替换掉
    new_md_path = os.path.splitext(md_path)[0] + "_new.md"
    with open(new_md_path,"w") as f:
        f.write(new_md_content)
    logger.info(f"已经将{new_md_content}写入到{new_md_path}中去了！！！")
    return new_md_path


def node_md_img(state:ImportGraphState)->ImportGraphState:

    # 这个node的作用:将md文件的图片存储在minio下面 同时解释图片的描述
    # 1.记录当前节点日志(同时将当前节点推送到前端)
    node_name = sys._getframe().f_code.co_name
    logger.info(f"当前执行的节点是{node_name},当前的状态是{state}")
    add_running_task(state['task_id'], node_name)

    #2.(需要传入的参数md_path,md_content,image_dir)
    md_path,md_content,images_dir = step1_get_params(state)

    #3.识别图片的内容，进行图片总结(传入md_content,images_dir)->得到(图片名，图片地址，图片上下文)
    images_infos = step2_get_images_infos(md_content,images_dir)

    #4.通过大模型总结图片的内容(输入:images_infos,以及文件的名字 输出:{图片名:图片的描述})
    images_description = step3_get_images_description(images_infos,md_path.stem)

    # 5.将图片上传到minio返回图片的miniourl 同时将md_content中的[]()替换成描述加上url
    new_md_content = step4_upload_img_replace(md_content,images_description,images_infos,md_path.stem)

    # 6.得到新的md_content 需要将这md_content替换掉原来的目录下的md_content 最终返回新的md_path
    new_md_path_str = step6_get_new_mdpath(new_md_content,md_path)
    state['md_content'] = new_md_content
    state['md_path'] = new_md_path_str

    logger.info(f"当前执行的节点是{node_name},当前的状态是{state}")
    add_done_task(state['task_id'], node_name)
    return state


if __name__ == "__main__":
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"本地测试 - 项目根目录: {PROJECT_ROOT}")

    # 测试MD文件路径（需要手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在: {test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程参数
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": ""
        }
        logger.info("开始本地测试 - MD图片处理流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态: {result_state}")