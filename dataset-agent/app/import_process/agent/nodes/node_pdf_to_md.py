import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

from app.conf.mineru_config import mineru_config
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.core.logger import logger
from app.utils.task_utils import add_running_task, add_done_task
import requests


def step1_validate_params(state):
    pdf_path = state['pdf_path']
    local_dir = state['local_dir']

    if pdf_path is None:
        logger.info(f"当前没有输入文件，请重新上传")
        raise ValueError(f"当前没有输入文件，请重新上传")
    if local_dir is None:
        logger.info(f"当前没有输出目录 现在在创建一个新的输出文件目录")
        local_dir = str(PROJECT_ROOT/"output")

    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)

    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"当前文件不存在")
    if not local_dir_obj.exists():
        logger.info(f"当前目录不存在，现在正在重新创建")
        local_dir_obj.mkdir(parents=True,exist_ok=True)

    return pdf_path_obj,local_dir_obj


def step2_get_download_url(pdf_path_obj):
    token = mineru_config.api_key
    url = f"{mineru_config.base_url}/file-urls/batch"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "files": [
            {"name": f"{pdf_path_obj.name}"}
        ],
        "model_version": "vlm"
    }
    response = requests.post(url, headers=header, json=data)
    result = response.json()
    if response.status_code != 200:
        logger.error(f"[step2] 请求失败，状态码: {response.status_code}，返回内容: {result}")
        raise FileNotFoundError(f"文件解析失败，状态码: {response.status_code}，原因: {result}")
    if result["code"] != 0:
        raise RuntimeError(f"请求minerU解析失败，检查地址是否正确???")
    batch_id = result["data"]["batch_id"]
    urls = result["data"]["file_urls"][0]

    # put上传文件需要严进严出(将文件读取到指定的url中) 转存到文件存储器比较严格 审核不过会报错 例如使用代理
    http_session = requests.Session()
    http_session.trust_env = False
    try:
       with open(pdf_path_obj, 'rb') as f:
           file_data = f.read()
           res_upload = http_session.put(urls, data=file_data)
    except Exception as e:
        if res_upload.status_code != 200:
            raise RuntimeError(f"文件上传失败!!!")
    finally:
        http_session.close()

    # 批量获取解析结果(并不是直接就能获取到的)
    url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"

    max_wait_seconds = 600
    interval_seconds = 3
    start_time = time.time()

    while(True):
        if time.time()-start_time>max_wait_seconds:
            return RuntimeError(f"超出等待时间！！！")
        # 获取zip地址
        res = requests.get(url, headers=header)
        #通过状态码进行判断当前是否继续执行
        if res.status_code!=200:
            if 500<=res.status_code<600:
                time.sleep(interval_seconds)
                continue
            raise RuntimeError(f"当前的状态码是{res.status_code},请求解析失败!!!")

        if res.json()['data']['extract_result'][0]['state'] == 'done':
            full_zip_url = res.json()['data']['extract_result'][0]['full_zip_url']
            return full_zip_url
        else:
            time.sleep(interval_seconds)
    # 这是claude写的代码(因为直接获取zip地址可能不会直接得到 需要轮询600s得到最终的结果 其中时间间隔是3s 先判断当前的状态码是不是500到600之间 若不是直接抛异常结束 若是在判断当前的提取结果的state是否为done 若是则直接返回下载的zip的最终地址)
    # while time.time() < deadline:
    #     res = requests.get(url, headers=header)
    #     result = res.json()
    #
    #     if res.status_code != 200 or result.get("code") != 0:
    #         raise RuntimeError(f"获取解析结果失败: {result.get('msg', '未知错误')}")
    #
    #     extract_result = result.get("data", {}).get("extract_result", [])
    #
    #     if extract_result:
    #         file_info = extract_result[0]
    #         state = file_info.get("state", "")
    #         logger.info(f"[MinerU] 当前任务状态: {state}")
    #
    #         if state == "done":
    #             zip_url = file_info.get("full_zip_url") or file_info.get("zip_url")
    #             if not zip_url:
    #                 raise RuntimeError(f"[MinerU] 任务完成但未找到下载链接")
    #             logger.info(f"[MinerU] 获取到zip下载链接: {zip_url}")
    #             return zip_url
    #
    #         elif state == "failed":
    #             raise RuntimeError(f"[MinerU] PDF解析失败: {file_info.get('err_msg', '未知错误')}")
    #
    #     logger.info(f"[MinerU] 任务尚未完成，{interval_seconds}s后重试...")
    #     time.sleep(interval_seconds)
    #
    # raise TimeoutError(f"[MinerU] 轮询超时，已等待{max_wait_seconds}s，batch_id={batch_id}")


def step3_downlaod_extract_pdf(download_zip_url, local_dir_obj,stem):
    #1.首先下载zip文件到本地
    zip_path = local_dir_obj/f"{stem}.zip"
    resp = requests.get(download_zip_url,stream=True)

    if resp.status_code !=200:
        raise RuntimeError(f"下载zip文件失败请确认下载地址是否正确？？？")

    with open(zip_path,'wb') as f:
        f.write(resp.content)

    #2解压zip文件到本地
    unzip_path = local_dir_obj/stem
    if unzip_path.exists():
        shutil.rmtree(unzip_path)
        logger.info(f"[step3] 已删除旧目录: {unzip_path}")
    unzip_path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path,'r') as zip_ref:
        zip_ref.extractall(unzip_path)

    #3.找到解压目录下的md文件并返回
    md_path = local_dir_obj/f"{stem}"/f"{stem}.md"
    if not md_path.exists():
        md_list = list(local_dir_obj.rglob("*.md"))
        if not md_list:
            raise FileNotFoundError(f"目录下没有md文件！！！")
        md_path = md_list[0]
        new_md_path = md_path.parent / f"{stem}.md"
        md_path.rename(new_md_path)
        md_path = new_md_path
    final_md_path = str(md_path.resolve())
    return final_md_path

def node_pdf_to_md(state:ImportGraphState)->ImportGraphState:


    logger.info(f">>>[Stub] 执行节点:{sys._getframe().f_code.co_name}")

    # 1.记录当前节点日志(同时将当前节点推送到前端)
    node_name = sys._getframe().f_code.co_name
    logger.info(f"当前执行的节点是{node_name},当前的状态是{state}")
    add_running_task(state['task_id'],node_name)


    try:
        # 2.进行参数校验(判断pdf路径是否存在 在判断磁盘中的pdf是否存在 目录路径是否存在 若不存在则创建)
        pdf_path_obj, local_dir_obj = step1_validate_params(state)

        # 3.调用minerU获取pdf转md的下载链接
        download_zip_url = step2_get_download_url(pdf_path_obj)

        # 4.下载并解压zip文件(先根据zip地址将zip文件下载到指定的目录下，再将zip文件解压到指定的目录)
        md_path = step3_downlaod_extract_pdf(download_zip_url, local_dir_obj, pdf_path_obj.stem)

        # 5.对md_path进行赋值 再将md_content赋值
        state['md_path'] = md_path
        state['local_dir'] = str(local_dir_obj)
        with open(md_path, 'rb') as f:
            state['md_content'] = f.read()
    except Exception as e:
        logger.info(f"使用mineru解析出现错误 其中错误是{e}")
        raise RuntimeError(f"结束！！！")
    finally:
        # 5.记录结束节点日志(同时将结束节点推送到前端)
        logger.info(f"当前结束节点的名称是{node_name},当前的状态是{state}")
        add_done_task(state['task_id'],node_name)

    return state



if __name__ == "__main__":
    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")
