from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from app.utils.logger import logger
from app.utils.auth import verify_api_key
import os
import aiohttp
import json
import re

# 初始化 FastAPI 应用
app = FastAPI(title="DeepClaude API")

# 配置跨域请求,解决405的问题
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)

# 本地大模型请求，获取<think></think>
LOCAL_URL = os.getenv("LOCAL_URL")
LOCAL_DP_MODEL = os.getenv("LOCAL_DP_MODEL")

# 外部大模型请求，基于<think>中的prompt获取精确结果
OTHER_LLM_API_KEY = os.getenv("OTHER_LLM_API_KEY")
OTHER_API_URL_KEY = os.getenv("OTHER_API_URL_KEY")
OTHER_MODEL = os.getenv("OTHER_MODEL")


# 检查环境变量是否设置
if not all([OTHER_LLM_API_KEY, OTHER_API_URL_KEY, OTHER_MODEL]):
    raise RuntimeError("Missing required environment variables")


# 校验你的key，这个key是方便你向外提供服务
@app.get("/", dependencies=[Depends(verify_api_key)])
async def root():
    """
    根路径，用于测试 API 是否正常运行。
    """
    logger.info("访问了根路径")
    return {"message": "Welcome to DeepClaude API"}


# 获取dp的think
async def get_prompts_from_dp_r1(messages: list, local_url: str = LOCAL_URL, local_model: str = LOCAL_DP_MODEL):

    prompt = ""
    logger.info(messages)
    # 列表推导式，一般是最方便的请求的结果，过滤出所有 role 为 user 的数据
    user_messages = [msg for msg in messages if msg['role'] == 'user']

    # 获取最后一条 user 数据
    if user_messages:
        last_user_message = user_messages[-1]
        prompt = last_user_message['content']
    else:
        print("没有找到 role 为 user 的数据")

    headers = {}
    payload = {
        "model": local_model,
        "prompt": f"{prompt},并显示你自己的思考过程",
        "stream": False  # 启用流式响应
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(local_url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_detail = await response.text()
                    logger.error(f"API 请求失败: {response.status}, 错误信息: {error_detail}")
                    return
                
                # 读取响应内容
                response_text = await response.text()
                logger.info(f"完整响应: {response_text}")

                # 解析 JSON 数据
                response_data = json.loads(response_text)
                response_content = response_data.get("response", "")

                # 使用正则表达式提取 <think> 标签内容
                think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
                think_match = think_pattern.search(response_content)

                if think_match:
                    think_content = think_match.group(1).strip()
                    if think_content:  # 检查内容是否为空
                        logger.info(f"think 标签内容: {think_content}")
                        last_user_message["content"] = think_content
                        messages[-1] = last_user_message
                        return messages
                    else:
                        logger.warning("think 标签存在，但内容为空")
                        return messages
                else:
                    logger.warning("响应中未找到 think 标签")
                    return None

    except aiohttp.ClientConnectorError as e:
        logger.error(f"连接失败: {e}")
    except Exception as e:
        logger.error(f"处理请求时发生错误: {e}")



async def stream_chat_completion(api_url: str, payload: dict, headers: dict):
    """
    使用 aiohttp 发送流式请求，并以 event stream 的形式返回数据。

    :param api_url: API 的 URL
    :param payload: 请求的 payload
    :param headers: 请求的 headers
    :return: 异步生成器，逐行返回流式数据
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_detail = await response.text()
                    logger.error(f"API 请求失败: {response.status}, 错误信息: {error_detail}")
                    yield json.dumps({"error": "API 请求失败", "detail": error_detail})
                    return

                # 以流式方式读取响应
                async for line in response.content:
                    if line:
                        yield line.decode('utf-8')
    except Exception as e:
        logger.error(f"流式请求发生错误: {e}")
        yield json.dumps({"error": "流式请求发生错误", "detail": str(e)})


async def final_request_handle(messages: list, stream: bool = True, key: str = OTHER_LLM_API_KEY):
    """
    封装流式请求逻辑，返回符合 SSE 格式的数据。

    :param messages: 消息列表
    :param stream: 是否启用流式响应
    :param key: API 密钥
    :return: 异步生成器，逐行返回流式数据
    """
    api_url = OTHER_API_URL_KEY  # 使用环境变量中的 API URL
    payload = {
        "model": OTHER_MODEL,  # 使用环境变量中的模型名称
        "messages": messages,
        "stream": stream,  # 启用流式响应
        "max_tokens": 4096,
        "stop": ["null"],
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "response_format": {"type": "text"}
    }

    headers = {
        "Authorization": f"Bearer {key}",  # 使用 f-string 格式化
        "Content-Type": "application/json"
    }

    logger.info(f"Sending request to {api_url} with payload: {payload}")

    # 调用流式方法并处理返回的数据
    async for chunk in stream_chat_completion(api_url, payload, headers):
       
        # 去除 chunk 的前缀 "data: "
        if chunk.startswith("data: "):
            chunk = chunk[len("data: "):].strip()

        # 处理空行
        if not chunk:
            continue

        # 处理 [DONE]
        if chunk == "[DONE]":
            # logger.info("Stream completed with [DONE]")
            yield f"data: {json.dumps({'content': '', 'done': True})}\n\n"
            return

        try:
            # 尝试解析 JSON 数据
            chunk_data = json.loads(chunk)
            # 提取 content 字段
            content = chunk_data.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if content:
                logger.info(f"Received content: {content}")
                yield f"data: {json.dumps({'content': content})}\n\n"  # 符合 SSE 格式
        except json.JSONDecodeError as e:
            # 如果 JSON 解析失败，记录错误并跳过
            # logger.error(f"JSON 解析错误: {e}, chunk: {chunk}")
            continue
        except Exception as e:
            # 其他错误处理
            logger.error(f"处理 chunk 时发生错误: {e}")
            yield f"data: {json.dumps({'error': '处理 chunk 时发生错误', 'detail': str(e)})}\n\n"



@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: Request):
    """
    处理聊天完成请求，返回流式响应。

    请求体格式应与 OpenAI API 保持一致，包含：
    - messages: 消息列表
    - model: 模型名称（可选）
    - stream: 是否使用流式输出（必须为 True）
    """
    try:
        # 1. 获取并验证请求数据
        body = await request.json()
        # Received request body: {'model': 'deepseek-chat', 'messages': [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': 'Test prompt using gpt-3.5-turbo'}], 'temperature': 1, 'max_tokens': 10, 'stream': False}
        logger.info(f"Received request body: {body}")

        messages = body.get("messages")
        if not messages:
            raise HTTPException(status_code=400, detail="messages 不能为空")

        # 2. 检查 API 密钥是否设置
        if not OTHER_LLM_API_KEY:
            raise HTTPException(status_code=500, detail="未设置 API 密钥")

        # 3. 获取think内容    
        messages =  await get_prompts_from_dp_r1(messages)
        logger.info(messages)


        # 4. 返回最终流式响应
        return StreamingResponse(
            final_request_handle(messages, stream=True, key=OTHER_LLM_API_KEY),
            media_type="text/event-stream"
        )
   

    except HTTPException as e:
        logger.error(f"HTTP 错误: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"处理请求时发生错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))
