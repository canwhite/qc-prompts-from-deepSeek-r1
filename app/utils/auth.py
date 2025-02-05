from fastapi import HTTPException, Header
from typing import Optional
import os
from dotenv import load_dotenv
from app.utils.logger import logger

# 加载 .env 文件
load_dotenv()

# 获取环境变量
ALLOW_API_KEY = os.getenv("ALLOW_API_KEY")
if not ALLOW_API_KEY:
    raise ValueError("ALLOW_API_KEY environment variable is not set")


async def verify_api_key(authorization: Optional[str] = Header(None)) -> None:
    """验证API密钥

    Args:
        authorization (Optional[str], optional): Authorization header中的API密钥. Defaults to Header(None).

    Raises:
        HTTPException: 当Authorization header缺失或API密钥无效时抛出401错误
    """
    if authorization is None:
        logger.warning("请求缺少Authorization header")
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header"
        )
    
    api_key = authorization.replace("Bearer ", "").strip()
    if api_key != ALLOW_API_KEY:
        logger.warning(f"无效的API密钥: {api_key}")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    logger.info("API密钥验证通过")
