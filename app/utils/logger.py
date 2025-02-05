import logging
import sys

def setup_logger(name: str = "OLLAMA-DP"):
    """设置一个标准的logger

    Args:
        name (str, optional): logger的名称. Defaults to "DeepClaude".

    Returns:
        logging.Logger: 配置好的logger实例
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    # 设置日志级别
    logger.setLevel(logging.INFO)

    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # 设置日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

# 创建一个默认的logger实例
logger = setup_logger()
