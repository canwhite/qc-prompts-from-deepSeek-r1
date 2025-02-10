## qc-prompts-from-deepSeek-r1 
Get the think tag from the deepseek-r1 deployed locally through ollama,   
extend the prompts, and then make requests to other models in a proprietary direction 

### pre
[ollama模型本地化]([https://mp.weixin.qq.com/s/UJjOK8Tkzp8q7_Bs7b4uzg])

### env

Supplement needed key to .env

### run
```
pip install poetry  
poetry shell
poetry install  
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

