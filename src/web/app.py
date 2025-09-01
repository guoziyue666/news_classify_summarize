import logging

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from starlette.staticfiles import StaticFiles
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from configuration import config
from runner.predict import Predictor, SummaryPredictor
from src.web.schemas import NewsClassifySummarizeRequest, NewsClassifySummarizeResponse
from web.service import NewsService

app = FastAPI()
app.mount('/static', StaticFiles(directory='../../templates'), name='static')

# 创建模型，分词器，设备
model = AutoModelForSequenceClassification.from_pretrained(config.CHECKPOINT_DIR / 'classify' / 'best')
tokenizer = AutoTokenizer.from_pretrained(config.CHECKPOINT_DIR / 'classify' / 'best')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建分类和摘要预测器
predictor = Predictor(model, tokenizer, device)
summary_predictor = SummaryPredictor(model, tokenizer, device)
service = NewsService(predictor, summary_predictor)

# 最简单的控制台日志配置
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 设置日志格式
    handlers=[logging.StreamHandler()]  # 输出到控制台
)


@app.get('/')
async def homepage():
    return FileResponse('../../templates/index.html')


@app.post('/news_classify_summarize')
async def handle_message(content: NewsClassifySummarizeRequest):
    print('-' * 100)
    user_content = content.content
    classify, summary = service.predict(user_content)
    # address = address_alignment(user_content)
    # response = NewsClassifySummarizeResponse(
    #     category=address['category'],
    #     abstract=address['abstract']
    # )
    response = NewsClassifySummarizeResponse(
        category=classify,
        summary=summary
    )
    logging.info(f'请求输入：{user_content}')
    logging.info(f'响应输出：{response}')
    return response


if __name__ == '__main__':
    uvicorn.run('app:app', host='0.0.0.0', port=8889, reload=True, reload_dirs=['../../templates'], workers=1)
