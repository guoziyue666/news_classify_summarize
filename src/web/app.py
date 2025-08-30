import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from starlette.staticfiles import StaticFiles

from src.web.schemas import NewsClassifySummarizeRequest, NewsClassifySummarizeResponse

app = FastAPI()

app.mount('/static', StaticFiles(directory='../../templates'), name='static')


@app.get('/')
async def homepage():
    return FileResponse('../../templates/index.html')


@app.post('/news_classify_summarize')
async def handle_message(message: NewsClassifySummarizeRequest):
    print('-' * 100)
    user_message = message.message
    address = address_alignment(user_message)
    response = NewsClassifySummarizeResponse(
        category=address['category'],
        abstract=address['abstract']
    )
    print(f'检验{response}')
    return response


if __name__ == '__main__':
    uvicorn.run('app:app', host='127.0.0.1', port=8889, reload=True, reload_dirs=['../../templates'], workers=1)
