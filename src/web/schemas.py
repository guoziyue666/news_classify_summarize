from pydantic import Field, BaseModel


class NewsClassifySummarizeRequest(BaseModel):
    content: str | None = Field(description='新闻内容')


class NewsClassifySummarizeResponse(BaseModel):
    category: str | None = Field(description='新闻分类')
    summary: str | None = Field(description='新闻摘要')
