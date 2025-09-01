import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from configuration import config
from web.app import summary_predictor


class Predictor:
    def __init__(self, model, tokenizer, device):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def predict(self, text: str | list):
        is_str = isinstance(text, str)
        if is_str:
            text = [text]
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1).tolist()
        results = [self.model.config.id2label[prediction] for prediction in predictions]
        if is_str:
            return results[0]
        return results


class SummaryPredictor:
    def __init__(self, model, tokenizer, device):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def predict(self, text: str | list):
        pass


if __name__ == '__main__':
    # 1. 创建模型, 分词器和设备
    model = AutoModelForSequenceClassification.from_pretrained(config.CHECKPOINT_DIR / 'classify' / 'best')
    tokenizer = AutoTokenizer.from_pretrained(config.CHECKPOINT_DIR / 'classify' / 'best')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2.创建预测器并预测
    text_one = "4月13日，全球首个人形机器人半程马拉松赛将在北京亦庄举行。赛事由北京市体育局、北京市经济和信息化局、中央广播电视总台北京总站、北京经济技术开发区管理委员会等单位联合主办。"
    text_list = [
        "4月13日，全球首个人形机器人半程马拉松赛将在北京亦庄举行。赛事由北京市体育局、北京市经济和信息化局、中央广播电视总台北京总站、北京经济技术开发区管理委员会等单位联合主办。随着比赛临近，参赛机器人在跑道上进行了首次路测。路测期间，机器人们表现如何？",
        "据日本鹿儿岛地方气象台消息，当地时间3日13时49分左右，位于鹿儿岛县和宫崎县交界地区雾岛山的新燃岳火山喷发，火山灰柱最大高度达5000米。",
        "自7月7日开始，在中国人民抗日战争纪念馆举办“为了民族解放与世界和平——纪念中国人民抗日战争暨世界反法西斯战争胜利80周年主题展览”，展出照片1525张、文物3237件。主题展览将作为基本陈列长期展出。",
        "近日，印度中央邦博帕尔市一座铁路立交桥引发全球关注。这座桥的致命缺陷并非偷工减料或结构坍塌，而是惊现90 度直角转弯，司机必须紧急刹车才能通过，被网友调侃为 “现实版神庙逃亡赛道”。更讽刺的是，这座桥至今尚未通车，却已因设计失误导致 7 名工程师停职、两家建筑公司被列入黑名单。",
    ]

    # predictor = Predictor(model, tokenizer, device)
    # print(predictor.predict(text_one))
    # print(predictor.predict(text_list))
    summary_predictor = SummaryPredictor(model, tokenizer, device)
    print(summary_predictor.predict(text_one))
    print(summary_predictor.predict(text_list))
