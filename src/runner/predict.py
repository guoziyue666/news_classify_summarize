import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSequenceClassification
from transformers.modeling_outputs import BaseModelOutput
from configuration import config


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
        self.model.eval()
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

    def beam_search(self, input_ids, attention_mask, max_length=28, num_beams=6):
        """
        自定义束搜索实现
        """
        batch_size = input_ids.size(0)
        # 初始化解码器输入（开始标记）
        decoder_input_ids = torch.ones((batch_size, 1), dtype=torch.long,
                                       device=self.device) * self.tokenizer.bos_token_id

        # 编码器前向传播
        encoder_outputs = self.model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask)

        # 为批量处理准备encoder_outputs和attention_mask
        batched_encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_outputs.last_hidden_state.repeat_interleave(num_beams, dim=0)
        )
        batched_attention_mask = attention_mask.repeat_interleave(num_beams, dim=0)

        # 初始化束 (每个输入样本对应num_beams个束)
        beams = []
        for i in range(batch_size):
            for _ in range(num_beams):
                beams.append({
                    'sequence': decoder_input_ids[i].clone(),
                    'score': 0.0,
                    'is_finished': False,
                    'input_id': i  # 记录属于哪个输入样本
                })

        # 逐步生成
        for step in range(max_length):
            # 收集所有活跃束的输入
            active_beams = [beam for beam in beams if not beam['is_finished']]
            if not active_beams:
                break

            # 准备解码器输入
            decoder_inputs = torch.stack([beam['sequence'] for beam in active_beams])

            # 解码器前向传播
            outputs = self.model(
                attention_mask=batched_attention_mask[:len(active_beams)],
                encoder_outputs=batched_encoder_outputs,
                decoder_input_ids=decoder_inputs,
                use_cache=False
            )

            # 获取下一个token的概率
            next_token_logits = outputs.logits[:, -1, :]
            next_token_probs = F.log_softmax(next_token_logits, dim=-1)

            # 扩展每个活跃束
            new_beams = []
            for i, beam in enumerate(active_beams):
                # 获取top-k个候选token
                topk_probs, topk_indices = torch.topk(next_token_probs[i], num_beams)

                for j in range(num_beams):
                    token_id = topk_indices[j].item()
                    token_prob = topk_probs[j].item()

                    # 创建新序列
                    new_sequence = torch.cat([beam['sequence'], torch.tensor([token_id], device=self.device)])

                    # 计算新得分（对数概率累加）
                    new_score = beam['score'] + token_prob

                    # 检查是否结束（生成eos token）
                    is_finished = (token_id == self.tokenizer.eos_token_id) or (step == max_length - 1)

                    new_beams.append({
                        'sequence': new_sequence,
                        'score': new_score,
                        'is_finished': is_finished,
                        'input_id': beam['input_id']
                    })

            # 按输入样本分组，为每个样本选择得分最高的num_beams个束
            grouped_beams = {}
            for beam in new_beams:
                input_id = beam['input_id']
                if input_id not in grouped_beams:
                    grouped_beams[input_id] = []
                grouped_beams[input_id].append(beam)

            # 选择每个输入样本得分最高的num_beams个束
            selected_beams = []
            for input_id, beam_group in grouped_beams.items():
                beam_group.sort(key=lambda x: x['score'], reverse=True)
                selected_beams.extend(beam_group[:num_beams])

            beams = selected_beams

        # 为每个输入样本选择得分最高的束
        grouped_final_beams = {}
        for beam in beams:
            input_id = beam['input_id']
            if input_id not in grouped_final_beams:
                grouped_final_beams[input_id] = []
            grouped_final_beams[input_id].append(beam)

        best_sequences = []
        for i in range(batch_size):
            if i in grouped_final_beams:
                best_beam = max(grouped_final_beams[i], key=lambda x: x['score'])
                best_sequences.append(best_beam['sequence'])
            else:
                # 如果没有完成的束，返回第一个
                best_sequences.append(beams[0]['sequence'])

        return torch.stack(best_sequences)

    def predict(self, text, use_custom_beam_search=True):
        '''
        :param text: 输入文本
        :param use_custom_beam_search: 是否使用自定义束搜索
        :return: 预测摘要
        '''
        is_str = isinstance(text, str)
        if is_str:
            text = [text]

        # 处理输入
        inputs = self.tokenizer(
            text,
            max_length=config.MAX_LENGTH,
            truncation=True,
            padding=True,
            return_tensors='pt',
            return_token_type_ids=False
        ).to(self.device)

        # 预测摘要
        self.model.eval()
        with torch.no_grad():
            if use_custom_beam_search:
                # 使用自定义束搜索
                outputs = self.beam_search(
                    inputs['input_ids'],
                    inputs['attention_mask'],
                    max_length=28,
                    num_beams=6
                )
            else:
                # 使用Hugging Face内置的束搜索
                outputs = self.model.generate(
                    **inputs,
                    max_length=28,
                    num_beams=6,
                    early_stopping=True
                )

        predictions = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        results = [''.join(prediction).replace(' ', '') for prediction in predictions]

        if is_str:
            return results[0]
        return results


if __name__ == '__main__':
    # 初始化分类模型和分词器
    model = AutoModelForSequenceClassification.from_pretrained(config.CHECKPOINT_DIR / 'classify' / 'best')
    tokenizer = AutoTokenizer.from_pretrained(config.CHECKPOINT_DIR / 'classify' / 'best')
    # 初始化摘要模型和分词器
    summary_model = AutoModelForSeq2SeqLM.from_pretrained(config.CHECKPOINT_DIR / 'summary')
    summary_tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_DIR / 'bart-base-chinese')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建分类和摘要预测器
    predictor = Predictor(model, tokenizer, device)
    summary_predictor = SummaryPredictor(summary_model, summary_tokenizer, device)

    # 测试文本
    text_one = "4月13日，全球首个人形机器人半程马拉松赛将在北京亦庄举行。赛事由北京市体育局、北京市经济和信息化局、中央广播电视总台北京总站、北京经济技术开发区管理委员会等单位联合主办。随着比赛临近，参赛机器人在跑道上进行了首次路测。路测期间，机器人们表现如何？"
    text = [
        "4月13日，全球首个人形机器人半程马拉松赛将在北京亦庄举行。赛事由北京市体育局、北京市经济和信息化局、中央广播电视总台北京总站、北京经济技术开发区管理委员会等单位联合主办。随着比赛临近，参赛机器人在跑道上进行了首次路测。路测期间，机器人们表现如何？",
        "据日本鹿儿岛地方气象台消息，当地时间3日13时49分左右，位于鹿儿岛县和宫崎县交界地区雾岛山的新燃岳火山喷发，火山灰柱最大高度达5000米。",
        "自7月7日开始，在中国人民抗日战争纪念馆举办“为了民族解放与世界和平——纪念中国人民抗日战争暨世界反法西斯战争胜利80周年主题展览”，展出照片1525张、文物3237件。主题展览将作为基本陈列长期展出。",
        "近日，印度中央邦博帕尔市一座铁路立交桥引发全球关注。这座桥的致命缺陷并非偷工减料或结构坍塌，而是惊现90 度直角转弯，司机必须紧急刹车才能通过，被网友调侃为 “现实版神庙逃亡赛道”。更讽刺的是，这座桥至今尚未通车，却已因设计失误导致 7 名工程师停职、两家建筑公司被列入黑名单。",
    ]

    for i in range(len(text)):
        print(f'文本：{text[i]}')
        print(f'类别：{predictor.predict(text[i])}')
        print(f'摘要：{summary_predictor.predict(text[i], use_custom_beam_search=True)}')
        # print(f'摘要：{summary_predictor.predict(text[i], use_custom_beam_search=False)}')
        print()
