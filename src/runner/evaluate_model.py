import warnings

warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    roc_auc_score
from tqdm import tqdm

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSequenceClassification

from preprocess.dataset import get_summarize_dataloader, get_dataloader
import jieba
# 添加logging模块用于控制jieba的日志输出
import logging
from evaluate import load

from configuration import config
from runner.predict import SummaryPredictor

# 禁用jieba的INFO级别日志输出
jieba.setLogLevel(logging.INFO)


def evaluate_classify():
    '''
    评估分类预测模型
    '''
    tokenizer = AutoTokenizer.from_pretrained(config.CHECKPOINT_DIR / 'classify' / 'best')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForSequenceClassification.from_pretrained(config.CHECKPOINT_DIR / 'classify' / 'best').to(device)

    test_dataloader = get_dataloader(tokenizer, 'test')
    model.eval()
    total_loss = 0
    all_labels, all_preds, all_probs = [], [], []
    for inputs in tqdm(test_dataloader, desc='[Evaluation]'):
        inputs = {key: value.to(device) for key, value in inputs.items()}
        # 构造标签列表
        labels = inputs['labels']
        all_labels.extend(labels.tolist())
        # 前向传播
        outputs = model(**inputs)
        loss = outputs.loss
        total_loss += loss.item()
        logits = outputs.logits

        # 构造预测列表
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.tolist())
        # 构建预测概率列表
        probs = torch.softmax(logits, dim=-1)
        all_probs.extend(probs.tolist())

    loss = total_loss / len(test_dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    report = classification_report(all_labels, all_preds, output_dict=True)
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovo')
    print(f'Test loss: {loss}')
    print(f'Test accuracy: {accuracy}')
    print(f'Test precision: {precision}')
    print(f'Test recall: {recall}')
    print(f'Test f1-score: {f1}')
    print(f'Test support: {report['macro avg']['support']}')
    print(f'Test auc: {auc}')


def evaluate_summary():
    '''
    评估摘要预测模型
    '''
    tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_DIR / 'bart-base-chinese')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForSeq2SeqLM.from_pretrained(config.CHECKPOINT_DIR / 'summary').to(device)

    summary_test_dataloader = get_summarize_dataloader(tokenizer, 'test')
    summary_predictor = SummaryPredictor(model, tokenizer, device)

    text_list, refers_list, preds_list = [], [], []
    model.eval()
    total_loss = 0
    for inputs in tqdm(summary_test_dataloader, desc='[Evaluation]'):
        inputs = {key: value.to(device) for key, value in inputs.items()}
        # 获取输入文本
        texts = [tokenizer.decode([token_id for token_id in input_id if token_id >= 0], skip_special_tokens=True) for
                 input_id in inputs['input_ids']]
        text = [input.replace(' ', '') for input in texts]
        text_list.extend(text)
        # 获取参考摘要
        refers = [tokenizer.decode([token_id for token_id in refer if token_id >= 0], skip_special_tokens=True) for
                  refer in inputs['labels']]
        refers = [refer.replace(' ', '') for refer in refers]
        refers_list.extend(refers)
        # 前向传播
        outputs = model(**inputs)
        loss = outputs.loss
        total_loss += loss.item()
        # logits = outputs.logits
        # 预测摘要
        # preds = torch.argmax(logits, dim=-1)
        # preds = [tokenizer.decode(pred, skip_special_tokens=True) for pred in preds]
        # print(type(preds))

    print(f'Test loss: {total_loss / len(summary_test_dataloader)}')
    # 获取预测摘要
    preds = summary_predictor.predict(text_list)
    preds = [pred.replace(' ', '') for pred in preds]
    preds_list.extend(preds)

    rouge_scores = load(str(config.ROUGE_DIR)).compute(
        references=refers_list,
        predictions=preds_list,
        tokenizer=jieba.lcut
    )

    for k, v in rouge_scores.items():
        print(f'Test {k}: {v}')


if __name__ == '__main__':
    evaluate_classify()
    # evaluate_summary()
