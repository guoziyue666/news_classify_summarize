import warnings

warnings.filterwarnings('ignore')
import jieba
from evaluate import load
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from configuration import config
from preprocess.dataset import get_summarize_dataloader
from runner.predict import SummaryPredictor

if __name__ == '__main__':
    model = AutoModelForSeq2SeqLM.from_pretrained(config.CHECKPOINT_DIR / 'summary')
    tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_DIR / 'bart-base-chinese')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = get_summarize_dataloader(tokenizer, 'test')
    summary_predictor = SummaryPredictor(model, tokenizer, device)
    refers_list, preds_list = [], []
    for batch in dataloader:
        inputs = [tokenizer.decode([token_id for token_id in input_id if token_id >= 0], skip_special_tokens=True) for
                  input_id in batch['input_ids']]
        inputs = [''.join([token for token in input if token != ' ']) for input in inputs]

        refers = [tokenizer.decode([token_id for token_id in refer if token_id >= 0], skip_special_tokens=True) for
                  refer in batch['labels']]
        refers = [''.join([token for token in refer if token != ' ']) for refer in refers]
        preds = summary_predictor.predict(inputs)

        refers_list.extend(refers)
        preds_list.extend(preds)

    rouge_scores = load(str(config.ROUGE_DIR)).compute(
        predictions=preds_list,
        references=refers_list,
        tokenizer=jieba.lcut
    )
    for k, v in rouge_scores.items():
        print(k, v)
        # break
