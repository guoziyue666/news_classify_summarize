import warnings
warnings.filterwarnings('ignore')
import jieba
from evaluate import load

from configuration import config

# preds（预测）
preds = [
    "The cat sat on the mat.",
    "A quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world.",
]

# refers（参考）
refers = [
    "The cat is sitting on the mat.",
    "A brown fox quickly jumps over the lazy dog.",
    "AI is changing the world in significant ways.",
]

rouge_scores = load(str(config.ROUGE_DIR)).compute(
    predictions=preds,
    references=refers,
    tokenizer=jieba.lcut
)
for k, v in rouge_scores.items():
    print(k, v)

# rouge1 0.6955636955636955     unigram（单个词）重合率
# rouge2 0.44570707070707066    bigram（两个连续的词）重合率
# rougeL 0.6955636955636955     最长公共子序列（LCS）
# rougeLsum 0.6955636955636955  引入句子边界后的LCS
