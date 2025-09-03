import random

from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer

from configuration import config


def process():
    # 1. 读取数据
    dataset_dict = load_dataset('csv', data_files=str(config.RAW_DATA_DIR / 'news.csv'))['train']
    print(dataset_dict)

    # 2. 随机选择10%的数据
    # dataset_size = len(dataset_dict)
    # sample_size = int(dataset_size * 0.1)
    # random_indices = random.sample(range(dataset_size), sample_size)
    # dataset_dict = dataset_dict.select(random_indices)
    # print(dataset_dict)

    # 3. 处理label
    all_labels = sorted(set(dataset_dict['category']))
    class_label = ClassLabel(names=all_labels)
    dataset_dict = dataset_dict.cast_column('category', class_label)
    # print(dataset_dict)
    with open(config.CHECKPOINT_DIR / 'labels.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_labels))

    dataset_dict = dataset_dict.train_test_split(test_size=0.2, seed=42)

    # 4. 处理新闻正文，生成输入数据
    tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_DIR / 'bert-base-chinese')

    def tokenize(batch):
        inputs = tokenizer(batch['text'], truncation=True)
        inputs['labels'] = batch['category']
        return inputs

    dataset_dict = dataset_dict.map(tokenize, batched=True, batch_size=1000,
                                    remove_columns=['category', 'summary', 'text'])
    print(dataset_dict['train'][0:3])

    # 5. 保存数据集
    dataset_dict.save_to_disk(config.CLASSIFY_PROCESSED_DATA_DIR)


def process_summarize():
    # 1. 读取数据
    dataset_dict = load_dataset('csv', data_files=str(config.RAW_DATA_DIR / 'news.csv'))['train']
    print(dataset_dict)

    # 2. 随机选择1%的数据
    dataset_size = len(dataset_dict)
    sample_size = int(dataset_size * 0.1)
    random_indices = random.sample(range(dataset_size), sample_size)
    dataset_dict = dataset_dict.select(random_indices)
    print(dataset_dict)

    # 3. 划分训练集和测试集
    dataset_dict = dataset_dict.train_test_split(test_size=0.2, seed=42)

    # 4. 创建分词器并定义分词函数
    tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_DIR / 'bart-base-chinese')

    # 在划分训练集和测试集之后，映射函数之前，添加过滤步骤
    def filter_empty_examples(example):
        return example['text'] is not None and example['text'].strip() != '' and \
            example['summary'] is not None and example['summary'].strip() != ''

    dataset_dict = dataset_dict.filter(filter_empty_examples)

    def tokenize(batch):
        inputs = tokenizer(text=batch['text'],
                           text_target=batch['summary'],
                           max_length=1024,
                           truncation=True,
                           return_token_type_ids=False)
        return inputs

    dataset_dict = dataset_dict.map(tokenize, batched=True, batch_size=100,
                                    remove_columns=['category', 'summary', 'text'])

    print(dataset_dict['train'][0:3])

    # 5. 保存数据集
    dataset_dict.save_to_disk(config.SUMMARY_PROCESSED_DATA_DIR)


if __name__ == '__main__':
    process()
    # process_summarize()
