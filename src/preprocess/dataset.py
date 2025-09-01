from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, AutoTokenizer

from configuration import config


def get_dataset(dtype='train'):
    dataset = load_from_disk(str(config.CLASSIFY_PROCESSED_DATA_DIR / dtype))
    dataset.set_format(type='torch')
    return dataset


def get_dataloader(tokenizer, dtype='train'):
    dataset = get_dataset(dtype)
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors='pt')
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_DIR / 'bert-base-chinese')
    dataloader = get_dataloader(tokenizer)
    for batch in dataloader:
        for k, v in batch.items():
            print(k, v.shape)
        break
