from pathlib import Path

# 目录
ROOT_DIR = Path(__file__).parent.parent.parent
RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw'
CLASSIFY_PROCESSED_DATA_DIR = ROOT_DIR / 'data' / 'processed' / 'classify'
CHECKPOINT_DIR = ROOT_DIR / 'checkpoint'
PRE_TRAINED_DIR = ROOT_DIR / 'pretrained'
LOGS_DIR = ROOT_DIR / 'logs'


# 训练参数
BATCH_SIZE = 16
MAX_LENGTH = 128