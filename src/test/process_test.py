from math import inf

from configuration import config

max_txt = 0
min_txt = inf
with open(config.RAW_DATA_DIR / 'news.csv', 'r', encoding='utf-8') as f:
    i = 1
    for line in f:
        if i == 1:
            i += 1
            continue
        txt = line.split(',')[2]
        max_txt = max(max_txt, len(txt))
        min_txt = min(min_txt, len(txt))
print(max_txt, min_txt)
