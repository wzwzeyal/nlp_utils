
import re
import numpy as np
import pandas as pd


def keep_only_heb(p_text):
    chars_pattern = re.compile(r"""[^אבגדהוזחטיכלמנסעפצקרשתןףץםך'\- "]""")
    return chars_pattern.sub('', p_text)


def remove_k_length(text, k=1):
    text = [word for word in text.split() if len(word) > k]
    return " ".join(text)

def remove_words(text, words_to_remove):
    text = [word for word in text.split() if word not in words_to_remove]
    return " ".join(text)


def consolidate_k_chars(text, k):
    while True:
        count = 0
        chars = set(text)
        for c in chars:
            if c * k in text:
                text = text.replace(c * k, c)
                count += 1
        if count == 0:
            break
    return text


def clean_data(df, dirty_column, clean_column, remove_stop_words=False, stop_words=None):
    df[clean_column] = df[dirty_column].map(lambda x: keep_only_heb(x))
    df[clean_column] = df[clean_column].map(lambda x: remove_k_length(x))
    df[clean_column] = df[clean_column].map(lambda x: consolidate_k_chars(x, 3))
    if (remove_stop_words):
        df[clean_column] = df[clean_column].map(lambda x: remove_words(x, stop_words))