"""
This script is just intended as a placeholder for your own text preprocessing
script
"""

from collections import Counter


def nltk_tokenizer(txt):
    import nltk

    return nltk.word_tokenize(txt)


def keras_tokenizer(txt):
    from tensorflow.keras.preprocessing.text import text_to_word_sequence

    return text_to_word_sequence(
        txt, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=False, split=" "
    )


class Text:
    def __init__(self, txt):
        self.txt = txt

    def tokenize(self, method="keras", split_sent=False):
        txt = self.txt
        if method == "keras":
            self.tokenizer = keras_tokenizer
        elif method == "nltk":
            self.tokenizer = nltk_tokenizer

        if split_sent:
            txt = txt.split(".")  # too simple
            self.tokens = [self.tokenizer(t) for t in txt]
        else:
            self.tokens = self.tokenizer(txt)

    def get_frequencies(self):
        return Counter(self.tokens)

    def get_tokens(self):
        return self.tokens
