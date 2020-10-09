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

    def tokenize(self, method="keras"):
        if method == "keras":
            self.tokens = keras_tokenizer(self.txt)
        elif method == "nltk":
            self.tokens = nltk_tokenizer(self.txt)

    def get_frequencies(self):
        return Counter(self.tokens)

    def get_tokens(self):
        return self.tokens
