"""
This script contain an example Text class

Each function contains:
An explanation as well as an example
Your job as a studygroup is to make the functions in class 2 and 3.

In class 3 we will then assemble a pipeline using the Text class at the end.


I suggest you start from the top and for each function:
    1) discuss potential solution (e.g. sentence segmentation splitting by ".")
    2) implement the simplest solution first for each function
    3) go back and improve upon the initial function
    

for class 2 it would be ideal if you have a simple version of the following
functions:
    sentence_segment
    tokenize
    ner_regex

Additional stuff which you might add is:
    A function for dependency parsing using stanza
    alternatives to each of the function (e.g. using tokenization using nltk)
    Add a function for stemming
    Add plotting functionality for word frequencies
    Add plotting functionality for dependency trees
"""
<<<<<<< HEAD

import re

def sentence_segment(txt):
    """
    txt (str): Text which you want to be segmented into sentences.

    hint: look up the re.split() function in the re module and 
    and .split() method for strings.

    Example:
    >>> txt = "NLP is very cool. It is also useful"
    >>> sentence_segment(txt)
    ["NLP is very cool", "It is also useful"]
    """

    splittet = re.split("\\.", txt) 

    return(splittet)

'''
** look into re.compile. 
** look at "look ahead". 
** NB: how do we make sure not 
to split A. B. Bernhard (for instance)
'''
    


txt = "This is a sentence. Another one"
txt_new = sentence_segment(txt)
print(txt_new)

def tokenize(sentences):
    """
    sentences (list): Sentences which you want to be tokenized

    Example:
    >>> sent = ["NLP is very cool"]
    >>> tokenize(sent)
    [["NLP", "is", "very", "cool"], ["It", "is", "also", "useful"]]
    """
    pass


def n_grams(tokenlist, n):
    """
    tokenlist (list): A list of tokens
    n (int): Indicate the n in n-gram. n=2 denotes bigrams
    Indicate the n in n-gram. n=2 denotes bigrams

    creates n-grams from a given tokenlist

    Example:
    >>> tokens = ["NLP", "is", "very", "cool"]
    >>> n_grams(tokens, n=2)
    [["NLP", "is"], ["is", "very"], ["very", "cool"]]
    """
    pass


def ner_regex(sentence_list):
    """
    sentence_list (list): a list of sentences

    Named entity recognition using regular expressions.

    alternative options:could also be a list of tokens and/or the raw text.
    This will result in how you can you can use it later on, but for now
    let's not dwell too much on this.

    hint: look into the re package/module

    Example:
    >>> sent = [["Karl Friston is very cool"], ["Darwin is kick-ass"]]
    >>> ner_regex(sent)
    [["Karl Friston"], ["Darwin"]]
    """
    pass


def token_frequencies(tokenlist):
n    """
    tokenlist (list): A list of tokens
    could also be a list of token

    return a list of tokens and their frequencies

    hint: look up the Counter class for python

    Example:
    >>> tokens = [["NLP", "is", "very", "cool"],
                  ["It", "is", "also", "useful"]]
    >>> token_frequencies(sent)
    {"NLP": 1, "is": 2, "very": 1, "cool": 1, "It": 1, "also": 1, "useful": 1}
    """
    pass


def lemmatize_stanza(tokenlist):
    """
    tokenlist (list): A list of tokens

    lemmatize a tokenlist using stanza

    hint: examine the stanza_example.py script
    """
    pass


def postag_stanza(tokenlist):
    """
    tokenlist (list): A list of tokens

    add a part-of-speech (POS) tag to each tokenlist using stanza

    hint: examine the stanza_example.py script
    """
    pass

=======
from nlp_functions import lemmatize_stanza as lemma
from tokenizer import *
>>>>>>> upstream/master

class Text():
    def __init__(self, txt):
        self.sentences = sentence_segment()


    def tokenize(self, tokenizer="myown"):
        if tokenizer == "myown":
            self.tokens = tokenize(self.sentences)
        if tokenizer == "stanza":
            self.tokens = tokenize_stanza(self.sentences)


    def ner(self, method="regex"):
        if method == "regex":
            res = ner_regex(self.tokens)
        else:
            raise ValueError(f"method {method} is not a valid method")
        return res

    # add methods to do pos-tagging, lemmatization
    # n-grams and token frequencies

    def get_df(self):
        """
        returns a dataframe containing the columns:
        sentence number, token, lemma, pos-tag
        andd optionally named-entities
        """
        pass

    # add methods to extract tokens, sentences
    # ner, pos-tags etc.

Text("this is my text")
Text.tokenize()
Text.n_grams(n=2)
Text.ner(method="regex")













