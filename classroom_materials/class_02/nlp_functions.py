

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
    pass


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
    """
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