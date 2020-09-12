"""
This is an example of how to use stanza
"""






def stanza_example(tokenlist, processors, return_df=True,
                   print_dependency=False):
    """
    An example using stanza for lemmatization, POS tagging and dependency
    parsing

    Examples (results omitted)
    >>> tl = [['This', 'is', 'token.ization', 'done', 'my', 'way!'],
              ['Sentence', 'split,', 'too!'],
              ['Las', 'Vegas', 'is', 'great', 'city']]
    >>> stanza_example(tokenlist=tl, processors='tokenize,pos') # POS tag only
    ...
    >>> tl = [tl[2]] # only use one sentence
    >>> stanza_example(tokenlist=tl, processors='tokenize,lemma,pos,depparse')
    ...
    """
    import stanza
    nlp = stanza.Pipeline(lang='en', processors=processors,
                          tokenize_pretokenized=True)
    doc = nlp(tokenlist)

    res = [(n_sent, word.text, word.lemma, word.upos, word.xpos, word.head, word.deprel)
           for n_sent, sent in enumerate(doc.sentences)
           for word in sent.words]

    if return_df:
        import pandas as pd
        return pd.DataFrame(res)
    return res




tl = [['This', 'is', 'token.ization', 'done', 'my', 'way!'],
      ['Sentence', 'split,', 'too!'],
      ['Las', 'Vegas', 'is', 'great', 'city']]
stanza_example(tokenlist=tl, processors='tokenize,pos')  # POS tag only

tl = [tl[2]]  # only use one sentence
stanza_example(tokenlist=tl, processors='tokenize,lemma,pos,depparse')


tl = [["green", "colorless", "ideas", "sleep", "furiously"]]
stanza_example(tokenlist=tl, processors='tokenize,lemma,pos,depparse')