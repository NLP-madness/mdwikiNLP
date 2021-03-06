"""
This is a document containing all of the functions
needed to make the 'text' class. 
"""

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
    n_grams
    ner_regex

Additional stuff which you might add is:
    A function for dependency parsing using stanza
    alternatives to each of the function (e.g. using tokenization using nltk)
    Add a function for stemming
    Add plotting functionality for word frequencies
    Add plotting functionality for dependency trees
"""


# string to test on:
txt = """These are several sentences. They will be splittet a lot. It is inevitable. 
It will happen although J.D. Gould would like it to be otherwise, or se he says.
This sentence tests (or intends to) test parenthes
and exclamations! At least that was the plan.
Another thing one might do is the following: testing this.
Abbreviations like are tricky. Does this come to mind?
I thought so. The little Martin Jr. thought it was good."""

## sentence segmentation:
def sentence_segment(txt):
    """
    txt (str): Text which you want to be segmented into
    sentences.

    Example:
    >>> txt = "NLP is very cool. It is also useful"
    >>> sentence_segment(txt)
    ["NLP is very cool", "It is also useful"]
    """

    # importing the module re:
    import re

    p1 = "(?<!\w\.\w)(?<![A-Z][a-z])[!:?.]\s"
    # [!:?.] --> trying to match.
    # \s (also matching to delete). Whitespace
    # (?<!\w\.\w) catches e.g. and the likes.
    # (?<![A-Z][a-z]) catches Ms. Smith, doesn't split Ms.
    # --> still lacking Mrs. Smith (to do).
    # newline?

    # https://regex101.com/r/nG1gU7/27 (inspiration).

    # use our regex:
    splittet = [w.replace("\n", "") for w in re.split(p1, txt)]
    splittet = [re.sub("\n|\.|\(|\)|,", "", w) for w in re.split(p1, txt)]

    # has to be done to accommodate the pipeline:
    # list of lists instead of list..
    sentences = [[sent] for sent in splittet if sent != ""]
    return sentences


""" Issues: getting the last dot """

# testing:
sentence_seg = sentence_segment(txt)
print(sentence_seg)

## tokenization
## using nltk?
def tokenize(sentences):
    """
    sentences (list): Sentences which you want to be tokenized

    Example:
    >>> sent = ["NLP is very cool"]
    >>> tokenize(sent)
    [["NLP", "is", "very", "cool"], ["It", "is", "also", "useful"]]
    """
    ## importing re:
    import re

    ## unlist (fixing issues):
    sentences_flat = [word for w in sentences for word in w]

    ## Split these: (keep words like J. D. Gould together)?
    ## More work required here..?
    output = [re.split("\W", b) for b in sentences_flat]

    ## This
    return output


# testing:
sentence_tok = tokenize(sentence_seg)
print(sentence_tok)

## n-grams (for unnested list):

## recursion:
def n_grams(tokenlist, n):
    """
    tokenlist (list): A list of tokens
    n (int): Indicate the n in n-gram. n=2 denotes bigrams

    creates n-grams from a given tokenlist

    Example:
    >>> tokens = ["NLP", "is", "very", "cool"]
    >>> n_grams(tokens, n=2)
    [["NLP", "is"], ["is", "very"], ["very", "cool"]]
    """

    # initialization:
    master_list = []  # empty list:
    sub_list = []  # empty list

    # for loop: (list comprehension?)
    for i in range(
        len(tokenlist) - (n - 1)
    ):  # loop through is dependent on n-1 (what gram we do)
        for j in range(n):  # how many things we will append
            sub_list.append(tokenlist[i + j])  # append to sub list.
        master_list.append(sub_list)  # append sub list to master list.
        sub_list = []  # clear the sub-list.

    # return:
    return master_list


## n-grams for nested list:
def n_grams2(tokenlist, n):
    """
    tokenlist (list): A list of tokens
    n (int): Indicate the n in n-gram. n=2 denotes bigrams

    creates n-grams from a given tokenlist

    Example:
    >>> tokens = ["NLP", "is", "very", "cool"]
    >>> n_grams(tokens, n=2)
    [["NLP", "is"], ["is", "very"], ["very", "cool"]]
    """

    # initialization:
    lst_complete = []  # empty list.
    lst_sentence = []  # empty list.
    lst_word = []  # empty list.

    for i in range(len(tokenlist)):  # sentences:
        for j in range(len(tokenlist[i]) - (n - 1)):
            for k in range(n):
                lst_word.append(tokenlist[i][j + k])  # append to word list.
            lst_sentence.append(lst_word)  # append to sentence list.
            lst_word = []  # clear word list.
        lst_complete.append(lst_sentence)
        lst_sentence = []  # clear sentence list.

    # return:
    return lst_complete


##  Named entity recognition:
# Obviously this cannot distinguish anything
# Starting a sentence (e.g., "I am" from "Michelle is").
# So, it is very insufficient.
def ner_regex(tokenlist):
    """
    tokenlist (list): A list of tokens

    peforms named entity recognition using regular expressions
    Example:
    >>> sent = [["Karl Friston is very cool"], ["Darwin is kick-ass"]]
    >>> ner_regex(sent)
    [["Karl Friston"], ["Darwin"]]
    """

    # import re:
    import re

    # capture group and non-capture groups:
    pattern = re.compile(r"(?<!^)([A-Z][a-z]+)")

    # using findall and compile from re.
    lst = []
    for i in tokenlist:
        unlisted = "".join(i)
        print(unlisted)
        recognized = re.findall(pattern, unlisted)
        lst.append(recognized)
    return lst


# testing:
sentence_reg = ner_regex(sentence_seg)
print(sentence_reg)

## Token frequencies


def token_frequencies(tokenlist):
    """
    tokenlist (list): A list of tokens

    return a list of tokens and their frequencies

    Example:
    >>> tokens = [["NLP", "is", "very", "cool"],
                  ["It", "is", "also", "useful"]]
    >>> token_frequencies(sent)
    {"NLP": 1, "is": 2, "very": 1, "cool": 1, "It": 1, "also": 1, "useful": 1}
    """
    # import Counter
    from collections import Counter

    # initialize our counter/dictionary:
    token_frq = Counter()

    # unlist (we don't care about which sentence for now):
    # this probably only works for "once" nested..
    tokens_list = [item for sublist in tokenlist for item in sublist]

    # https://docs.python.org/2/library/collections.html
    for word in tokens_list:
        token_frq[word] += 1

    return token_frq


## Lemmatize using stanza (redundant now)
def lemmatize_stanza(tokenlist, return_df=False):
    """
    tokenlist (list): A list of tokens

    lemmatize a tokenlist using stanza
    """

    import stanza

    nlp = stanza.Pipeline(
        lang="en", processors="tokenize,lemma", tokenize_pretokenized=True
    )
    doc = nlp(tokenlist)

    res = [
        (word.lemma) for n_sent, sent in enumerate(doc.sentences) for word in sent.words
    ]

    if return_df:
        import pandas as pd

        return pd.DataFrame(res)
    return res


## POS-tag using stanza (redundant now):


def postag_stanza(tokenlist, return_df=False):
    """
    tokenlist (list): A list of tokens

    add a part-of-speech (POS) tag to each tokenlist using stanza
    """

    import stanza

    nlp = stanza.Pipeline(
        lang="en", processors="tokenize,lemma,mwt,pos", tokenize_pretokenized=True
    )

    doc = nlp(tokenlist)

    res = [
        (word.lemma, word.pos)
        for n_sent, sent in enumerate(doc.sentences)  # n_sent sentence number?
        for word in sent.words
    ]

    if return_df:
        import pandas as pd

        return pd.DataFrame(res)
    return res


### new super-function which returns everything ###

## trouble-shooting:
print(sentence_reg)

lst = []
for n, sentence in enumerate(sentence_reg):
    print(n + 1)
    print(sentence)
    if sentence == []:
        placeholder = (n + 1, "None", False)
        lst.append(placeholder)
    if sentence != []:
        for i in sentence:
            placeholder = (n + 1, i, True)
            lst.append(placeholder)
    placeholder = None

"""

import pandas as pd

testFrame = pd.DataFrame(lst, columns=["sentence num", "token", "ner"])

testFrame

#
ners_clean = [(n_sentence + 1, sent) for n_sentence, sent in enumerate(sentence_reg)]
ners_clean
"""


def stanza_panda(segmented, tokenlist, return_df=True):
    """
    write doc-string.
    """

    import stanza

    ## stanza stuff ##
    nlp = stanza.Pipeline(
        lang="en", processors="tokenize, mwt, pos, lemma", tokenize_pretokenized=True
    )

    doc = nlp(tokenlist)

    ## obtained:
    # n_sent: sentence number
    # sent: token
    # word.lemma: lemma
    # word.pos: POS-tag
    # lacking (ner)..

    res = [
        (n_sentence + 1, word.id, word.text, word.lemma, word.pos)
        for n_sentence, sent in enumerate(doc.sentences)  # n_sent sentence number?
        for word in sent.words
    ]

    # NER on sentences:
    # ners = ner_regex(segmented)
    # ners_clean = [(n_sentence + 1,) for n_sentence, sent in enumerate(ners)]

    ## return pandas dataframe ##
    if return_df:
        import pandas as pd

        return pd.DataFrame(
            res, columns=["sentence num", "word num", "token", "lemma", "pos"]
        )
    return res


"""
# testing:
print(sentence_tok)
stanza_test = stanza_panda(sentence_seg, sentence_tok, True)
stanza_test
testFrame

## combining (based on order - does not work)
combined = pd.concat([stanza_test, testFrame], axis=1, join="inner")
combined

## merge left?
merged_left = pd.merge(left=stanza_test, right=testFrame, how='left', left_on='species_id', right_on='species_id')
combined = pd.DataFrame.merge([stanza_test, testFrame])
combined
"""
