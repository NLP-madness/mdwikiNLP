
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

#set-up:

#string to test on:
txt = """These are several sentences. They will be splittet a lot. It is inevitable. It will happen although J.D. Gould
would like it to be otherwise, or se he says.
This sentence tests (or intends to) test parenthes
and exclamations! At least that was the plan.
Another thing one might do is the following: testing this.
Abbreviations like are tricky. Does this come to mind?
I thought so. The little Martin Jr. thought it was good."""

#importing re:
import re

#actual function:
def sentence_segment(txt):
    """
    txt (str): Text which you want to be segmented into
    sentences.

    Example:
    >>> txt = "NLP is very cool. It is also useful"
    >>> sentence_segment(txt)
    ["NLP is very cool", "It is also useful"]
    """

    p1 = "(?<!\w\.\w)(?<![A-Z][a-z])[!:?.]\s"
    # [!:?.] --> trying to match.  
    # \s (also matching to delete). Whitespace 
    # (?<!\w\.\w) catches e.g. and the likes. 
    # (?<![A-Z][a-z]) catches Ms. Smith, doesn't split Ms. 
    # --> still lacking Mrs. Smith (to do). 
    # newline? 

    #https://regex101.com/r/nG1gU7/27 (inspiration).

    #use our regex:
    splittet = [w.replace("\n", "") for w in re.split(p1, txt)]

    return(splittet)

#questions: is re.compile smarter in some way?

#testing the function:
segmented = sentence_segment(txt)
print(segmented)

### using a list comprehension:
## probably this is the issue (i.e., remove comma).
def tokenize(sentences):
    """
    sentences (list): Sentences which you want to be tokenized

    Example:
    >>> sent = ["NLP is very cool"]
    >>> tokenize(sent)
    [["NLP", "is", "very", "cool"], ["It", "is", "also", "useful"]]
    """
    output = [b.split() for b in sentences]
    return(output)

#testing tokenize:
tokenized = tokenize(segmented)
print(tokenized)

#this one does it for one list:
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

    #initialization:
    master_list = [] #empty list:
    sub_list = [] #empty list

    #for loop: (list comprehension?)
    for i in range(len(tokenlist)-(n-1)): #loop through is dependent on n-1 (what gram we do)
        for j in range(n): #how many things we will append
            sub_list.append(tokenlist[i+j]) #append to sub list.
        master_list.append(sub_list) #append sub list to master list.
        sub_list = [] #clear the sub-list.

    #return:
    return(master_list)

#testing n_grams:
tokenlist = ["NLP", "is", "very", "useful"]
print(n_grams(tokenlist, 1))
print(n_grams(tokenlist, 2))
print(n_grams(tokenlist, 3))
print(n_grams(tokenlist, 4))

#this one does it for list of lists:
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

    #initialization:
    lst_complete = [] #empty list.
    lst_sentence = [] #empty list.
    lst_word = [] #empty list.

    for i in range(len(tokenlist)): #sentences:
        for j in range(len(tokenlist[i])-(n-1)):
            for k in range(n):
                lst_word.append(tokenlist[i][j+k]) #append to word list.
            lst_sentence.append(lst_word) #append to sentence list.
            lst_word = [] #clear word list.
        lst_complete.append(lst_sentence)
        lst_sentence = [] #clear sentence list.

    #return:
    return(lst_complete)


#testing on subset for clarity:
tokenized = tokenized[0:3]
print(tokenized)

#for different n:
print(n_grams2(tokenized, 1))
print(n_grams2(tokenized, 2))
print(n_grams2(tokenized, 3))

#for moving forward:
n_grammed = n_grams2(tokenized, 2)

#Named entity recognition:
#Obviously this cannot distinguish anything 
#Starting a sentence (e.g., "I am" from "Michelle is").
#So, it is very insufficient. 
def ner_regex(tokenlist):
    """
    tokenlist (list): A list of tokens

    peforms named entity recognition using regular expressions
    Example:
    >>> sent = [["Karl Friston is very cool"], ["Darwin is kick-ass"]]
    >>> ner_regex(sent)
    [["Karl Friston"], ["Darwin"]]
    """
    #capture group and non-capture groups: 
    pattern = re.compile(r"((?:[A-Z][a-z]+)(?:\s[A-Z][a-z]+)?)")
    
    #using findall and compile from re. 
    lst = []
    for i in txt:
        unlisted = "".join(i)
        recognized = re.findall(pattern, unlisted)
        lst.append(recognized)
    return(lst)

#testing the function:
txt = [["Karl Friston is very cool"], ["Darwin is kick-ass"]]
print(ner_regex(txt)) 

#import Counter: 
from collections import Counter 

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
    #initialize our counter/dictionary: 
    token_frq = Counter()

    #unlist (we don't care about which sentence for now): 
    #this probably only works for "once" nested..
    tokens_list = [item for sublist in tokenlist for item in sublist]

    #https://docs.python.org/2/library/collections.html
    for word in tokens_list: 
        token_frq[word] += 1

    return(token_frq)

#testing the function: 
tokens = [["NLP", "is", "very", "cool"],["It", "is", "also", "useful"]]

token_list = token_frequencies(tokens)
token_list

#lemmatization: 
#stolen from Kenneth: 

def lemmatize_stanza(tokenlist, processors, return_df=True,
                   print_dependency=False):
    """
    tokenlist (list): A list of tokens

    lemmatize a tokenlist using stanza
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


#testing the function: 
tl = [['This', 'is', 'tokenization', 'done', 'my', 'way!'],
      ['Sentence', 'split,', 'too!'],
      ['Las', 'Vegas', 'is', 'great', 'city']]

#this works: 
lemmatize_stanza(tokenlist=tl, processors='tokenize,lemma')

#why doesn't this work?
#lemmatize_stanza(tokenlist=t1, processors='lemma')


def postag_stanza(tokenlist):
    """
    tokenlist (list): A list of tokens

    add a part-of-speech (POS) tag to each tokenlist using stanza
    """
    pass


class Text():
    def __init__(self, txt):
        self.sentences = sentence_segment()
        self.tokens = tokenize(self.sentences)

    def ner(self, method="regex"):
        res = ner_regex(self.tokens)
        return res

    def get_df(self):
        """
        returns a dataframe containing the columns:
        sentence number, token, lemma, pos-tag, named-entity
        """
        pass
