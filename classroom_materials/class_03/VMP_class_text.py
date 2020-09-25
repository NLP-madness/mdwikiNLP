# load from functions_VMP
import sys

sys.path.insert(1, "../class_02/")
from functions_VMP import *

# the 'Text' class:
class Text:
    def __init__(self, txt):
        self.sentences = sentence_segment(txt)
        self.tokens = tokenize(self.sentences)

    def ner(self, method="regex"):
        res = ner_regex(self.sentences)  # that is the input we worked with..
        return res

    def ngram(self, n=2):
        res = n_grams2(self.tokens, n)
        return res

    def token_frq(self):
        res = token_frequencies(self.tokens)
        return res

    def get_df(self):
        """
        returns a dataframe containing the columns:
        sentence number, token, lemma, pos-tag, named-entity
        """

        panda_df = stanza_panda(self.sentences, self.tokens)
        return panda_df


# String to test out on :
txt = """These are several sentences. They will be splittet a lot. It is inevitable. It will happen although J.D. Gould
would like it to be otherwise, or se he says.
This sentence tests (or intends to) test parenthes
and exclamations! At least that was the plan.
Another thing one might do is the following: testing this.
Abbreviations like are tricky. Does this come to mind?
I thought so. The little Martin Jr. thought it was good."""

# Testing it out - using the class:
ClassMember1 = Text(txt)

# Testing the init:
ClassMember1.sentences
ClassMember1.tokens

# Testing ner:
ClassMember1.ner()  # some empty ones..

# Testing get-df:
ClassMember1.get_df()

# Testing ngrams
ClassMember1.ngram()
ClassMember1.ngram(3)

# Testing token frq:
ClassMember1.token_frq()  # counter object returned.
