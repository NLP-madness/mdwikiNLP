from collections import Counter
from nltk.util import ngrams 

text = "the quick person did not realize his speed and the quick person bumped "
n_gram = 2
Counter(ngrams(text.split(), n_gram))

## accessing text corpora ## 
# https://www.nltk.org/book/ch02.html

from nltk.corpus import gutenberg
import nltk
nltk.download('gutenberg')
gutenberg.fileids()

# Fiction: 
emma = gutenberg.words('austen-emma.txt')
print(emma[1:100])