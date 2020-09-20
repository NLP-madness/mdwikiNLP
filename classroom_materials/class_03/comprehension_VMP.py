#importing sys: 
import sys

#setting path to the file: 
sys.path.insert(1, '../class_02') 

#importing functions from there: 
import class_02_VMP as vmp

#task 3.8 (un-smoothed unigrams and bigrams): 
txt = """I am the best blue cat. I sat in the sand. 
When I sat in the world"""

#segment into sentences: 
txt_seg = vmp.sentence_segment(txt)
print(txt_seg)

#tokenize these sentences: 
txt_token = vmp.tokenize(txt_seg)
print(txt_token)

#use the n_grams function: 
unigrams = vmp.n_grams2(txt_token, 1)
bigrams = vmp.n_grams2(txt_token, 2)

print(unigrams)
print(bigrams)

#flatten? 
unigram_flat = [item for sublist in unigrams for item in sublist]
bigrams_flat = [item for sublist in bigrams for item in sublist]

#compute frequency: 
from collections import Counter

#unigram is simply the probability of a word (MLE): 
print(unigram_flat)
print(bigrams_flat)

#token frequencies? 
unigrams_frq = vmp.token_frequencies(unigram_flat)
unigrams_frq

bigrams_frq = vmp.token_frequencies(bigrams_flat)
bigrams_frq

#pandas? 
#issues: comma and end/start of sentence: 
import pandas as pd 

#as dataframe: 
bigrams_df = pd.DataFrame(bigrams_flat, columns=["first", "second"])
print(bigrams_df.head(n = 10)) 

#as "matrix": 
bigrams_matrix = bigrams_df.groupby(['first','second']).size().unstack(fill_value=0)
bigrams_matrix

#as probabilities: 
## df.sum(axis=1) (sum over rows): 
bigrams_matrix.loc[:, 0:] = bigrams_matrix.iloc[:, 0:].div(bigrams_matrix.sum(axis = 1), axis=0)
bigrams_matrix.head(n = 10)

##### Generate Random Sequences ###### 



### issues: 
# 1. start/end of sentence. 
# 2. commas and other things that don't end sentences but are not words. 
# 2.1. e.g., we have both "Ipsum", and "Ipsum," as different words.
#       (tokenize).  
# 3. do it for unigrams. 
# 

## Example / Test: 