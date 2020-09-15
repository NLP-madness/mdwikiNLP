#importing sys: 
import sys

#setting path to the file: 
sys.path.insert(1, '../class_02')

#importing functions from there: 
import class_02_VMP as vmp

#task 3.8 (unsmoothed unigrams and bigrams): 
txt = """
There are many variations of passages of Lorem Ipsum available, 
but the majority have suffered alteration in some form, 
by injected humour, or randomised words which don't look even 
slightly believable. If you are going to use a passage of Lorem Ipsum, 
you need to be sure there isn't anything embarrassing hidden in the 
middle of text. All the Lorem Ipsum generators on the Internet tend 
to repeat predefined chunks as necessary, making this the first 
true generator on the Internet. It uses a dictionary of over 200 
Latin words, combined with a handful of model sentence structures, 
to generate Lorem Ipsum which looks reasonable. The generated 
Lorem Ipsum is therefore always free from repetition, injected humour, 
or non-characteristic words etc."""


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

def grams(n_gram_list, n): 
    

