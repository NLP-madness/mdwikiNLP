# Question 3.8 (first ~ 100 lines)

corpus ="""Monty Python (sometimes known as The Pythons) were a British surreal comedy group who created the sketch comedy show Monty Python's Flying Circus,
that first aired on the BBC on October 5, 1969. Forty-five episodes were made over four series. The Python phenomenon developed from the television series
into something larger in scope and impact, spawning touring stage shows, films, numerous albums, several books, and a stage musical.
The group's influence on comedy has been compared to The Beatles' influence on music."""

#importing stuff. 
import collections, nltk, sys, re 

#functions that we need: 
# \n is also an issue: 
def sentence_segment(txt):
    p1 = "(?<!\w\.\w)(?<![A-Z][a-z])[!:?.\v]\s"
    splittet = [f"<s> {w} </s>" for w in re.split(p1, txt)]
    return(splittet)

#now it keeps the dot at the end of sentence: 
corpus_segment = sentence_segment(corpus)
print(corpus_segment)

#tweaking to get all characters unique: 
#still has the comma issue and other characters that are immediately next to a word.
def tokenize(sentences):
    output = [re.split(" ", b) for b in sentences]
    return(output)

corpus_token = tokenize(corpus_segment)
print(corpus_token)

#unlisting: 
token_flat = [item for sublist in corpus_token for item in sublist]
print(token_flat)


#testing: 
#import Counter: 
from collections import Counter 
token_frq = Counter()

#tuples for bigrams 
token_bigram = [(token_flat[w], token_flat[w+1],) for w in range(len(token_flat)-1)]
#token_unigram = [(w,) for w in token_flat]

#function to get frequency: 
def token_frequencies(tokenlist):
    #import Counter: 
    from collections import Counter 

    #initialize our counter/dictionary: 
    token_frq = Counter()

    #https://docs.python.org/2/library/collections.html
    for i in tokenlist: 
        token_frq[i] += 1

    return(token_frq)

#use the function: 
count_unigram = token_frequencies(token_flat)
count_bigram = token_frequencies(token_bigram)
print(count_unigram)
print(count_bigram)

### scale the values ### 
# this could be A LOT smarter # 
def scaling(scaleobj, n, unigrams = []): #n can be 1 or 2 (unigram, bigram): 
    
    #import Counter: 
    from collections import Counter 

    #initialize our counter/dictionary: 
    token_frq = Counter()
    
    #unigrams: 
    if n == 1:
        N = float(sum(scaleobj.values()))
        for word in scaleobj: 
            token_frq[word] = scaleobj[word]/N
    
    if n == 2: 
        for words in scaleobj: 
            first_word = words[0] #what the bigram starts with: 
            print(f"first = {first_word}")
            w_n_minus1 = unigrams[words[0]]
            print(f"minus 1 = {w_n_minus1}")
            print(f"conditional = {scaleobj[words]}")
            token_frq[words] = scaleobj[words] / w_n_minus1

    return token_frq


#using the function: 
scaled_uni = scaling(count_unigram, 1)
scaled_bi = scaling(count_bigram, 2, count_unigram)

print(sum(scaled_uni.values())) # good. 
print(sum(scaled_bi.values())) # probably good. 

### Question 3.9 ### 

# loading files
import requests
def download_txt(url, name=None, write=True):
    if write and name is None:
        raise ValueError("Name is None")
    r = requests.get(url)

    # write content to a .txt file
    with open(name, 'w') as f:
        f.write(r.content.decode("windows-1252"))

url = """http://www.glozman.com/TextPages/Harry%20Potter%201%20-%20Sorcerer's%20Stone.txt"""
book_name = "harry_potter_sorcerers_stone.txt"

download_txt(url, name=book_name)
with open(book_name, "r") as f:
    harry = f.read()

url = """http://www.glozman.com/TextPages/Auel,%20Jean%20-%20Earth's%20Children%2001%20-%20The%20Clan%20of%20the%20Cave%20Bear.txt"""
book_name = "The Clan of the Cave Bear.txt"

download_txt(url, name=book_name)
with open(book_name, "r") as f:
    children = f.read()

children[0:100]
harry[0:100]

### processing Harry Potter & Jean M. Aul"



#### stolen code ##### 

#here you construct the unigram language model 
#turn this into bigram as well?
def unigram(tokens):    
    model = collections.defaultdict(lambda: 0.01) #smoothing 0.01 (generalizing to unseen).
    for f in tokens: #could also be bi-gram. 
        try:
            model[f] += 1 
        except KeyError: #when would this happen?
            model[f] = 1
            continue
    N = float(sum(model.values())) #all the values. 
    for word in model:
        model[word] = model[word]/N #normalize. 
    return model

model

#computes perplexity of the unigram model on a testset  
def perplexity(testset, model):
    testset = testset.split() #split by whitespace. 
    perplexity = 1 #not sure? maybe 2. 
    N = 0 #not sure? 
    for word in testset:
        N += 1 #to normalize. 
        perplexity = perplexity * (1/model[word]) 
    perplexity = pow(perplexity, 1/float(N)) #normalize. 
    return perplexity

testset1 = "Monty" 
testset2 = "abracadabra gobbledygook rubbish"

model["abracadabra"]

model = unigram(tokens)
print(perplexity(testset1, model))
print(perplexity(testset2, model))

#random text: https://stackoverflow.com/questions/40927221/how-to-choose-keys-from-a-python-dictionary-based-on-weighted-probability
import random
def weighted_random_by_dct(dct):
    rand_val = random.random() #random value. 
    total = 0 #start from 0. 
    for k, v in dct.items(): #k = key, v = value. 
        total += v #
        if rand_val <= total: #when it gets above. 
            return k #return. 
    assert False, 'unreachable'

for k, v in model.items(): 
    print(k)
    print(v) 


weighted_random_by_dct(model)