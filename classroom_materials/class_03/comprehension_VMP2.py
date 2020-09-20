### Question 3.8 (line ~1-65) ### 

#test corpus: 
corpus ="""Monty Python (sometimes known as The Pythons) were a British surreal comedy group who created the sketch comedy show Monty Python's Flying Circus,
that first aired on the BBC on October 5, 1969. Forty-five episodes were made over four series. The Python phenomenon developed from the television series
into something larger in scope and impact, spawning touring stage shows, films, numerous albums, several books, and a stage musical.
The group's influence on comedy has been compared to The Beatles' influence on music."""

#importing stuff. 
import collections, nltk, sys, re 

#functions that we need: 
# \n is also an issue: 
def preprocessing(corpus): 

    #sentence segmentation: 
    p1 = "(?<!\w\.\w)(?<![A-Z][a-z])[!:?.\v]\s"
    segmented = [f"<s> {w} </s>" for w in re.split(p1, corpus)]

    #tokenization: 
    tokenized = [re.split(" ", b) for b in segmented]

    #unlisting: 
    token_flat = [item for sublist in tokenized for item in sublist]
    #flat is used for unigrams.. 

    #tuples for bigrams 
    token_bigram = [(token_flat[w], token_flat[w+1],) for w in range(len(token_flat)-1)]

    #token frequencies: 
    from collections import Counter 
    unigram_count = Counter()
    bigram_count = Counter()

    #run this on both: 
    for i in token_flat: 
        unigram_count[i] += 1

    for i in token_bigram: 
        bigram_count[i] += 1

    #set-up for scaling values: 
    unigram_frq = Counter()
    bigram_frq = Counter() 

    #scaling values (for unigram):
    N = float(sum(unigram_count.values()))
    for word in unigram_count: 
        unigram_frq[word] = unigram_count[word]/N
    
    #scaling values (for bigram): 
    for words in bigram_count: 
        first_word = words[0] #what the bigram starts with: 
        w_n_minus1 = unigram_count[words[0]]
        bigram_frq[words] = bigram_count[words] / w_n_minus1

    return(unigram_frq, bigram_frq)


#run the function and get unigrams and bigrams (unsmoothed): 
unigram, bigram = preprocessing(corpus)

### Question 3.9 (line ~63-100) ### 

# importing files (Kenneth code): 
# we are importing Harry Potter and The Clan of the Cave Bear. 
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

# run the function (takes some time): 
harry_uni, harry_bi = preprocessing(harry)
children_uni, children_bi = preprocessing(children)

# most probable unigrams and bigrams (Harry Potter): 
print(harry_uni.most_common(10))
print(harry_bi.most_common(10))

#most probably unigrams and bigrams (Children): 
print(children_uni.most_common(10))
print(children_bi.most_common(10))

### Question 3.10 (line ~103-end)### 

## would be nice to do sentences ## 

#function for unigrams & bigrams: 
import random
def pick_next_unigram(dct):
    rand_val = random.random() #random value. 
    total = 0 #start from 0. 
    for k, v in dct.items(): #k = key, v = value. 
        total += v #
        if rand_val <= total: #when it gets above. 
            return k #return. 
    assert False, 'unreachable'

#pick bigram: 
def pick_next_bigram(dct, start_value): 

    #set-up
    rand_val = random.random() #between 0-1
    total = 0 

    #picking candidates: 
    candidates = [key for key in dct.keys() if key[0] == start_value]

    for i in candidates: 
        total += dct[i] 
        if rand_val <= total: 
            return i[1]
    assert False, 'unreachable'

#function (for bigrams): 
def pick_sentence(ngram, n):
    
    #we begin the sentence at start of sentence.. 
    start_value = '<s>' 
    print(start_value)

    #while-loop until we reach end of sentence.
    while start_value != '</s>': 
        if n == 2: 
            start_value = pick_next_bigram(ngram, start_value) 
        if n == 1: 
            start_value = pick_next_unigram(ngram)
        print(start_value) 

## Compare bigrams and unigrams ## 
pick_sentence(harry_bi, 2) #bi-gram
pick_sentence(children_bi, 2) #bi-gram
pick_sentence(harry_uni, 1) #uni-gram 
pick_sentence(children_uni, 1) #uni-gram


