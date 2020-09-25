## Part 1: Practice Problems

# import stuff

from collections import Counter

# STEP 1 (get all the necessary data in):
## training set:
negative = [
    "just totally dull",
    "completely predictable and lacking energy",
    "no surprises and very few laughs",
]

positive = ["very profound", "the most fun film of the summer"]

## test set:
test_set = "predictable with no originality"

## 2. tokenization:
neg_token = [word.split() for word in negative]
pos_token = [word.split() for word in positive]
test_token = test_set.split()

## 3. turn into dictionaries: (flatten):

# 3.1 flatten:
neg_flat = [item for sublist in neg_token for item in sublist]
pos_flat = [item for sublist in pos_token for item in sublist]

# 3.3 put stuff into dictionaries:
def make_dict(tokens):

    dict = Counter()

    for i in tokens:
        dict[i] += 1
    return dict


neg_dict = make_dict(neg_flat)
pos_dict = make_dict(pos_flat)
test_dict = make_dict(test_token)

## 4. get amount of words in each category (& total)

# 4.1 total:
total = Counter()
total.update(neg_dict)
total.update(pos_dict)
total_vocab = len(total)  # 20 words in total vocabulary size.

# 4.2 positive and negative (total words used):
# used later..
def returnSum(dict):

    sum = 0
    for i in dict:
        sum = sum + dict[i]

    return sum


## Q1: Compute priors:

negative_prior = 3 / 5
positive_prior = 2 / 5

## Q1: likelihood for each word given class:
def word_likelihood(dict_category, dict_words, total_vocab=20):

    ## compute words used in the category:
    words_training = returnSum(dict_category)

    ## find out how likely each word is:
    new_dct = Counter()
    for key in dict_words:
        frequency = dict_category[key]  # 0 if not found..
        smoothed_likelihood = (frequency + 1) / (words_training + total_vocab)
        new_dct[key] = smoothed_likelihood
    return new_dct


negative_likelihoods = word_likelihood(neg_dict, total)
positive_likelihoods = word_likelihood(pos_dict, total)

## Q2: test-set: ##
def total_likelihood(total_dict, dict_category, dict_words, prior):

    ## compute words used in category:
    words_training = returnSum(dict_category)
    total_vocab = len(total_dict)

    ## how likelily is each words:
    prior
    for key in dict_words:
        if total_dict[key] > 0:
            frequency = dict_category[key]
            smoothed_likelihood = (frequency + 1) / (words_training + total_vocab)
            prior = prior * smoothed_likelihood
        else:
            pass

    return prior


negative_on_test = total_likelihood(total, neg_dict, test_dict, 0.6)
positive_on_test = total_likelihood(total, pos_dict, test_dict, 0.4)

# negative most likely:
print(negative_on_test, positive_on_test)

## Q3:
""" 
** Would using binary multinomial Naive Bayes change anything? **

Not clear whether it would change the actual outcome 
in this case (would have to be tested). It does change 
counts because we would have fewer words in both 
the positive and the negative category (as we do have duplicates).
In negative we have (and, and) and in the positive 
group we have (the, the). 
"""

## Q4:
"""
** Why do you add |V| to the denominator of add-1 smoothing, 
instead of just counting the words in one class? **

If we don't normalize by vocabulary size that would 
favor a category with no words in it which shares vocabulary
with a very big category (which then has many words). 
The category without any words in it would then always
have a probability of 100% for any word which does occur 
in the other category and happens to come in the test set. 

e.g. 
positive_words = ... 
negative_words = "i", "am", "upset", "you", "the", "worst"
test_words = "i", "am", "mean" 

if we did not normalize then for positive words
we would have a 100% chance of all the test words,
since there is no divisor. 

"""

## Q5: What would the answer to question 2 be without add-1 smoothing?


def no_smooth_likelihood(dict_category, dict_words, prior, total_vocab=20):

    ## compute words used in category:
    words_training = returnSum(dict_category)

    ## how likelily is each words:
    prior
    for key in dict_words:
        frequency = dict_category[key]
        unsmoothed_likelihood = (frequency) / (words_training + total_vocab)
        prior = prior * unsmoothed_likelihood

    return prior


negative_on_test = no_smooth_likelihood(neg_dict, test_dict, 0.6)
positive_on_test = no_smooth_likelihood(pos_dict, test_dict, 0.4)

# well, none of them: both have posterior of 0%.
# that is why we use add-1 smoothing...
print(negative_on_test, positive_on_test)


## Q6:
"""
** Can you think of any other features (or preprocessing) 
that you could add that might be useful in predicting sentiment? 
(This will come in handy for next HW!). ** 

* capitalization (shouting)
* emoticons (sentiment, obviously)
* negation (obviously)
"""

## Q7:

"""
** Naive Bayes treats words as if they are independent 
conditioned upon the class (that is why we multiply 
the individual probabilities). For which (if any) 
of the new features you suggested does this 
independence assumption roughly hold? **

* could hold for capitalization 
* also ok for emoticons 
* does not hold for negation (definitely pertaining to structure).
"""

## PART 2: Challenge Problems
