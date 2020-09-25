"""
prepping for class
"""
from collections import Counter

import pandas as pd
import numpy as np
import sys
import sklearn as sk
import os

# working directory stuff.
os.getcwd()
os.chdir(
    "/home/victormp/Dropbox/MastersSem1/NLP/mdwikiNLP/classroom_materials/class_04"
)

# importing the text class:
sys.path.insert(1, "../class_03/")
from VMP_class_text import Text


## Split into training and testing:
## Could also look at sklearn (inbuild function).
def splitter(df, test_pct=0.2, downsample=True):

    # create mask (from Kenneth):
    mask = np.random.choice(
        [0, 1], p=[1 - test_pct, test_pct], size=df.shape[0]
    )  # size?

    # apply mask
    df["mask"] = mask  # new column?
    test_df = df[df["mask"] == 1]
    train_df = df[df["mask"] == 0]

    # removing the column again:
    test_df = test_df.drop("mask", axis="columns").reset_index()
    train_df = train_df.drop("mask", axis="columns").reset_index()

    return train_df, test_df


## Reading the data:
def read_split(txt, test_pct=0):

    # reading data:
    data = pd.read_csv(txt, encoding="latin-1")  # latin-1?
    data = data[["v1", "v2"]]  # this is where the information is.
    data.columns = ["category", "text"]

    # splitting (if we do so):
    if test_pct:
        test, train = splitter(data, test_pct)
        return test, train

    return data


# how we want to use it:
test, train = read_split("spam.csv", test_pct=0.2)
test.head()
train.head()

## add option to do by n-grams:
def make_dct(df):
    dct = {}
    for i in df["text"]:
        t = Text(i)
        t_dict = t.token_frq()
        dct.update(t_dict)
    return dct


## run on the whole training set.
def voc_size(df):
    dct = {}
    for i in df["text"]:
        t = Text(i)
        t_dict = t.token_frq()
        dct.update(t_dict)
    voc_size = len(dct)
    return voc_size


### TRYING OUT STUFF ###

ham_dct
j = "ham"
f"{j}_dct"

uniqueValues = train["category"].unique()

dct_list = []
for i in range(len(uniqueValues)):
    levels = f"{uniqueValues[i]}_dct"
    vars()[levels] = {}

    dct_list.append(levels)

dct_list

### Function not ready yet (below)

## train as input, split to categories:
def dct_categories(df, column_class, column_txt):

    ## finding number of categories:
    unique_cat = df[column_class].unique()

    for i in range(len(unique_cat)):
        levels = f"{unique_cat[i]}_dct"
        vars()[levels] = {}

    for n, i in enumerate(train[column_txt]):
        # find the class that i belongs to
        iclass = train[column_class][n]

        # create the token frequencies dct:
        instance = Text(i)
        token_frequencies = instance.token_frq()

        for j in unique_cat:
            if iclass == j:
                break
            else:
                pass
        #

    # return list_of_dicts


## does it work??
print(dct_categories(train, "category", "text"))


# check that the functions work:
train_dct, test_dct = dct_train_test(train, test)

## Naive Bayes
