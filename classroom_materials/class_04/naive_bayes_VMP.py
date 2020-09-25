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

# pseudo-code:
dct = {}
for i in test["text"]:
    t = Text(i)
    t_dict = t.token_frq()
    dct.update(t_dict)

dct

for i in train:
    t = Text(i)
    # optionally n-gram frequencies.
    t.token_frequencies()


## python challenge (numpy to condense script - fast).
