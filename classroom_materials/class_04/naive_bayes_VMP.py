"""
prepping for class
"""
from collections import Counter

import pandas as pd
import numpy as np
import sys
import sklearn as sk 

sys.path.insert(1, "../class_03/")
from VMP_class_text import Text 

## Split into bing and bong. 
def splitter(df, test_pct = 0.2, downsample = True):
    
    # create mask (from Kenneth): 
    mask = np.random.choice([0, 1], p=[1-test_pct, test_pct],
                            size=df.shape[0])

    # apply mask
    df["mask"] = mask
    test_df = df[df["mask"] == 1]
    train_df = df[df["mask"] == 0]


## Reading the data: 
def read_split(txt, test_pct = 0.2): 

    #reading data: 
    data = pd.read_csv(txt, encoding = "latin-1") #latin-1? 
    data = data[["v1", "v2"]] #this is where the information is.
    data.columns = ["category", "text"]

    #splitting (if we do so): 
    if test_pct: 
        pass 
    return data 


reader("spam.csv")

#pseudo-code: 
for each train: 
    t = Text(test)
    # optionally n-gram frequencies. 
    t.token_frequencies() 



## python challenge (numpy to condense script - fast). 
