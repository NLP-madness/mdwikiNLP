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
def splitter(df, test_size = 0.2, downsample = True):
    sk.model_selection_ 


## Reading the data: 
def reader(df): 
    data = pd.read_csv(df, encoding = "latin-1") #latin-1? 
    data = data[["v1", "v2"]] #this is where the information is.


reader("spam.csv")

#pseudo-code: 
for each train: 
    t = Text(test)
    # optionally n-gram frequencies. 
    t.token_frequencies() 



## python challenge (numpy to condense script - fast). 
