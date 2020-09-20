# Class 3: Lemmatization and tagging


### TL:DR
 - Required:
   - Make sure you have download the Danish and English model for Stanza prior to class (see [class 2](classroom_materials/class_02/class_02.md))
   - Make sure that your studygroup have liveshare (or alternative methods) ready for collaboration. Liveshare should allow you all to run and write code.
 - Required if not familiar:
   - Make sure you are familiar with python classes, either using [learnpython.org](https://www.learnpython.org/en/Classes_and_Objects) or using [this video](https://www.youtube.com/watch?v=ZDa-Z5JzLYM)
   - Make sure you are familiar with pandas dataframes, e.g. using [learnpython.org](https://www.learnpython.org/en/Pandas_Basics)
 - Recommended: 
   - Download and set up [Kite](https://www.kite.com)
   - Turn on [format on save](https://code.visualstudio.com/updates/v1_49#_only-format-modified-text), which auto format you code when saved
   - If you find your python skills challenged I would recommend watching [this video](https://www.youtube.com/watch?v=rfscVS0vtbw) or looking into [this free book](https://automatetheboringstuff.com)
 - Optional:
   - read up on state-of-the-art Tokenization and implement it (see section). It is not that hard.

---
## Plan for Class
We will continue where we left of in the previous class.
  - Kite, what is it and why it might be useful
  - Script structure and importing functions
  - Introduction to the `Text` class and Pandas
  - Group work on the `Text` class and functions

before the next class (class 3) you should have finished the `Text` class and the functions:

`n_grams()`

`token_frequencies()`

`lemmatize_stanza()`

`postag_stanza()`


---

## Kite
[Kite](https://www.kite.com) is an intelligent autocomplete for Python. It contains a short tutorial when installed. As it is very short I recommend going through it. Kite is free and you can even use their pro license with if you create a user using a valid AU email.


## Format on Save
VS codes latest update ([august 2020](https://code.visualstudio.com/updates/v1_49#_only-format-modified-text)) includes a format on save functionality which autoformats you code when you save. This should hopefully save you quite a bit of time. When turned on and opening a python script it will prompt you to install an autoformat tool for python, I would recommend black.


## (Optional) State-of-the-art Tokenization
Tokenization is hard and many modern language models uses techniques which itself learn the tokenization, but also include sub-word tokens. To read more on this see [this blog post](https://blog.floydhub.com/tokenization-nlp/). For the interested I would recommend trying to implement the BPE algorithm for inspiration see [the original article](https://www.aclweb.org/anthology/P16-1162.pdf) presenting BPE for NLP.

## Materials used in Class
Python script with exercises: [text_preprocessor.py](https://github.com/auNLP/mdwikiNLP/blob/master/classroom_materials/class_02/text_processor.py)

Python script with a Stanza example: [stanza_example.py](https://github.com/auNLP/mdwikiNLP/blob/master/classroom_materials/class_02/stanza_example.py)


<!--
-->