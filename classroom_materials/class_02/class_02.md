# Class 2: Preprocessing, Counts, and Collocations 


### TL:DR
 - Required:
   - Download the Danish and English model for Stanza prior to class
   - Make sure you have joined the Element forum (Matrix)
   - Make sure you have everything set up from [class 1](classroom_materials/class_01/class_01.md)
 - Required if not familiar with these: 
   - GitHub:
     - Watch the Youtube series: [Using GitHub with Visual Studio Code](https://www.youtube.com/watch?v=3Tn58KQvWtU&list=PLpPVLI0A0OkLBWbcctmGxxF6VHWSQw1hi)
     - Set up a GitHub with your studygroup and discover/resolve any issues you might have
   - Live Share
     - Watch video: [Live Share with Visual Studio Code](https://www.youtube.com/watch?v=8Ck2QhMxAYg)
     - Set up a live share session with your studygroup and discover/resolve any issues you might have
 - Recommended: 
   - Read the paper on [stanza](https://arxiv.org/abs/2003.07082) (6 pages)
   - Look into regular expressions. For example [this 5 minute video](https://www.youtube.com/watch?v=UQQsYXa1EHs) or [this 6 minute read](https://medium.com/better-programming/introduction-to-regex-8c18abdd4f70)
   - Set up a [linter](https://code.visualstudio.com/docs/python/linting), i.e. code spell-checker for VS code. I use flake8 but you can use any.

---

## Plan for Class

In this class and the following we will build a `Text` class in Python which contains a which will be able to extract a variety of feautures from text. These include:
- sentence segmentation
- tokenization
- n-grams
- Token frequencies
- Outputting to dataframe
- Lemmatization *using Stanza*
- Part-of-Speech (POS) *tagging using Stanza*
- Named Entity Recognition (NER) *using regular expressions*

The class will also introduce some introduction to the module `os` for navigating your operating system.


---

## Download models for Stanza
Before class you should have downloaded the Danish and English model for Stanford's NLP library Stanza. If you don't have stanza installed see [class 1](classroom_materials/class_01/class_01.md).

For a guide on downloading stanza models follow these step for a bit more control (or if you have any issues) see their online [guide](https://stanfordnlp.github.io/stanza/download_models.html):
- Open your python editor of choice
- Run the following:

```
import stanza
stanza.download('en')
stanza.download('da')
```
<!---
should be:

```python 
import stanza
stanza.download('en')
stanza.download('da')
```

but python formatting intentionally left out as it breaks the page
-->

## Github with Visual Studio Code
So instead of spending time in introducing GitHub in class. I recommend everyone check out the Youtube series: [Using GitHub with Visual Studio Code](https://www.youtube.com/watch?v=3Tn58KQvWtU&list=PLpPVLI0A0OkLBWbcctmGxxF6VHWSQw1hi). Note that the last video gives a couple of shorthands for creating and cloning repositories which are very convenient. I recommmend you also try creating a GitHub with your studygroup and spent time resolving any issues. For a *much more* extensive introduction see this [video](https://www.youtube.com/watch?v=RGOj5yH7evk).


As always you are free to use git however you like just so long that you are able to collaborate with your studygroup online. This is important given that future classes might end up taking place online due to Covid-19. 


## Live Share with Visual Studio Code
So instead of spending time in introducing Live Share in class. I recommend everyone check out the video: [Live Share with Visual Studio Code](https://www.youtube.com/watch?v=8Ck2QhMxAYg), which should give you a decent introduction. I recommmend you starting a session with your studygroup and see if you can:

1) See what eachother is working on 
2) Run code
3) See the result of code run by others
4) Open script located in the folder on the sharing parts laptop


As always you are free to not use Live Share just so long that you are able to collaborate with your studygroup online. This is important given that future classes might end up taking place online due to Covid-19.


## Code Spell-Checking
In VS code you can use a code spell-checker called a linter which will help debug and write prettier code. I recommend enabling this especially when working in groups. It might however cause some initial frustration, but I promise you that your code will be more readable after. Companies typically mandate some kind of linter so getting it under your skin is recommended. You can set it up following [these instructions](https://code.visualstudio.com/docs/python/linting). I personally use and recommend flake8, but using any linter is fine. 
---

## Materials used in Class
Python script with exercises: [text_preprocessor.py](https://github.com/auNLP/mdwikiNLP/blob/master/classroom_materials/class_02/text_processor.py)

Python script with a Stanza example: [stanza_example.py](https://github.com/auNLP/mdwikiNLP/blob/master/classroom_materials/class_02/stanza_example.py)

<!--
Things to say before class
- any questions on GitHub/Live share?
- linting
- go through exercise script
-->