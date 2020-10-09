# Class 6: Scikit-learn and Vectorization


### TL:DR
 - Required:
   - Have your best model ready from last time
   - If you didn't do so for the last class split up in your study group and watch one video each on a machine learning method and meet back up and briefly present the model to each other. See the link in the subsection on class 5.
   - Watch [this short introduction](https://www.youtube.com/watch?v=WN18JksF9Cg) to TF-IDF using scikit-learn.
   - **please read the plan for class (changed due to covid-19)**
 - Highly recommended:
   - if unfamiliar watch this video on [PCA](https://www.youtube.com/watch?v=FgakZw6K1QQ)
 - Recommended:
   - Skim the documentation for [CountVectorizers](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) and [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer)
- Optional
  - Read up on AutoML and Auto-Sklearn, see the description below

During last class we only breifly touched upon text vectorization. The operation of turning text into vectors. Today we will look a bit more into this. Most notably the TF-IDF, which we briefly touched upon during our first class.

---

## Plan for Class
Due to the classroom size we will shift to a different classroom style similar to last class. 
The introduction will again be in a form of a video. Which will include:
* Examples of vectorization using n-grams, binary, stopwords and ones own tokenizer
* Introduction to TF-IDF using scikit-learn

Following the video the goal is to see which studygroup can gain the best performance using only the `multinomialNB` as a classifier. Thus you are only allowed to optimize the preprocessing rather than the model itself.

I will also add a video on using dimensionality reduction on the TF-IDF matrix, which you should watch halfway through the class
Again I will be avaliable during class if needed.
---



## AutoML
During this and the previous class we searched for the best performing models, however there actually exist very good systems which can perform a similar search. For an introduction you can watch [this video by statquest](https://www.youtube.com/watch?v=SEwxvjfxxmE). For the more interested user I recommend that you check out these papers on auto-sklearn [(1)](http://papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning.pdf) [(2)](https://library.oapen.org/bitstream/handle/20.500.12657/23012/1007149.pdf?sequence=1#page=119) as well as their [package](https://automl.github.io/auto-sklearn/master/). I haven't personally had time to experiment with this during class but AutoML is a growing field, also within NLP.

---

## Materials used in Class
Will be up later.


<!--
* one-hot
* tf-idf
-->
