# Class 8: Clustering

Note: This lecture will take place on zoom at 8.15 using the meeting ID: 314 159 1711 or this [link](https://aarhusuniversity.zoom.us/j/3141591711). Furthermore changes to the plan may occur but required readings will not change.


### TL:DR
 - Required:
    - Train a topic model following this [guide](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/), you have to do at least 1-13 (including), but I highly recommend going all the way to 17 (and while you are there why not read the rest 😉). If you do run the 14 change out the `c_v` coherence with `u_mass` for faster computation
 - Highly recommended:
   - watch this introduction the workings of [LDA](https://www.youtube.com/watch?v=T05t-SqKArY)
   - Bring your own data and even train the topic model on your own data before class
 - Recommended:
   -  Check out using sklearn for topic models using this [guide](https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730) 
 - Optional
   - read the original [paper](https://www.jmlr.org/papers/v3/blei03a) on LDA by David Blei
   - check out the original [paper](https://dl.acm.org/doi/abs/10.1145/1557019.1557121) on the Mallet algorithm (technical)

---

## Plan for Class

In this class we will examine Topic Modelling using Latent Dirichlet Allocation (LDA) not to be confused with Linear Discriminant Analysis. Before class you are expected to have trained your own topic model following this [guide] which will use `gensim`. Note that Scikit-Learn also have an implementation of LDA (see a guide [here](https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730)), but Gensim have more built-in functionality and a broader array of options and alternative model implementations. This guide is mostly so that you get through the steps of training a topic model.

During class we will instead focus on:
- Brief introduction to tagging software for exam projects
  - Including text classification and sequence labeling
- Discuss results obtained in the guide and any potential issues (short)
- Examine how topic models are trained
- Examining topics of topic models
- Application of topic models
- (Alternatives to topic models)
- (15 minutes guest talk)

*parenthesis denote potentially*

Note: that if you run the  
---



# Bring your own Data
By now I suspect some of you have started looking into your own projects. Topic modeling is a great way to get an overview of your data. I strongly suggest you bring your data if you have any. You are naturally free to run the topic model on the data in 

---

## Materials used in Class


---
## Related Material
Video: [Matti Lyra - Evaluating Topic Models](https://www.youtube.com/watch?v=UkmIljRIG_M)
Medium article: [Application of topic models](https://medium.com/@fatmafatma/industrial-applications-of-topic-model-100e48a15ce4#_edn1)


<!--
(We said topic modeling but you could also frame this more generally as unsupervised learning or clustering algs)
* Connect topic modeling to bayesian modeling
* Maybe invite Jan to present
-->
