# Class 9: Huggingface Transformers

Note: This lecture will take place on zoom at 8.15 using the meeting ID: 314 159 1711 or this [link](https://aarhusuniversity.zoom.us/j/3141591711). Furthermore changes to the plan may occur but required readings will not change.

### TL:DR
 - Required:
   - Read the [white paper](https://arxiv.org/abs/1910.03771) on the Transformer Library
 - Recommended:
   -  Read the original paper on transformers; [Attention is all You Need](https://arxiv.org/abs/1706.03762)
   -  Read the original paper on BERT; [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
 - Optional
   -  

---

## Plan for Class

In this and the following class we will use the `transformers` and `PyTorch` package by Huggingface 🤗 to do Named Entity Recognition classification on token sequences. I will start of from where the lecture left of so the plan for the class will be updated after the class itself.
<!--
- Classification of NER using Embeddings and a feed-forward NN
- Classification of NER using LSTM
- Classification of NER using BERT 
- Recent developments in Transformer (transformers are RNN and hopfield networks)
  - What are the implication that attention == memory?
-->

---

## HuggingFace Transformers
HuggingFace is an NLP company which have the open-source library `transformers` as well as a couple of other convenient libraries for NLP. The `transformers` library is very easy to use and is a good place to start with transformers. It also allows you to dig deeper either using Google's `Tensorflow` or Facebook's `PyTorch`. For the class please read Huggingface's white paper on their `transformers` library ([ref](https://arxiv.org/abs/1910.03771))

## Why PyTorch?
Why use PyTorch as opposed to Tensorflow with Keras? Naturally Keras is generally considered a higher-level API compared to PyTorch, personally I believe PyTorch has the right level of abstraction where errors are somewhat understandable while for Keras errors almost always require you to do a google search. Keras seems to penalize the user too much when they seek to go deeper while for PyTorch the transition seems easier. PyTorch is in general used more en research and Tensorflow is used more in production.

---

## Materials used in Class
[exercises.ipynb](https://github.com/auNLP/mdwikiNLP/blob/master/classroom_materials/class_09/exercise.ipynb)


---
## Related Material
- LSTM using Pytorch ([ref](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html))
- MNIST feed-forward Neural Net using PyTorch ([ref](https://www.youtube.com/watch?v=oPhxf2fXHkQ))
- Use preloaded word-embeddings with pyTorch ([ref](https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76))
- The Annotated Transformer by Harvard ([ref](http://nlp.seas.harvard.edu/2018/04/03/attention.html))