# Class 7: Word Embeddings

Note: This lecture will take place on zoom at 8.15 using the meeting ID: 626 2173 3545 or this [link](https://aarhusuniversity.zoom.us/j/62621733545).


### TL:DR
 - Required:
   - Make sure you have zoom installed an working prior to the class
   - make sure you have downloaded the Danish word embeddings before class. You might also be interested in downloading English the [word embedding](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) by Google if you want to examine them.
   - watch the first four videos in this [playlist](https://www.youtube.com/playlist?list=PLhWB2ZsrULv-wEM8JDKA1zk8_2Lc88I-s)
     - answer the three question for the videos prior to class (see the section: 'Playlist on Word Embeddings')
  
 - Highly recommended:
   - read this wonderful [article of Word2Vec](http://jalammar.github.io/illustrated-word2vec/)
 - Recommended:
   -  watch the rest of the videos in the [playlist](https://www.youtube.com/playlist?list=PLhWB2ZsrULv-wEM8JDKA1zk8_2Lc88I-s)
 - Optional
   - Read the paper on [Word2Vec](https://arxiv.org/abs/1310.4546) by T. Mikolov or see the [video reading](https://www.youtube.com/watch?v=yexR53My2O4) of the paper.
   - Read up on the application of cross-lingual word embeddings for instance in machine translation without training data.
   - Read up alternative techniques for word embeddings: GloVe [\[1\]](https://nlp.stanford.edu/pubs/glove.pdf) and fasttext [\[2\]](https://arxiv.org/abs/1607.04606), [\[3\]](https://arxiv.org/abs/1607.01759) [\[4\]](https://arxiv.org/abs/1612.03651)

---

## Plan for Class
This class will take place on zoom at 8.15 using the meeting ID: 626 2173 3545 or this [link](https://aarhusuniversity.zoom.us/j/62621733545). I encourage that you meet up with your study group. The call will be recorded for students who are not present unless someone voice an opinion against this.

Prior to the class you should have some idea of what a word embedding is and what are some of the desired properties. For class we will go through:
- Question for the videos
  - what is a word embedding?
- Validation of word embeddings? Do they contain meaningful information?
  - With examples using Danish word embeddings
- How to train word embeddings using NN
- (A short presentation on how word embedding can be used for an exam project)


---

## Word embedding techniques
Word2Vec is the most well-known word embeddings technique. I was introduced in a series of papers but the typically referred to paper is ['Distributed Representations of Words and \[...\]'](https://arxiv.org/abs/1310.4546), which I recommend reading. This paper is a bit difficult to read to so I recommend the [video reading](https://www.youtube.com/watch?v=yexR53My2O4) by Yannic Kilcher.

Other popular techiques include GloVe [\[1\]](https://nlp.stanford.edu/pubs/glove.pdf) and fasttext [\[2\]](https://arxiv.org/abs/1607.04606), [\[3\]](https://arxiv.org/abs/1607.01759) [\[4\]](https://arxiv.org/abs/1612.03651), which are also both worth getting familiar with.



## Playlist on Word Embeddings
For class you are expected to watch the first four videos in the [video series](https://www.youtube.com/playlist?list=PLhWB2ZsrULv-wEM8JDKA1zk8_2Lc88I-s) on word embeddings by Andrew NG, but I highly recommend that you watch the rest of the playlist as well. While you watch these please consider the following question an prepare an answer to the best of your ability:

- What is a word embedding?
- What are the desired properties of word embeddings?
- What are the potential uses of word embeddings?


## Cross-lingual Word Embeddings
Wish I had the time to go through Cross-lingual word embeddings. The idea is that you overlay word embeddings for two languages and then this act as a way of translating between languages. It is a really cool application of word embeddings so if you feel like digging into it I recommend this [podcast episode](https://soundcloud.com/nlp-highlights/57-a-survey-of-cross-lingual-word-embedding-models-with-sebastian-ruder) (in general this podcast is great). You can also watch Mannings PhD student give a [talk](https://www.youtube.com/watch?v=3wWZBGN-iX8&list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z&index=20) on how use word embeddings for translation without any training data.  


## Download Danish Word embeddings:
You can download the Danish word embeddings using: 
```
from danlp.models.embeddings  import load_wv_with_gensim
word_embeddings = load_wv_with_gensim('conll17.da.wv')
word_embeddings.save("word2vec.model")  # save it on your computer
```
There is also other versions of Danish word embedding available (e.g. one trained from wikipedia) these can be found on DaNLP's Github.

---

## Materials used in Class
To be announced

---
## Related Material

PyData talk: Beyond word2vec: GloVe, fastText, StarSpace - Konstantinos Perifanos ([link](https://www.youtube.com/watch?v=6xPnEh_tJEc))
