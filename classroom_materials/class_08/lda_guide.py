# Run in python console
import nltk


import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


# 5 prepare stopwords
if nltk.download("stopwords"):
    from nltk.corpus import stopwords

# NLTK Stop words
stop_words = stopwords.words("english")
stop_words.extend(["from", "subject", "re", "edu", "use"])

# 6. Import Newsgroups Data
# Import Dataset
df = pd.read_json(
    "https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json"
)
print(df.target_names.unique())
df.head()

# 7. Remove emails and newline characters
# Convert to list
data = df.content.values.tolist()

# Remove Emails
data = [re.sub("\S*@\S*\s?", "", sent) for sent in data]

# Remove new line characters
data = [re.sub("\s+", " ", sent) for sent in data]

# Remove distracting single quotes
data = [re.sub("'", "", sent) for sent in data]

pprint(data[:1])

# 8. Tokenize words and Clean-up text
def sent_to_words(sentences):
    for sentence in sentences:
        yield (
            gensim.utils.simple_preprocess(str(sentence), deacc=True)
        )  # deacc=True removes punctuations


data_words = list(sent_to_words(data))

print(data_words[:1])

# 9. Creating Bigram and Trigram Models
# Build the bigram and trigram models
bigram = gensim.models.Phrases(
    data_words, min_count=5, threshold=100
)  # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data_words[0]]])

# 10. Remove Stopwords, Make Bigrams and Lemmatize
# Define functions for stopwords, bigrams, trigrams and lemmatization


def remove_stopwords(texts):
    return [
        [word for word in simple_preprocess(str(doc)) if word not in stop_words]
        for doc in texts
    ]


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(
            [token.lemma_ for token in doc if token.pos_ in allowed_postags]
        )
    return texts_out


# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load("en", disable=["parser", "ner"])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(
    data_words_bigrams, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]
)

print(data_lemmatized[:1])

# 11. Create the Dictionary and Corpus needed for Topic Modeling
# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])

# 12. Building the Topic Model
# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(
    corpus=corpus,
    id2word=id2word,
    num_topics=20,
    random_state=100,
    update_every=1,
    chunksize=100,
    passes=10,
    alpha="auto",
    per_word_topics=True,
)

# 13. View the topics in LDA model
# Print the Keyword in the 10 topics
print(lda_model.print_topics())
doc_lda = lda_model[corpus]

# 14. Compute Model Perplexity and Coherence Score
# Compute Perplexity
print(
    "\nPerplexity: ", lda_model.log_perplexity(corpus)
)  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(
    model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence="u_mass"
)
coherence_lda = coherence_model_lda.get_coherence()
print("\nCoherence Score: ", coherence_lda)


# 15. Visualize the topics-keywords
# Visualize the topics
# pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
pyLDAvis.save_html(vis, "lda.html")
