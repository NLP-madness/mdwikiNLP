"""
"""
import sentencepiece as spm
import os

with open('../class_01/harry_potter_sorcerers_stone.txt', "r") as f:
    txt = f.read()
spm.SentencePieceTrainer.train('../class_01/harry_potter_sorcerers_stone.txt')



import urllib.request
import io
import sentencepiece as spm

# Loads model from URL as iterator and stores the model to BytesIO.
model = io.BytesIO()
with urllib.request.urlopen(
    'https://raw.githubusercontent.com/google/sentencepiece/master/data/botchan.txt'
) as response:
  spm.SentencePieceTrainer.train(
      sentence_iterator=response, model_writer=model, vocab_size=1000)

# Serialize the model as file.
# with open('out.model', 'wb') as f:
#   f.write(model.getvalue())

# Directly load the model from serialized model.
sp = spm.SentencePieceProcessor(model_proto=model.getvalue())
print(sp.encode('this is test'))


import tensorflow as tf
import tensorflow_text as text

tf.modules.

tf.text.SentencepieceTokenizer

model = open('test_model.model', 'rb').read()
s1 = text.SentencepieceTokenizer(model=model)
print(s1.tokenize(['hello world']))
print(s1.tokenize_with_offsets(['hello world']))

s2 = text.SentencepieceTokenizer(model=model, out_type=tf.dtypes.string)
print(s2.tokenize(['hello world']))
print(s2.tokenize_with_offsets(['hello world']))

tf.keras.preprocessing.text.Tokenizer
tf.keras.preprocessing.text.se