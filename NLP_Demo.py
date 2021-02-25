"""Python text to speech using NLP"""
"""Imports"""

import nltk
import nltk.corpus
# Tokenizer
from nltk.tokenize import word_tokenize

str1='If my eyes could show my soul, everyone would cry when they saw me smile.'
"""Word tokenize"""
print('Word token:')
print(word_tokenize(str1))
# Number Of Token
print('len of token/ Number of token:')
print(len(word_tokenize(str1)))
# Sentence Tokenizer
from nltk.tokenize import sent_tokenize
str2="If you die you’re completely happy and your soul somewhere lives on. I’m not afraid of dying. Total peace after death, becoming someone else is the best hope I’ve got."
print('Sentence token:')
# print(sent_tokenize(str2))
# Bigram and N-Grams

str3="Can anybody hear me? Am I talking to myself! I'm getting mad."
str3_token= word_tokenize(str3)
"""Bigram actually giving 2 token at a time in the list"""
str3_bigram= list(nltk.bigrams(str3_token))
print('Following Bi-gram:')
# print(str3_bigram)
""" Trigram similarly passes 3 token at a time"""
str3_trigram= list(nltk.trigrams(str3_token))
# Printing the trigram
print('Following Tri-gram:')
# print(str3_trigram)
""" N-gram gives freedom how many tokens you wanted to have in your list if it's more than 3. 
Here I'm habig 5 par tuple"""
str3_ngram=list(nltk.ngrams(str3_token,5))
print('Following n-gram with 5 tokens: ')
# print(str3_ngram)
"""For finding the root word, we have to finfd out the stem/root 
Here comes stemming"""
from nltk.stem import PorterStemmer
# Initiating an object
my_stem= PorterStemmer()
# Printing root word
print(my_stem.stem('eating')) # Same goes going, shopping etc. Don't work for tense change and goes etc
tom="If you die you’re completely happy and your soul somewhere lives on"
tom_token= word_tokenize(tom)
"""Finds parts of speech of the following sentence. word--abbriviation
 like you-- proper pronoun"""
print(nltk.pos_tag(tom_token))

"""Name entity recognization"""

from  nltk import ne_chunk
pres= "Barack Obama is the 45th president of America"
pres_token=word_tokenize(pres)
pres_token_pos=nltk.pos_tag(pres_token)
"""Recongnize Donald trump as a entity"""
print(ne_chunk(pres_token_pos))
from gtts import gTTS
from IPython.display import Audio
tts=gTTS("If you die you’re completely happy and your soul somewhere lives on")
tts.save('test.wav')
sound_file='test.wav'
Audio(sound_file,autoplay=True)




