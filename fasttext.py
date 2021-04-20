# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 10:43:20 2021

@author: athira
"""

import re
import nltk
import unicodedata
import inflect
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import FastText
import numpy as np

# DEMO SHOWN IN S7 ( LINES 20 - 83)

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    return text

def remove_non_ascii(words):
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def lemmatize_verbs(words):
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    words=lemmatize_verbs(words)
    return words

f = open('pride.txt','r',encoding='utf8')
article_text = f.read()
processed_article = article_text.lower()
processed_article = strip_html(processed_article)
all_sentences = nltk.sent_tokenize(processed_article)

sentences=[]
for sent in all_sentences:
   words= nltk.word_tokenize(sent)
   sentences.append(normalize(words))

# VECTORIZATION USING FASTTEXT

embedding_size = 60
window_size = 30
min_word = 1
down_sampling = 1e-2

ft_model = FastText([words],
                      size=embedding_size,
                      window=window_size,
                      min_count=min_word,
                      sample=down_sampling,
                      sg=1,
                      iter=100)

# function that calculates vector form of each sentence
def sent_vectorizer(sent, model):
    sent_vec =[]
    numw = 0
    
    for w in sent:
        try:
            if numw == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            numw+=1
        except:
            pass
    
    return sent, np.asarray(sent_vec) / numw
  
  
X=[]
for sentence in sentences:
    sent,vector=sent_vectorizer(sentence, ft_model)
    X.append(vector)  

# 60-dimensional vector representation of the first sentence in the dataset
print (X[0])




