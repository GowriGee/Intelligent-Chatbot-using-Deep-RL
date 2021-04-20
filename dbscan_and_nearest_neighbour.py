# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 15:15:51 2021

@author: athir
"""

from gensim.models import FastText
from sklearn.neighbors import NearestNeighbors
import nltk
import numpy as np 
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
  

f = open('pride.txt','r',encoding='utf8')
article_text = f.read()
processed_article = article_text.lower()
p=nltk.sent_tokenize(processed_article)

sentences=[]
for sent in p:
   sentences.append(nltk.word_tokenize(sent))
   
#Vectorization using fasttext
   
embedding_size = 60
window_size = 30
min_word = 1
down_sampling = 1e-2


model=FastText(sentences,size=embedding_size,
                      window=window_size,
                      min_count=min_word,
                      sample=down_sampling,
                      sg=1,
                      iter=100)
  
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
     
    return np.asarray(sent_vec) / numw
  
  
X=[]
for sentence in sentences:
    X.append(sent_vectorizer(sentence, model))   


neigh = NearestNeighbors(n_neighbors=5)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
plt.show()

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
 
Y=model.fit_transform(X)

dbscan = DBSCAN(metric='cosine', eps=0.7, min_samples=5) # you can change these parameters, given just for example 
cluster_labels = dbscan.fit_predict(X)
core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True
labels=dbscan.labels_
for index, sentence in enumerate(sentences):
    if (labels[index]==1):
        print(labels[index],sentence)
plt.scatter(Y[:, 0], Y[:, 1], c=cluster_labels, cmap="plasma")

plt.show()
