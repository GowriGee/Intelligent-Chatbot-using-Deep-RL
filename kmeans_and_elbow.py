
from gensim.models import FastText
from sklearn.cluster import KMeans
import nltk
import numpy as np 
import matplotlib.pyplot as plt
import re
import unicodedata
import inflect
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


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

# training data
f = open('pride.txt','r',encoding='utf8')
article_text = f.read()
processed_article = article_text.lower()
p=nltk.sent_tokenize(processed_article)

sentences=[]
for sent in p:
   words=nltk.word_tokenize(sent)
   sentences.append(words)


embedding_size = 60
window_size = 30
min_word = 1
down_sampling = 1e-2
#model = Word2Vec(sentences, min_count=1)
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
X=np.array(X)
  

inertias = []
mapping2 = {}
K = range(1, 15)
 
for k in K:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    inertias.append(kmeanModel.inertia_)
    mapping2[k] = kmeanModel.inertia_
    
plt.plot(K,inertias, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('The Elbow Method')
plt.show()

NUM_CLUSTERS=2

  
     
     
kmeans = KMeans(n_clusters=NUM_CLUSTERS,init='k-means++')
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
labels = kmeans.labels_
centers = kmeans.cluster_centers_
  
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.show()
 
