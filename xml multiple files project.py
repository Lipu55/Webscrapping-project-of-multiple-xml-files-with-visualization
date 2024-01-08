# -*- coding: utf-8 -*-
"""
Created on Mon May  1 23:49:46 2023

"""

import os
import numpy as np
import pandas as pd
import bs4 as bs
import urllib.request
import re
import spacy
import re,string,unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lem=WordNetLemmatizer()
os.chdir(r"D:\Datascience Classes\1st may\xml_many articles")
from glob import glob
path=r"D:\Datascience Classes\1st may\xml_many articles"
all_files=glob(os.path.join(path, "*.xml"))
import xml.etree.ElementTree as ET
dfs=[]
for filename in all_files:
    tree=ET.parse(filename)
    root=tree.getroot()
    root=ET.tostring(root,encoding='utf8').decode('utf8')
    dfs.append(root)
dfs[0]
############
import bs4 as bs
import urllib.request
import re
parsed_article=bs.BeautifulSoup(dfs[0],'xml')
paragraphs=parsed_article.find_all('p')

article_text_full= ""

for p in paragraphs:
    article_text_full += p.text
    print(p.text)

def data_prepracessing(each_file):
    parsed_article=bs.BeautifulSoup(each_file,'xml')
    paragraphs=parsed_article.find_all('para')
    
    article_text_full= ""
    
    for p in paragraphs:
        article_text_full += p.text
        print(p.text)
        
    return article_text_full
data=[data_prepracessing(each_file) for each_file in dfs]
#=========== after combine all the articles ===========
from bs4 import BeautifulSoup
soup=BeautifulSoup(dfs[0], 'html.parser')
print(soup.prettify())
parsed_article=bs.BeautifulSoup(dfs[0],'xml')
paragraphs=parsed_article.find_all('para')

def remove_stop_word(file):
    nlp=spacy.load("en_core_web_sm")
    
    punctuations=string.punctuation
    from nltk.corpus import stopwords
    stopwords=stopwords.words('english')
    SYMBOLS= " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”"]
    stopwords=nltk.corpus.stopwords.words('english')+SYMBOLS
    
    doc=nlp(file, disable=['parser','ner'])
    tokens=[tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
    tokens=[tok for tok in tokens if tok not in stopwords and tok not in punctuations]
    s=[lem.lemmatize(word) for word in tokens]
    tokens= ' '.join(s)
    
    article_text=re.sub(r'\[[0-9]*\]', ' ',tokens)
    article_text=re.sub(r'\s+', ' ', article_text)
    
    formatted_article_text=re.sub('[^a-zA-Z]', ' ', article_text)
    formatted_article_text=re.sub(r'\s+', ' ', formatted_article_text)
    formatted_article_text=re.sub(r'\W*\b\w{1,3}\b', "",formatted_article_text)
    
    return formatted_article_text

clean_data=[remove_stop_word(file) for file in data]
all_words= ' '.join(clean_data)
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import pandas as pd
vectorizer=CountVectorizer(stop_words=stopwords.words('english'),ngram_range=(2,2)).fit(clean_data)
X=vectorizer.transform(clean_data).toarray()
data_final=pd.DataFrame(X,columns=vectorizer.get_feature_names())
#### machine learning package along with NLP 

from sklearn.feature_extraction.text import TfidfTransformer
tran=TfidfTransformer().fit(data_final.values)
X=tran.transform(X).toarray()
X=normalize(X)

from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

distortions=[]
inertias=[]
mapping1={}
mapping2={}
K=range(1,15)

for k in K:
    #Building and fitting the model 
    kmeanModel=KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    
    distortions.append(sum(np.min(cdist(X,kmeanModel.cluster_centers_,
                      'euclidean'),axis=1)) / X.shape[0])
    inertias.append(kmeanModel.inertia_)
    
    mapping1[k]=sum(np.min(cdist(X,kmeanModel.cluster_centers_,
                      'euclidean'),axis=1)) / X.shape[0]
    mapping2[k]=kmeanModel.inertia_

for key,val in mapping1.items():
    print(str(key)+' : '+str(val))
plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()


from sklearn.cluster import KMeans
true_k=11
model=KMeans(n_clusters=true_k,init='k-means++',max_iter=100,n_init=1)
model.fit(X)

cluster_result_data=pd.DataFrame(clean_data,columns=['text'])
cluster_result_data['group']=model.predict(X)

order_centroids=model.cluster_centers_.argsort()[:, ::-1]
terms=vectorizer.get_feature_names()


for i in range(true_k):
    print('Cluster %d:' % i),
    for ind in order_centroids[i, :50]:
        print(' %s' % terms[ind])
        
        
def n_gram_score(file):
    nlp=spacy.load("en_core_web_sm")
    
    punctuations=string.punctuation
    from nltk.corpus import stopwords
    stopwords=stopwords.words('english')
    SYMBOLS= " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”"]
    stopwords=nltk.corpus.stopwords.words('english')+SYMBOLS
    
    doc=nlp(file,disable=['parser', 'ner'])
    tokens= [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
    tokens=[tok for tok in tokens if tok not in stopwords and tok not in punctuations]
    s=[lem.lemmatize(word) for word in tokens]
    tokens= ' '.join(s)
    
    
    article_text=re.sub(r'\[[0-9]*\]', ' ',tokens)
    article_text=re.sub(r'\s+', ' ', article_text)
    
    formatted_article_text=re.sub('[^a-zA-Z]', ' ', article_text)
    formatted_article_text=re.sub(r'\s+', ' ', formatted_article_text)
    formatted_article_text=re.sub(r'\W*\b\w{1,3}\b', "",formatted_article_text)
 
    
    n_grams = ngrams(word_tokenize(formatted_article_text), 2)
    get_ngrams=[ ' '.join(grams) for grams in n_grams]
    
    
    
    word_frequencies={}
    for word in get_ngrams:
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word]=1
            else:
                word_frequencies[word] +=1
                
    maximum_frequency=max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word]=(word_frequencies[word]/maximum_frequency)
    
    nlp=spacy.load("en_core_web_sm")
    doc=nlp(file)
    sentence_list=[str(sentence) for idno, sentence in enumerate(doc.sents)]
        
        
    sentence_scores={}
    for sent in sentence_list:
        sent_grams=ngrams(word_tokenize(sent), 2)
        sent_ngrams=[ ' '.join(grams) for grams in sent_grams]
            
        for word in sent_ngrams:
            if word in word_frequencies.keys():
                    
                if sent not in sentence_scores.keys():
                    sentence_scores[sent]=word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]
                    
    mean_value=np.mean(list(sentence_scores.values()))
    return mean_value

cluster_result_data['N_gram_score']=[n_gram_score(sen) for sen in data]
                
from wordcloud import WordCloud

normal_words=' '.join([text for text in cluster_result_data.loc[cluster_result_data['group']==0,'text'] ])
wordcloud=WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(normal_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis('off')
plt.show()

import matplotlib.pyplot as plt

for num in range(0,6):
    
    normal_words=' '.join([text for text in cluster_result_data.loc[cluster_result_data['group']==num,'text'] ])
    wordcloud=WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(normal_words)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud,interpolation="bilinear")
    plt.axis('off')
    plt.title('group:'+str(num))
    plt.show()

def token(sentence):
    tok=sentence.split
                        
    return tok

cluster_result_data['words']=[token(sentance) for sentance in cluster_result_data['text']]
