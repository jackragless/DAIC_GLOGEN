#!/usr/bin/env python
# coding: utf-8

# In[5]:


from train_kw_extractor import ai_parse_utils

import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.tokenize.treebank import TreebankWordDetokenizer
import math
from textblob import TextBlob as tb

import pandas as pd
unigram = pd.read_csv('./data/unigram_freq.csv')
common_unigram = list(unigram[:10000]['word'])

def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)


# In[6]:


def manual_add_kw(wiki_object):
    kw_to_add = []
    for sent in nltk.sent_tokenize(wiki_object['text']):
        i = 0
        while i < len(nltk.word_tokenize(sent)):
            word = nltk.word_tokenize(sent)
            if len(word[i])>3 and word[i].upper()==word[i] and word[i][0].isalpha():
                kw_to_add.append(word[i])
                i += 1
            elif i>0 and word[i][0].isupper():
                temp_kw_phrase = []
                for j in range(i,len(word)):
                    if word[j][0].isupper():
                        temp_kw_phrase.append(word[j])
                    else:
                        break
                if len(temp_kw_phrase)>=3:
                    kw_to_add.append(TreebankWordDetokenizer().detokenize(temp_kw_phrase))
                    i += len(temp_kw_phrase)
                else:
                    i += 1 
            else:
                i += 1
                
    return kw_to_add


# In[7]:


def driver(corpus):

    tb_corpus = []
    count = 0
    length = len(corpus)
    for page in corpus:
        count += 1
        print(str(int(100*count/length)) + '%','<--->',count,'/',length,end='\r')
        if page['text'] and page['kw'] and page['title']:
            page['kw'] += manual_add_kw(page)
            temp_doc = ''
            for sent in nltk.sent_tokenize(page['text']):
                temp_tree = ai_parse_utils.parseSent(sent)
                if temp_tree:
                    candidate_phrases = [page[0] for page in ai_parse_utils.getPhraseNodes(temp_tree,[])]
                    for i in range(len(candidate_phrases)):
                        if nltk.word_tokenize(candidate_phrases[i])[0].lower() in stop_words:
                            candidate_phrases[i] = TreebankWordDetokenizer().detokenize(nltk.word_tokenize(candidate_phrases[i])[1:])
                    temp_sent = sent
                    for candidate in candidate_phrases:
                        temp_sent = temp_sent.replace(candidate,candidate.replace(' ','ZZZ'))
                temp_doc += ' ' + temp_sent
            tb_corpus.append(tb(temp_doc))
        else:
            corpus.remove(page)
            
            
    for i, blob in enumerate(tb_corpus):
        scores = {word: tfidf(word, blob, tb_corpus) for word in blob.words}
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        temp = []
        for word, score in sorted_words[:int(0.05*len(blob.words))]:
            if word.lower() not in common_unigram and word.lower() not in stop_words:
                temp.append(word.replace('ZZZ',' '))
        if corpus[i]['kw']:
            corpus[i]['kw'] += temp
        else:
            corpus[i]['kw'] = temp
        corpus[i]['kw'] = list(set(corpus[i]['kw']))
            
    return corpus


# In[8]:


# with open('/home/jackragless/projects/data/DAIC_GLOGEN/wiki_orig_mined_dataframe.pkl', 'rb') as f:
#     corpus = pickle.load(f)[0:100]
# final = driver(corpus)

