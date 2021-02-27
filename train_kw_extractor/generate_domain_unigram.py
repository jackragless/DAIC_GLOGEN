#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import nltk
import pandas as pd
import import_ipynb
import preprocess_utils_ai


# In[ ]:


def driver(corpus):
    
    loop_count = 0
    clean_text_obj = []

    for obj in corpus:
        loop_count += 1

        if obj['text']:
            clean_text_obj.append(preprocess_utils_ai.clean_text(obj['text'],False,True,False,True,True))
            
    
    all_words = []
    count = 0
    for obj in clean_text_obj:
        count += 1
        for sent in nltk.sent_tokenize(obj):
            sent = sent.replace('.','')
            all_words += nltk.word_tokenize(sent)
            
            
    unigram = pd.Series(all_words).value_counts()
    unigram_df = unigram.reset_index()
    unigram_df.columns=['word', 'freq']
    
    return unigram_df
    


# In[36]:


# with open('orig_wiki_corpus.pkl', 'rb') as f:
#     corpus = pickle.load(f)
    
# FINAL_OUTPUT = driver(corpus)
    
# unigram_df.to_csv('domain_specific_unigram.csv')

