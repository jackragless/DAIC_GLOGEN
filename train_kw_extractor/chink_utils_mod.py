#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from bs4 import BeautifulSoup as bs
import pandas as pd
import spacy
get_ipython().system('python3 -m spacy download en')
import en_core_web_sm
ner_model = en_core_web_sm.load()
import pickle
from names_dataset import NameDataset
name_dataset = NameDataset()
org_df = pd.read_csv('/home/jackragless/projects/data/DAIC_GLOGEN/companies_and_unis.csv')
unigram = pd.read_csv('/home/jackragless/projects/data/DAIC_GLOGEN/unigram_freq.csv')


# In[1]:


def kw_in_text_check(wiki_object):
    final = []
    for kw in wiki_object['kw']:
        if wiki_object['text'].lower().find(kw.lower())!=-1:
            final.append(kw)
            
    wiki_object['kw'] = final
    return wiki_object


# In[2]:


def consec_cap_detect(word_string):
    for i in range(0,len(word_string)-1):
        if word_string[i].isupper() and word_string[i+1].isupper():
            return True
    return False


# In[3]:


def person_detect(candidate_string):
    candidate_string = candidate_string.split()
    if len(candidate_string) == 2:
        if name_dataset.search_first_name(candidate_string[0]) == True and name_dataset.search_last_name(candidate_string[1]) == True:
            return True
    elif len(candidate_string) == 3:
        if name_dataset.search_first_name(candidate_string[0]) == True and len(candidate_string[1])>=2 and candidate_string[1][0].isalpha() and candidate_string[1][1] == '.':
            return True
        elif name_dataset.search_first_name(candidate_string[0]) == True and name_dataset.search_first_name(candidate_string[1]) and name_dataset.search_last_name(candidate_string[2]) == True:
            return True
    return False


# In[4]:


def common_word_detect(candidate_string): #could add stopwords
    common_unigram = list(unigram[:10000]['word'])
    if candidate_string.lower() in common_unigram:
        return True
    else:
        return False


# In[5]:


def location_name_detect(whole_text):
    final = []
    doc = ner_model(whole_text)
    for ent in doc.ents:
        if ent.label_ == 'GPE':
            final.append(ent.text)
    return final


# In[6]:


def misc_filters(candidate_string):
    if len(candidate_string) == 1:
        return True
    elif not any(c.isalpha() for c in candidate_string):
        return True
    return False


# In[7]:


def label_keyword_array(kw_arr, text):
    temp_dict = {}
    for i in kw_arr:
        if consec_cap_detect(i):
            temp_dict[i] = 'P'
        else:
            temp_dict[i] = 'K'
            
    for j in temp_dict:
        if (person_detect(j) or common_word_detect(j) or misc_filters(j)) and temp_dict[j]!='P':
            temp_dict[j] = 'R'
            
    location_ners = location_name_detect(text)
    for k in temp_dict:
        if k in location_ners and temp_dict[j]!='P':
            temp_dict[k] = 'R'
        elif k.lower() in org_df:
            temp_dict[k] = 'R'
        elif len(k.split()) > 4:
            temp_dict[k] = 'R'
    
    return temp_dict

