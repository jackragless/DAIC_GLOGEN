#!/usr/bin/env python
# coding: utf-8

# In[ ]:



from train_kw_extractor import preprocess_utils_ai

from bs4 import BeautifulSoup as bs
import pandas as pd
import spacy
import en_core_web_sm
ner_model = en_core_web_sm.load()
import pickle
from names_dataset import NameDataset
name_dataset = NameDataset()
# org_df = pd.read_csv('/home/jackragless/projects/data/DAIC_GLOGEN/companies_and_unis.csv')
# unigram = pd.read_csv('/home/jackragless/projects/data/DAIC_GLOGEN/unigram_freq.csv')


# In[ ]:


def kw_in_text_check(wiki_object):
    final = []
    for kw in wiki_object['kw']:
        if wiki_object['text'].lower().find(kw.lower())!=-1:
            final.append(kw)
            
    wiki_object['kw'] = final
    return wiki_object


# In[ ]:


def clean_text(wiki_object):
    
    test_str = wiki_object['text']
    if test_str.find('Version history') != -1:
        result = [i for i in range(len(test_str)) if test_str.startswith('Version history', i)] 
        test_str = test_str[:result[-1]]
    elif test_str.find('See also') != -1:
        result = [i for i in range(len(test_str)) if test_str.startswith('See also', i)] 
        test_str = test_str[:result[-1]]
    elif test_str.find('References') != -1:
        result = [i for i in range(len(test_str)) if test_str.startswith('References', i)] 
        test_str = test_str[:result[-1]]
        
        
    clean_text = preprocess_utils_ai.clean_text(test_str,False,False,False,False,False)
    
    wiki_object['text'] = clean_text
    
    wiki_object['kw'].append(wiki_object['title'])
    
    return wiki_object


# In[ ]:


def consec_cap_detect(word_string):
    for i in range(0,len(word_string)-1):
        if word_string[i].isupper() and word_string[i+1].isupper():
            return True
    return False


# In[ ]:


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


# In[ ]:


def common_word_detect(candidate_string, unigram): #could add stopwords
    common_unigram = list(unigram[:10000]['word'])
    if candidate_string.lower() in common_unigram:
        return True
    else:
        return False


# In[ ]:


def location_name_detect(whole_text):
    final = []
    doc = ner_model(whole_text)
    for ent in doc.ents:
        if ent.label_ == 'GPE':
            final.append(ent.text)
    return final


# In[ ]:


def misc_filters(candidate_string):
    if len(candidate_string) == 1:
        return True
    elif not any(c.isalpha() for c in candidate_string):
        return True
    return False


# In[ ]:


def label_keyword_array(kw_arr, text, unigram):
    temp_dict = {}
    for i in kw_arr:
        if consec_cap_detect(i):
            temp_dict[i] = 'P'
        else:
            temp_dict[i] = 'K'
            
    for j in temp_dict:
        if (person_detect(j) or common_word_detect(j,unigram) or misc_filters(j)) and temp_dict[j]!='P':
            temp_dict[j] = 'R'
            
    location_ners = location_name_detect(text)
    for k in temp_dict:
        if k in location_ners and temp_dict[j]!='P':
            temp_dict[k] = 'R'
#         elif k.lower() in org_df:
#             temp_dict[k] = 'R'
        elif len(k.split()) > 4:
            temp_dict[k] = 'R'
    
    return temp_dict


# In[ ]:


def driver(corpus, unigram):
    
    #include only non-empty wiki objects
    #clean text in wiki_objects
    count = 0
    wiki_object = []
    for i in range(len(corpus)):
        if corpus[i]['text'] is not None and corpus[i]['kw'] is not None and corpus[i]['title'] is not None:
            wiki_object.append(clean_text(corpus[i]))
        count += 1
        print('CLEANING:', int(100*count/len(corpus)),'% <--->', str(count)+'/'+str(len(corpus)), end='\r')
        
    
    #chinking labels applied here
    count = 0
    for i in wiki_object:
        i['kw'] = label_keyword_array(i['kw'], i['text'],unigram)
        count += 1
        print('CHINKING:', int(100*count/len(wiki_object)),'% <--->', str(count) + '/' + str(len(wiki_object)), end='\r')
            
    
    #chink labels == 'R' are removed
    count = 0
    FINAL_OUTPUT = []
    for i in wiki_object:
        temp_arr = []
        for j in i['kw']:
            if wiki_object[count]['kw'][j] == 'K' or wiki_object[count]['kw'][j] == 'P':
                temp_arr.append(j)
        temp = wiki_object[count]
        temp['kw'] = temp_arr
        FINAL_OUTPUT.append(temp)
        count += 1
#         print(str(count) + '/' + str(len(wiki_object)), end='\r')
        
    return FINAL_OUTPUT


# In[ ]:


# with open('orig_wiki_corpus.pkl', 'rb') as f:
#     corpus = pickle.load(f)

# FINAL_OUTPUT = driver(corpus)

# with open('wiki_corpus_chinked.pkl', 'wb') as f:
#     pickle.dump(FINAL, f)

