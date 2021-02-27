# wikt_def_predict module operates on wiktionary output from wiky_def_parse
# predicts best definition based a keyword's POS / semantic comparison between candidate definition and source text


import nltk
import pickle
import wikipedia
import pandas as pd
import numpy as np
from semantic_text_similarity.models import ClinicalBertSimilarity

from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()

from glossary_generator import preprocess_utils, wikt_def_parse

import gensim.downloader
glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')



# initialise bert semanitc prediction model
clinical_model = ClinicalBertSimilarity(device='cuda', batch_size=10)



# returns word embedding similarity of source text title and definition domain 
def title_domain_predict(title, domain_arr, domain_exclusions):

    title_tokens = nltk.word_tokenize(title.lower())

    FINAL = []
    for dt in domain_arr:
        condition = False
        for excl in domain_exclusions:
            if excl in dt:
                condition = True
                break
                    
        if condition == True:
            FINAL.append(0)
        else: 
            temp3 = []
            for dtt in dt:
                temp2 = []
                for dttt in dtt.split():
                    temp1 = []
                    for tt in title_tokens:
                        if dttt in ['general','uncountable','countable']:
                            break
                        else:
                            try:
                                similarity = glove_vectors.similarity(dttt,tt)
                            except:
                                similarity = 0
                        temp1.append(similarity)
                    if len(temp1)>0:
                        temp2.append(np.mean(temp1))
                    else:
                        temp2.append(0.0)
                temp3.append(np.mean(temp2))
            FINAL.append(np.mean(temp3))
            
    return FINAL



# returns BERT semantic similarity of source text and definition text
def def_semantic_predict(curtext, clean_defs, domain_exclusions):
    final = []
    for i in clean_defs:
        for domain in domain_exclusions:
            if domain in nltk.word_tokenize(i):
                final.append(0)
                break
        else:
            final += list(clinical_model.predict([(i,curtext)]))
    return final



# returns BERT sematic similarity of source text and definition example
def ex_semantic_predict(curtext, clean_exs):

    final = []
    for ex in clean_exs:
        if ex == 'null':
            final.append(0)
        else:
            final += list(clinical_model.predict([(curtext, ex)]))
            
    return final



# selects best definition by combining above similarity measures
def select_best_index(predict1_probs, predict2_probs):
    
    argsort = np.argsort(predict1_probs)[::-1][:2]
    plus1dev = np.mean(predict2_probs) + np.std(predict2_probs)
    
    #this check ensures instances where example/domain data unavailable not penalised
    if len(predict2_probs)>1 and (predict2_probs[argsort[0]] == 0 or predict2_probs[argsort[1]]== 0):
        return argsort[0]
    
    #return element that is both top ranked by def_semantic_predict and +1stdev in secondary factor (domain/example)
    for i in argsort:
        if predict2_probs[i] >= plus1dev:
            return i
    return argsort[0]


def acronym_text_search(acr, text):
    if len(acr)>2:
        all_words = []
        for sent in nltk.sent_tokenize(text):
            if sent.find(acr)!=-1:
                focus_sent = preprocess_utils.clean_text(sent[:sent.find(acr)],False,True,True,False,False).replace('-',' ')
                break
        focus_words = nltk.word_tokenize(focus_sent)
        initials = ''.join([word[0] for word in focus_words]).upper()
        search = initials.find(acr)
        if search!=-1:
            return ' '.join(focus_words[search:search+len(acr)])
        
    return 'not-found'
    


def driver(title, curtext, kw, pos, depth):
    
    # converts NLTK POS acronyms to full words for wiktionary parsing
    pos_association = {
    'CC':['conjunction'],
    'CD':['numeral'],
    'DT':['determiner'],
    'EX':[],
    'FW':[],
    'IN':['preposition','conjunction'],
    'JJ':['adjective'],
    'JJR':['adjective'],
    'JJS':['adjective'],
    'LS':[],
    'MD':['verb'],
    'NN':['noun','proper noun'],
    'NNS':['noun'],
    'NNP':['noun', 'proper noun'],
    'NNPS':['noun', 'proper noun'],
    'PDT':['determiner'],
    'POS':[],
    'PRP':['pronoun'],
    'PRP$':['pronoun'],
    'RB':['adverb'],
    'RBR':['adverb'],
    'RBS':['adverb'],
    'RP':['preposition'],
    'TO':[],
    'UH':['interjection'],
    'VB':['verb'],
    'VBG':['verb'],
    'VBD':['verb'],
    'VBN':['verb'],
    'VBP':['verb'],
    'VBZ':['verb'],
    'WDT':['determiner'],
    'WP':['pronoun'],
    'WRB':['adverb'],
    'noun':['noun', 'proper noun'],
    'verb':['verb']
    }
    
    if kw.upper() == kw:
        acroynm_search_result = acronym_text_search(kw,curtext)
        if acroynm_search_result != 'not-found':
            return acroynm_search_result + '.'
    
    
    # error handling
    try:
        wikt_object = wikt_def_parse.define(kw)[pos]
    except IndexError:
        try:
            wikt_object = wikt_def_parse.define(kw.lower())[pos]
        except (IndexError,KeyError):
            try:
                # if a term is unavailable on Wikionary, try first sentence of Wikipedia
#                 wiki = '(wiki) ' + nltk.sent_tokenize(wikipedia.page(kw, auto_suggest=False).content)[0]
                wiki = nltk.sent_tokenize(wikipedia.page(kw, auto_suggest=False).content)[0] + '.'
                return wiki
            except:
                return 'invalid-term'
    except KeyError:
        return 'invalid-pos'
    
    
    orig_defs = [obj['def'] for obj in wikt_object]
    orig_exs = [obj['ex'] for obj in wikt_object]
    domain_arr = [preprocess_utils.clean_text(sent[:sent.find(')')],False,False,True,False,True).replace('.','').replace('(','').strip().split(', ')for sent in orig_defs]
    clean_defs = [preprocess_utils.clean_text(definition,False,True,True,True,True) for definition in orig_defs]
    clean_exs = [preprocess_utils.clean_text(example,False,True,True,True,True) for example in orig_exs]
    curtext = preprocess_utils.clean_text(curtext,True,True,True,True,True)
    
    #this list can be modified as desired to remove unwanted Wiktionary domains from prediction
    domain_exclusions = ['dated', 'obsolete', 'rare']
    
    
    predict1_probs = def_semantic_predict(curtext, clean_defs, domain_exclusions)
    

    # select to use ex_sematic_predict or title_domain_predict as secondary prediction data
    # based on whether more examples or domains available in given Wiktionary object
    no_ex_count = 0
    no_domain_count = 0
    for i in range(len(domain_arr)):
        if domain_arr[i][0] == 'general':
            no_domain_count += 1
        if clean_exs[i] == 'null':
            no_ex_count += 1
    if no_ex_count <= no_domain_count:
        predict2_probs = ex_semantic_predict(curtext, clean_exs)
    else:
        predict2_probs = title_domain_predict(title, domain_arr, domain_exclusions)
    
    
    best_index = select_best_index(predict1_probs, predict2_probs)
    prediction = orig_defs[best_index].replace('.','')
    prediction = prediction[prediction.find(')')+1:]
    

    #handles wikt definitions like 'plural of X' or 'past participle of Y'
    pred_last_word = nltk.word_tokenize(prediction)[-1]
    if depth==0 and (porter_stemmer.stem(pred_last_word) in porter_stemmer.stem(kw) or porter_stemmer.stem(kw) in porter_stemmer.stem(pred_last_word)):
        depth += 1
        if driver(title, curtext, pred_last_word, pos, depth) == 'invalid-pos':
            pred_last_pos = pos_association[nltk.pos_tag([pred_last_word])[0][1]][0]
            prediction += '. ' + driver(title, curtext, lemmatizer.lemmatize(pred_last_word), pred_last_pos, depth)
        else:
            prediction += '. ' + driver(title, curtext, lemmatizer.lemmatize(pred_last_word), pos, depth)
            
    if pos == 'NNP' and prediction.lower().find('surname'):
        return 'invalid-term'
            
    return prediction.strip()

