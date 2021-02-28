# returns unigram for given wiki corpus in descending or of occurence frequency

import pickle
import nltk
import pandas as pd
from tqdm import tqdm
from train_kw_extractor import ai_preprocess_utils


def driver(corpus):
    
    clean_text_obj = []

    for obj in corpus:
        if obj['text']:
            clean_text_obj.append(ai_preprocess_utils.clean_text(obj['text'],False,True,False,True,True))    
    all_words = []
    count = 0
    for obj in tqdm(clean_text_obj, desc='GENERATING UNIGRAM'):
        count += 1
        for sent in nltk.sent_tokenize(obj):

            sent = sent.replace('.','')
            all_words += nltk.word_tokenize(sent)         
            
    unigram = pd.Series(all_words).value_counts()
    unigram_df = unigram.reset_index()
    unigram_df.columns=['word', 'freq']
    
    return unigram_df