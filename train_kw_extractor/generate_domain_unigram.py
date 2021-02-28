import pickle
import nltk
import pandas as pd
from tqdm import tqdm
from train_kw_extractor import preprocess_utils_ai


def driver(corpus):
    
    loop_count = 0
    clean_text_obj = []

    for obj in corpus:
        loop_count += 1

        if obj['text']:
            clean_text_obj.append(preprocess_utils_ai.clean_text(obj['text'],False,True,False,True,True))
            
    
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
    


# with open('orig_wiki_corpus.pkl', 'rb') as f:
#     corpus = pickle.load(f)
    
# FINAL_OUTPUT = driver(corpus)
    
# unigram_df.to_csv('domain_specific_unigram.csv')

