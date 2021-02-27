#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
from train_kw_extractor import *


# In[ ]:


print('MINING CORPUS...')
wiki_corpus = P1_wikipedia_corpus_miner.driver('Category:Artificial intelligence')
with open('orig_wiki_corpus.pkl', 'wb') as f:
    pickle.dump(wiki_corpus, f)


# In[ ]:


print('MANUALLY ADDING KEYWORDS...')
wiki_corpus_kw_added = P2_manual_add_keywords.driver(corpus)
with open('wiki_corpus_addedkw.pkl', 'wb') as f:
    pickle.dump(corpus, f)


# In[ ]:


print('GENERATING UNIGRAM...')
unigram = generate_domain_unigram.driver(wiki_corpus_kw_added)
unigram.to_csv('domain_specific_unigram.csv')


# In[ ]:


print('CHINKING CORPUS...')
processed_wiki_pages = P3_wiki_corpus_cleaning_chinking.driver(wiki_corpus_kw_added, unigram)
with open('wiki_corpus_chinked.pkl', 'wb') as f:
    pickle.dump(processed_wiki_pages, f)


# In[ ]:


print('CONVERTING CORPUS TO BIO FORMAT...')
biogen = P4_biogen_format_conversion.driver(processed_wiki_pages)
with open('biogen.pkl', 'wb') as f:
    pickle.dump(processed_wiki_pages, f)


# In[ ]:


print('TRAINING BERT MODEL...')
P5_simpletransformers_bert_training_evaluation.driver(biogen)
print('PIPELINE COMPLETE.')

