#!/usr/bin/env python
# coding: utf-8

# In[1]:


import simpletransformers
import logging
import pandas as pd
import pickle
from simpletransformers.ner import NERModel, NERArgs
import os


# In[2]:


model_args = {
'num_train_epochs': 3,
'train_batch_size' : 32,
'eval_batch_size' : 32,
'evaluate_during_training' : False,
'save_model_every_epoch' : True,
'save_eval_checkpoints' : False,
'save_steps' : -1,
'output_dir':'bert-model-files/outputs/',
'cache_dir':'bert-model-files/cache_dir',
'tensorboard_dir':'bert-model-files/runs'
}


# In[3]:


def bert_train(corpus, train_cutoff):
    
    train_data = pd.DataFrame(
    corpus[:train_cutoff], columns=["sentence_id", "words", "labels"]
    )
    
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)
    

    
    model = NERModel(
    "bert", "bert-base-cased", labels= ["B", "I", "O"], use_cuda=True, args = model_args
    )
    
    model.train_model(train_data)


# In[4]:


def bert_eval(corpus, train_cutoff):
    
    eval_data = pd.DataFrame(
    corpus[train_cutoff:], columns=["sentence_id", "words", "labels"]
    )
    
    base_loc = 'bert-model-files/outputs'
    
    model_eval_results = []
    for foldername in os.listdir(base_loc):
        if foldername.find('epoch')!=-1:
            model = NERModel(
            "bert", base_loc+'/'+foldername, use_cuda = True, args = model_args
            )
            print(base_loc+'/'+foldername)
            model_eval_results.append([foldername,model.eval_model(eval_data)[0]])
            pd.DataFrame(model_eval_results).to_csv('bert-model-files/model_evaluation_metrics.csv')


# In[5]:


def driver(corpus):
    
    train_cutoff = int(0.8 * len(corpus))
    
    bert_train(corpus, train_cutoff)
    bert_eval(corpus, train_cutoff)


# In[6]:


# with open('biogen_stopwords_latest_wgrammar.pkl', 'rb') as f:
#     corpus = pickle.load(f)
# driver(corpus)

