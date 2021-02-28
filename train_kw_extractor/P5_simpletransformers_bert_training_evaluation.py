# pipeline 5: simpletransformers_bert_training_evaluation trains and evaluates keyword extraction model on bert-base-uncased

import simpletransformers
import logging
import pandas as pd
import pickle
from simpletransformers.ner import NERModel, NERArgs
import os



model_args = {
'num_train_epochs': 3,
'train_batch_size' : 32,
'eval_batch_size' : 32,
'evaluate_during_training' : False,
'save_model_every_epoch' : True,
'save_eval_checkpoints' : False,
'save_steps' : -1,
'output_dir':'data/bert-model-files/outputs/',
'cache_dir':'data/bert-model-files/cache_dir',
'tensorboard_dir':'data/bert-model-files/runs'
}




def bert_train(corpus, train_cutoff):
    
    train_data = pd.DataFrame(
    corpus[:train_cutoff], columns=["sentence_id", "words", "labels"]
    )
    
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)
    
    
    model = NERModel(
    "bert", "bert-base-cased", labels= ["B", "I", "O"], use_cuda=True, args = model_args #change model here if desired
    )
    
    model.train_model(train_data)




def bert_eval(corpus, train_cutoff):
    
    eval_data = pd.DataFrame(
    corpus[train_cutoff:], columns=["sentence_id", "words", "labels"]
    )
    
    base_loc = 'data/bert-model-files/outputs'
    
    model_eval_results = []
    for foldername in os.listdir(base_loc):
        if foldername.find('epoch')!=-1:
            model = NERModel(
            "bert", base_loc+'/'+foldername, use_cuda = True, args = model_args
            )
            print(base_loc+'/'+foldername)
            model_eval_results.append([foldername,model.eval_model(eval_data)[0]])
            pd.DataFrame(model_eval_results).to_csv('data/bert-model-files/model_evaluation_metrics.csv')




def driver(corpus):
    
    train_cutoff = int(0.8 * len(corpus)) #arbitarily set to 80/20 training/evaluation data split
    
    bert_train(corpus, train_cutoff)
    bert_eval(corpus, train_cutoff)

