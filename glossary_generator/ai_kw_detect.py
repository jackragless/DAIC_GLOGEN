# ai_kw_detect module uses bert prediction taken sent_tokenized text and returns word tokens
# tagged 'B', 'I', 'O' 
# https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)


import simpletransformers
import pickle
from simpletransformers.ner import NERModel, NERArgs
import torch
import nltk



if torch.cuda.is_available() == False:
    print('!!!GPU not detected; required for AI module. Check graphic drivers!!!')



model = NERModel(
    "bert", '/home/jackragless/projects/data/DAIC_GLOGEN/best_bert_kw_model', labels= ["B", "I", "O"], use_cuda=True
)



# predict function reformats raw output
def predict(sent_toks):
    FINAL = []
    accept = ['B','I']
    # performing prediction using bert
    predictions, raw_values = model.predict(sent_toks)

    # separating predictions in non-keywords ("O" tokens), keywords ("B" tokens)
    # and keyphrases ("B,I..." token sequence) 
    for sent in predictions:
        temp = ''
        temp_arr = []
        for prediction in sent:
            if list(prediction.values())[0] in accept:
                temp += ' ' + list(prediction.keys())[0]
            else:
                if temp!='':
                    temp_arr.append(temp.strip())
                    temp = ''
        FINAL.append(temp_arr)
        
    return FINAL

