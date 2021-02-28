from tqdm import tqdm
import pickle
import numpy as np
import nltk
import numpy
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))  




def para_kw_space_remove(curpage, kw_pool):
    curstring = curpage['text']
    for kw in kw_pool:
        if curstring.find(kw)!=-1:
            highlight = curstring[curstring.find(kw):curstring.find(kw)+len(kw)].replace(' ', '|/|')
            curstring = str(curstring[:curstring.find(kw)]) + str(highlight) + str(curstring[curstring.find(kw)+len(kw):])
    return curstring




def biogen(curpage, kw_pool):
    sents = nltk.sent_tokenize(para_kw_space_remove(curpage, kw_pool))
    final_label_arr = []
    for sent in sents:
        temp_label_arr = []
        if len(sent)<=512:
            for phrase in nltk.word_tokenize(sent):
                count = 0
                if phrase.replace('|/|',' ') in kw_pool:
                    phrase_words = phrase.replace('|/|',' ').split()
                    for word in phrase_words:
                        if count == 0:
                            temp_label_arr.append('B')
                        elif count > 0:
                            temp_label_arr.append('I')
                        count+=1
                else:
                    temp_label_arr.append('O')
        final_label_arr.append(temp_label_arr)

    final_sent_arr = []
    for sent in sents:
        final_sent_arr.append(nltk.word_tokenize(sent.replace('|/|',' ')))

    return final_sent_arr, final_label_arr



def same_len(sentences, labels):
    filt_s = []
    filt_l = []
    for i in range(len(sentences)):
        if len(sentences[i]) == len(labels[i]):
            filt_s.append(sentences[i])
            filt_l.append(labels[i])
    return filt_s, filt_l




def driver(corpus):

    kw_pool = []
    for page in corpus:
        kw_pool += page['kw']
    kw_pool = list(set(kw_pool))
    kw_pool = np.sort(kw_pool)

    
    labels = []
    sentences = []
    count = 0
    for page in tqdm(corpus, desc='CONVERTING CORPUS TO BIO FORMAT'):
        count += 1
        #print(int(100*count/len(corpus)),'% <--->',str(count) + '/' + str(len(corpus)), end='\r')
        s, l= biogen(page, kw_pool)
        sentences += s
        labels += l
    sentences, labels = same_len(sentences, labels)
    
    
    FINAL_OUTPUT = []
    sent_num = 0
    for sent in range(len(sentences)):
        sent_num += 1
        for word in range(len(sentences[sent])):
            FINAL_OUTPUT.append([sent_num,sentences[sent][word],labels[sent][word]])
            
            
    return FINAL_OUTPUT



# with open("wiki_corpus_chinked_wgrammar.pkl", 'rb') as f:
#     corpus = pickle.load(f)

# FINAL_OUTPUT = driver(corpus)

# fileObj = open('biogen.pkl', 'wb')
# pickle.dump(FINAL_OUTPUT,fileObj)

