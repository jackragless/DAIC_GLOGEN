#converts text into [sentnum, word, label] objects for simpletransformers training
#eg The white cat sat on the matt. kw = 'white cat', 'matt'
# [[1,'The','O'],
#  [1,'white','B'],
#  [1,'cat','I'],
#  [1,'sat','O'],
#  [1,'on','O'],
#  [1,'the','O'],
#  [1,'matt','B'],
#  [1,'.','O']]


from tqdm import tqdm
import pickle
import numpy as np
import nltk
import numpy
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))  


# removes spaces separating words in a keyphrase
# allows for easy BIO parsing
def para_kw_space_remove(curpage, kw_pool):
    curstring = curpage['text']
    for kw in kw_pool:
        if curstring.find(kw)!=-1:
            highlight = curstring[curstring.find(kw):curstring.find(kw)+len(kw)].replace(' ', '|/|')
            curstring = str(curstring[:curstring.find(kw)]) + str(highlight) + str(curstring[curstring.find(kw)+len(kw):])
    return curstring


# returns array of tokenized sents, array of corresponding BIO labels
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


# sanity check to ensure tok-sentence and label array are same length
def same_len(sentences, labels):
    filt_s = []
    filt_l = []
    for i in range(len(sentences)):
        if len(sentences[i]) == len(labels[i]):
            filt_s.append(sentences[i])
            filt_l.append(labels[i])
    return filt_s, filt_l


def driver(corpus):

	# this mechanism pools all keywords from all wiki texts together
	# ensures AI not confused by input where a keyword is labelled 'B'/'I' in one page and 'O' in another (because author failed to hyperlink)
    kw_pool = []
    for page in corpus:
        kw_pool += page['kw']
    kw_pool = list(set(kw_pool))
    kw_pool = np.sort(kw_pool)

    
    labels = []
    sentences = []
    for page in tqdm(corpus, desc='CONVERTING CORPUS TO BIO FORMAT'):
        s, l= biogen(page, kw_pool)
        sentences += s
        labels += l
    sentences, labels = same_len(sentences, labels)
    
    
    FINAL_OUTPUT = []
    sent_num = 0
    for sent in range(len(sentences)):
        sent_num += 1
        for word in range(len(sentences[sent])):
            FINAL_OUTPUT.append([sent_num,sentences[sent][word],labels[sent][word]]) #combines separate sentence and label arrays into one object
            
            
    return FINAL_OUTPUT