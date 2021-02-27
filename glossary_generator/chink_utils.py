# kw_chink_utils module applies manual conditions to chink undesired outputs from BERT predict


from names_dataset import NameDataset
name_dataset = NameDataset()
import pandas as pd
unigram = pd.read_csv('./data/unigram_freq.csv')
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()


# iterates through keyword array and applies functions below
def keywords(kw_arr):
    kw_arr_final = kw_arr
    for kw in kw_arr_final:
        if common_word_detect(kw[0]) == True:
            kw_arr_final.remove(kw)
    return kw_arr_final



# returns True if keyword is a person's name
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



# returns True if keyword found among high frequency unigram terms
common_unigram = list(unigram[:10000]['word']) #arbitrary cutoff to select most common terms

def common_word_detect(candidate_string):
    if candidate_string != candidate_string.upper() and lemmatizer.lemmatize(candidate_string.lower()) in common_unigram:
        return True
    else:
        return False

