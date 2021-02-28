import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import re
nltk.download('wordnet', quiet=True)
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()
from nltk.tokenize.treebank import TreebankWordDetokenizer



def remove_grammar(text_to_clean):
    final = ''
    for i in text_to_clean:
        if i.isalpha() or i.isdigit() or i=='-' or i==' ' or i=='.':
            final += i
    return final



def remove_bracket_content(text_to_clean):
    return re.sub("[\[\(].*?[\]\)]", "", text_to_clean)




test = "The initiative is coordinated by the International Food Policy Research Institute and the University of Minnesota and is supported by a grant to IFPRI by the Bill & Melinda Gates Foundation.Phase I of HarvestChoice ran from October 2006 to June 2010, while Phase II began in December 2010 for a period of 4 years and a total budget of some $8.2M. Purpose of the Initiative Farming entails a great deal of risk and uncertainties. Weather varies, price fluctuates, soil degrades, pest damages, and, even climate changes. Farmers everywhere must cope with these uncertainties. Throughout the history of agriculture, many options, such as fertilizer application, irrigation, improved varieties, and farming machinery have been developed to help manage the risks, increase yields, increase efficiency, and, increasingly, promote sustainability of the overall system.With these techniques and tools in mind, each farmer must assess their local context and analyze the costs and benefits of adopting them, such as the additional labor and/or investment required. "



def fix_nospace_sents(text_to_clean):
    final = text_to_clean
    for i in range(len(final)-2):
        if (final[i].isalpha() and final[i].islower()) and final[i+1] == '.' and (final[i+2].isalpha() and final[i+2].isupper()):
            final = final[:i+2] + ' ' + final[i+2:]
    return final




def print_settings():
    print("""
        1) remove_bracket_content_bool
        2) remove_grammar_bool
        3) remove_stopword_bool
        4) lemmatize_bool
        5) lowercase_bool
        """
    )




def clean_text(text_to_clean, remove_bracket_content_bool, remove_grammar_bool, remove_stopword_bool, lemmatize_bool, lowercase_bool):

    text_to_clean = text_to_clean.replace('\n','').replace('"',"'").replace('=','')
    
    if remove_bracket_content_bool == True:
        text_to_clean = remove_bracket_content(text_to_clean)
    
    if remove_grammar_bool == True:
        text_to_clean = remove_grammar(text_to_clean)
        
    clean_sent_arr = []
        
    for sent in nltk.sent_tokenize(text_to_clean):
        
        temp_sent = []
        
        if lowercase_bool == True:
            sent = sent.lower()
        
        if remove_stopword_bool == True and lemmatize_bool == True:
            for word in nltk.word_tokenize(sent):   
                if word.lower() not in stop_words:
                    temp_sent.append(lemmatizer.lemmatize(word))
                    
        if remove_stopword_bool == False and lemmatize_bool == False:
            for word in nltk.word_tokenize(sent):   
                    temp_sent.append(word)

        elif remove_stopword_bool == False and lemmatize_bool == True:
            for word in nltk.word_tokenize(sent):   
                    temp_sent.append(lemmatizer.lemmatize(word))
                    
        elif remove_stopword_bool == True and lemmatize_bool == False:
            for word in nltk.word_tokenize(sent):   
                if word.lower() not in stop_words:
                    temp_sent.append(word) 
                    
        clean_sent_arr.append(TreebankWordDetokenizer().detokenize(temp_sent))
        
    final_clean_text = ' '.join(clean_sent_arr)
    final_clean_text = re.sub(' +', ' ', final_clean_text).strip()
    final_clean_text = fix_nospace_sents(final_clean_text)

    return final_clean_text




def pos_tag(text): 
    for_tagging = []
    for sent in nltk.sent_tokenize(text):
        if sent[-1] == '.':
            for_tagging.append(nltk.word_tokenize(sent[:-1]))
        else:
            for_tagging.append(nltk.word_tokenize(sent))

    tagged = []
    sent_num = 0
    for sent in for_tagging:
        sent_num += 1
        for word in nltk.pos_tag(sent):
            if word[0].lower() not in stop_words:
                word = list(word)
                word.insert(0,sent_num)
                tagged.append(word)
    return tagged

