import nltk
import pyfiglet
import os
import wikipedia
from tqdm import tqdm
from glossary_generator import *

#converts NLTK POS tags to Wiktionary POS tags
pos_association = {
    'CC': ['conjunction'],
    'CD': ['numeral'],
    'DT': ['determiner'],
    'EX': [],
    'FW': [],
    'IN': ['preposition', 'conjunction'],
    'JJ': ['adjective'],
    'JJR': ['adjective'],
    'JJS': ['adjective'],
    'LS': [],
    'MD': ['verb'],
    'NN': ['noun', 'proper noun'],
    'NNS': ['noun'],
    'NNP': ['noun', 'proper noun'],
    'NNPS': ['noun', 'proper noun'], 
    'PDT': ['determiner'],
    'POS': [],
    'PRP': ['pronoun'],
    'PRP$': ['pronoun'],
    'RB': ['adverb'],
    'RBR': ['adverb'],
    'RBS': ['adverb'],
    'RP': ['preposition'],
    'TO': [], 
    'UH': ['interjection'],
    'VB': ['verb'],
    'VBG': ['verb'],
    'VBD': ['verb'],
    'VBN': ['verb'],
    'VBP': ['verb'],
    'VBZ': ['verb'],
    'WDT': ['determiner'],
    'WP': ['pronoun'],
    'WRB': ['adverb'],
    # to deal with keyphrases
    'noun': ['noun', 'proper noun'],
    'verb': ['verb']
}



# generates constituency tree, predicts keyword/phrases from text, then ensures kephrases are valid compared to constituency parsing
def generate_keywords(processed_text):
    candidate_phrases = []
    pos = []
    tok_sents = []
    for sent in nltk.sent_tokenize(processed_text):
        temp_tree = parse_utils.parseSent(sent)
        if temp_tree:
            tok_sents.append(sent[:-1])
            candidate_phrases += parse_utils.getPhraseNodes(temp_tree, [])
            pos.append(parse_utils.getWordNodes(temp_tree, []))

    pred_kw = ai_kw_detect.predict(tok_sents)

    FINAL_KW = []
    kw_only = []
    for i in range(len(pred_kw)):
        if pred_kw[i]:
            for j in range(len(pred_kw[i])):
                if pred_kw[i][j].find('.') == -1:
                    if len(pred_kw[i][j].split()) == 1:
                        for k in range(len(pos[i])):
                            if pred_kw[i][j] == pos[i][k][0] and pred_kw[i][j] not in kw_only:
                                try:
                                    FINAL_KW.append(
                                        [pos[i][k][0], pos_association[pos[i][k][1]]])
                                    kw_only.append(pos[i][k][0])
                                except:
                                    pass

                                break
                    else:
                        for m in range(len(candidate_phrases)):

                            if pred_kw[i][j] == candidate_phrases[m][0] and pred_kw[i][j] not in kw_only:
                                try:
                                    FINAL_KW.append(
                                        [candidate_phrases[m][0], pos_association[candidate_phrases[m][1]]]) #convert NLTK POS ---> Wikt POS
                                    kw_only.append(candidate_phrases[m][0])
                                except:
                                    pass
                                    
                                break
    return FINAL_KW



# extracts definitions from Wiktionary using wikt_def_predict
def generate_definitions(title, clean_text, keyword_arr):
    final = []
    count = 0
    for kw in tqdm(keyword_arr, desc="WIKTIONARY DEFINITION GENERATION"):
        count += 1
        if kw[0].lower() not in [kw[0].lower() for kw in final]:
            for pos in kw[1]:

                temp_def = wikt_def_predict.driver(
                    title.strip(), clean_text.strip(), kw[0].strip(), pos.strip(), 0)

                if temp_def != 'invalid-pos' and temp_def != 'invalid-term':
                    final.append([kw[0], temp_def.replace(
                        'invalid-pos', '').replace('invalid-term', '')])
                    break
    return final


#appends references '...keyword|1|...' to end of keywords in-text 
def add_def_refs(orig_text, keywords_only):
    indexes = []
    index_sum = 0
    for kw in keywords_only:
        temp_index = orig_text[index_sum:].lower().find(
            kw.lower() + ' ') + len(kw)
        index_sum += temp_index
        indexes.append(index_sum)

    ref_text = orig_text
    index_adjust = 0
    for i in range(len(indexes)):
        ref_text = ref_text[:indexes[i]+index_adjust] + \
            '|{}|'.format(i+1) + ref_text[indexes[i]+index_adjust:]
        index_adjust += len(str(i+1)) + 2
    return ref_text


# prepends glossary to original .txt input to form one string
def gen_final_doc(corpus_obj):
    final_doc = '===GLOGEN GLOSSARY===\n\n'

    def_count = 0
    for definition in corpus_obj['glossary']:
        def_count += 1
        temp_sent = definition[1]
        final_doc += '|{}| '.format(def_count) + definition[0] + ' : ' + (
            temp_sent + '.').replace('..', '') + '\n'

    final_doc += '\n===DOCUMENT BODY===\n\n' + corpus_obj['text_w_ref']
    return final_doc



# user interface for providing .txt files
print(pyfiglet.figlet_format("DAIC GLOGEN"))
print('DESCRIPTION: GLOGEN automatically generates glossaries and prepends them to given .txt files. \nENSURE: <filename>.txt == original text title.')
answer = ''
while True:
    answer = input('\nType "yes" / "no" to starting GLOGEN:\n>>>')
    if answer.lower().startswith("y"):

        while True:
            txt_address = input(
                'Type address where .txt files are stored. If same address as main.py press ENTER.\n>>>')
            if os.path.exists(txt_address) or len(txt_address) == 0:
            	break
            else:
                print('INVALID ADDRESS --- TRY AGAIN')

        break
    elif answer.lower().startswith("n"):
        exit()
    else:
        print('INVALID INPUT --- TRY AGAIN.')
        continue

if txt_address == '':
    txt_address = os.getcwd()
if txt_address[-1] == '/':
    txt_address = txt_address[:-1]


print('\n')


if os.path.exists(txt_address + '/GLOGEN') == False:
    os.mkdir(txt_address + '/GLOGEN')



# iterates through all text files in txt_address and applies above functions, then outputs result to GLOGEN folder
for i, filename in enumerate(os.listdir(txt_address)):

    if filename.endswith('.txt'):

        print('DOC', str(i+1)+'/'+str(len(os.listdir(txt_address))), '---', filename)
        orig_text = open(txt_address+'/'+filename).read()
        processed = preprocess_utils.clean_text(
            orig_text, False, False, True, False, False)

        print('AI KEYWORD PREDICTION:')
        keywords_and_pos = chink_utils.keywords(generate_keywords(processed))

        glossary = generate_definitions(
            filename[:-4], processed, keywords_and_pos)

        keywords_only = [obj[0] for obj in glossary]
        text_w_ref = add_def_refs(orig_text, keywords_only)

        temp_obj = {
            'title': filename[:-4],
            'text_w_ref': add_def_refs(orig_text, keywords_only),
            'glossary': glossary
        }
        text_file = open(
            txt_address + "/GLOGEN/{}.txt".format(temp_obj['title']), "w")
        text_file.write(gen_final_doc(temp_obj))
        text_file.close()
        print('GLOGEN DOC ADDED.\n')



# pages = ['Augmented reality-assisted surgery','Universal Scene Description','Junaio','USens','ARCore','Adjusted mutual information','Algorithmic information theory','Anti-information','Ascendency','Asymptotic equipartition property']
# for page in pages:
#     text_file = open("text_files/{}.txt".format(page), "w")
#     text_file.write(wikipedia.page(page,auto_suggest=False).content)
#     text_file.close()

