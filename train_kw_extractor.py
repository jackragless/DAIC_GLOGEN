import pickle
import wikipedia
from train_kw_extractor import *


# user input interface

print('Select a Wikipedia category page from: https://en.wikipedia.org/wiki/Wikipedia:Contents/Categories\n')

while True:
	wiki_cat = input('\nEnter the title of your selected page (eg. Category:Computer Science)\n>>>')

	error_message_1 = '\nINVALID WIKIPEDIA CATEGORY PAGE (SPACING / CASING IS IMPORTANT!) --- TRY AGAIN'
	
	if wiki_cat.find('Category:')==-1:
		print(error_message_1)
	else: 
		try:
			wikipedia.page(wiki_cat, auto_suggest = False)
			break
		except:
			print(error_message_1)


valid_integer_condition = False
while valid_integer_condition == False:
	max_depth = input('\nEnter the depth you want the resursive mining function to enter; the deeper one recursively mines, the more esoteric the pages\n>>>')

	error_message_2 = '\nINVALID INPUT; MUST BE A POSITIVE INTEGER (0<=x<=10) --- TRY AGAIN'

	try:
		int(max_depth)
		if int(max_depth)>=0 and int(max_depth)<=10:
			valid_integer_condition = True
		else:
			print(error_message_2)
	except:
		print(error_message_2)



#pipeline 1
wiki_corpus = P1_wikipedia_corpus_miner.driver(wiki_cat, int(max_depth))
with open('data/orig_wiki_corpus.pkl', 'wb') as f:
    pickle.dump(wiki_corpus, f)
print('\n')


#pipeline 2
wiki_corpus_kw_added = P2_manual_add_keywords.driver(wiki_corpus)
with open('data/wiki_corpus_addedkw.pkl', 'wb') as f:
    pickle.dump(wiki_corpus, f)
print('\n')

#unigram generation
unigram = generate_domain_unigram.driver(wiki_corpus_kw_added)
unigram.to_csv('data/domain_specific_unigram.csv')
print('\n')


#pipeline 3
processed_wiki_pages = P3_wiki_corpus_cleaning_chinking.driver(wiki_corpus_kw_added, unigram)
with open('data/wiki_corpus_chinked.pkl', 'wb') as f:
    pickle.dump(processed_wiki_pages, f)
print('\n')


#pipeline 4
biogen = P4_biogen_format_conversion.driver(processed_wiki_pages)
with open('data/biogen.pkl', 'wb') as f:
    pickle.dump(processed_wiki_pages, f)
print('\n')

#pipeline 5
print('TRAINING BERT MODEL:')
P5_simpletransformers_bert_training_evaluation.driver(biogen) #find models at data/bert-model-files/outputs
print('\nAI TRAINING PIPELINE COMPLETE.')