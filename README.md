# GLOGEN
### Domain-specific Keyword Extraction & GLOssary GENeration
---------------------------------------------
### Features
###### Module 1: train keyword extraction model:
1. INTERFACE (train_kw_extractor.py)
2. WIKI MINER (P1_wikipedia_corpus_miner.py): Recursively mines all pages from a given Wikipedia Category (domain) (https://en.wikipedia.org/wiki/Wikipedia:Contents/Categories); extracts all hyperlinks to form keyword/keyphrase dataset.
3. RULEBASED KEYWORD ADDER (P2_manual_add_keywords.py): Adds additional, non-link keywords using TF-IDF (OPTIONAL).
4. CLEANER / CHINKER (P3_wiki_corpus_cleaning_chinking.py): Cleans raw Wiki page text and chinks unwanted keywords incl GEO, PERSON, ORG using entity classification / rule-based.
5. BIOGEN (P4_biogen_format_conversion.py): Converts text to Beginning-Inside-Outside (BIO) format for training.
6. BERT TRAINER (P5_simpletransformers_bert_training_evaluation.py): Trains bert-base-uncased keyword extraction model using SimpleTransformers. 

###### Module 2: glossary generator:
1. GLOSSARY GENERATOR INTERFACE (glossary_generator.py): Takes .txt files from specified directory as input.
2. PREPROCESS UTILITIES (preprocess_utils.py): apply various text cleaning techniques incl lemmatisation, stopword removal, grammar removal, tokenisation etc.
3. AI KEYWORD DETECT (ai_kw_detect.py): generates array of predicted keywords/phrases from given text using pretrained model from Module 1.
4. PARSING UTILITIES (parse_utils.py): generates constituency tree / POS tags of text.
5. WIKTIONARY DEFINITION PARSER (wikt_def_parse.py): modified code from https://github.com/Suyash458/WiktionaryParser
6. WIKTIONARY DEFINITION PREDICTOR (wikt_def_predict.py): takes wiktionary object ---> selects candidate definitions with same POS tag as keyword 
---> predicts best definition via semantic similarity between definition sentence / definition domain / definition example and the source text.


## Installation
Create a conda virtual environment and install dependencies specified in requirements.txt.

A pretrained keyword extraction model trained on Category:Computer science corpus (~50k pages) is available at:
https://drive.google.com/file/d/17hj6DERAMS5nGeihyBTYSbc2INKqvaHG/view?usp=sharing
To use this pretrained model simply extract it to the main folder where `glossary_generator.py` is located.

## Usage
###### Module 1: train keyword extraction model:
Run `python train_kw_extractor.py`
Modify model training settings (eg. epoch freq) in `train_kw_extractor/P5_simpletransformers_bert_training_evaluation.py`
Models from training located in `data/bert-model-files/outputs`
Evaluation metrics in `data`

###### Module 2: glossary generator:
Run `python glossary_generator.py`
Modify supporting files in `glossary_generator` directory.

See examples of GLOGEN glossary generation in `GLOGEN_output_examples`

##License
Apache-2.0
