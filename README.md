# ---GLOGEN---
### Domain Specific Keyword Identifier & GLOssary GENerator
---------------------------------------------
### Features
###### Module 1: train keyword extraction model:
1. WIKI MINER: Recursively mines all pages from a given Wikipedia Category (domain) (https://en.wikipedia.org/wiki/Wikipedia:Contents/Categories); extracts all hyperlinks to form keyword/keyphrase dataset.
2. RULEBASED KEYWORD ADDER: Adds additional, non-link keywords using TF-IDF (OPTIONAL).
3. CLEANER / CHINKER: Cleans raw Wiki page text and chinks unwanted keywords incl GEO, PERSON, ORG using entity classification / rule-based.
4. BIOGEN: Converts text to Beggining-Inside-Outside (BIO) format for training.
5. BERT TRAINER: Trains bert-base-uncased keyword extraction model using SimpleTransformers.

###### Module 2: glossary generator:
1. GLOSSARY GENERATOR INTERFACE: Takes .txt files from specified directory as input.
2. PREPROCESS UTILITIES: apply various text cleaning techniques incl lemmatisation, stopword removal, grammar removal, tokenisation etc.
3. AI KW DETECT: generates array of predicted keywords/phrases from given text using pretrained model from Module 1.
4. PARSE UTILITIES: generates constituency tree / POS tags of text.
5. WIKTIONARY DEFINITION PARSER: modified code from https://github.com/Suyash458/WiktionaryParser
6. WIKTIONARY DEFINITION PREDICTOR: takes wiktionary object ---> selects candidate definitions with same POS tag as keyword 
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

###### Module 2: glossary generator:
Run `python glossary_generator.py`
Modify supporting files in `glossary_generator` directory.