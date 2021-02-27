#!/usr/bin/env python
# coding: utf-8

# In[3]:


import nltk
nltk.download('punkt')
import benepar
benepar.download('benepar_en3')
parser = benepar.Parser("benepar_en3")
norm_pos_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WRB', ',', '.']


# In[17]:


def parseSent(sent):
    try:
        const = parser.parse(sent)
        return const
    except:
        return


# In[18]:


def getAllNodes(parent, arr):
    for node in parent:
        if type(node) is nltk.Tree:
            arr.append([node.label(),node.leaves()])
            getAllNodes(node, arr)
    return arr


# In[19]:


test = 'Greenfoot is an integrated development environment using Java or Stride designed primarily for educational purposes at the high school and undergraduate level.'


# In[36]:


def getPhraseNodes(parent, arr):
    for node in parent:
        if type(node) is nltk.Tree:
            if node.label() == 'NP' and len(node.leaves()) in range(2,6):
                temp_cat_str = ''
                condition = True
                for child in node:
                    if child.label() == 'NP' or child.label() == 'VP' or child.label() == 'PP':
                        condition = False
                if condition == True:
                    for leaf in node.leaves():
                        temp_cat_str += ' ' + leaf
                    arr.append([temp_cat_str.strip(),'noun'])
            if node.label() == 'VP' and len(node.leaves()) in range(2,6):
                temp_cat_str = ''
                condition = True
                for child in node:
                    if child.label() == 'NP' or child.label() == 'VP' or child.label() == 'PP':
                        condition = False
                if condition == True:
                    for leaf in node.leaves():
                        temp_cat_str += ' ' + leaf
                    arr.append([temp_cat_str.strip(),'verb'])
            
            getPhraseNodes(node, arr)
            
    return arr


# In[ ]:


def getWordNodes(parent, arr):
    for node in parent:
        if type(node) is nltk.Tree:
            for leaf in node.leaves():
                if node.label() in norm_pos_tags:
                    arr.append([leaf,node.label()])
            getWordNodes(node, arr)
    return arr

