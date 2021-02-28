# parse_utils module implements POS tagging and constituency parsing


import nltk
nltk.download('punkt',quiet=True)
import benepar
benepar.download('benepar_en3',quiet=True)
parser = benepar.Parser("benepar_en3")


# returns raw constituency tree
def parseSent(sent):
    try:
        const = parser.parse(sent)
        return const
    except:
        return


# returns all nodes from tree as array
def getAllNodes(parent, arr):
    for node in parent:
        if type(node) is nltk.Tree:
            arr.append([node.label(),node.leaves()])
            getAllNodes(node, arr)
    return arr


# returns phrase (NP or VP) nodes from tree as array
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
    

# returns all word nodes from tree as array
def getWordNodes(parent, arr):
    for node in parent:
        if type(node) is nltk.Tree:
            for leaf in node.leaves():
                if node.label() in ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WRB', ',', '.']:
                    arr.append([leaf,node.label()])
            getWordNodes(node, arr)
    return arr

