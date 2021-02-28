# pipeline 1 --- wikipedia_corpus_miner recursively mines all pages titles within a given category to a specified depth
# data mined for each wiki page title (incl title, text, keywords(links)) and rolled into array of wiki objects

import wikipedia
import pandas as pd
import wikipediaapi
wiki = wikipediaapi.Wikipedia('en')
import pickle
from bs4 import BeautifulSoup as bs
from tqdm import tqdm


# recursively mine pages within category; categories listed within categories 
# create exponential growth in pages for each depth level
def mine_page_titles(title, max_depth, depth):
    final = []
    arr = wiki.page(title).categorymembers
    arr = list(arr.keys())
    if depth > max_depth: #recommended == 3
        return
    for i in arr:
        if 'Category:' in i:
            result = mine_page_titles(i, max_depth, depth + 1)
            if result:
                final += result
        else:
            final.append(i)
    return final


# extracts <a href>'s from page HTML containing "/wiki/" 
# (ie. hyperlinks to other articles; we regard these as keywords)
def parse_kw(title, curpage):
    try:
        curpage = curpage.html()
        curpage = curpage[curpage.find('<div class="mw-parser-output"'):]
        curpage = curpage[curpage.find('<p>'):curpage.find('id="See_also"')]

        terms = []

        curpage = bs(curpage,'html.parser')
    
        for i in curpage.find_all('a'):
            if i['href'].find('/wiki/')!=-1 and i.text!='':
                terms.append(i.text.strip())

        return terms
    except:
        return



def driver(wiki_category_page_title, max_depth):

    print('FETCHING WIKI PAGES...')
    wiki_objs = mine_page_titles(wiki_category_page_title,max_depth,0)
    
    for page in wiki_objs:
        if page.find(':')!=-1 or page.find('List')!=-1:
            wiki_objs.remove(page)
    wiki_objs = list(set(wiki_objs))


    # this prompt gives user chance to exit if corpus to be mined too large/small
    print('Wikipedia mining at depth == {} has returned {} unique pages.'.format(max_depth,len(wiki_objs)))
    while True:
        continue_ans = input('Do you wish to proceed mining the data from these pages (yes / no)?\n>>>')
        if continue_ans.lower().startswith("y"):
            break

        elif answer.lower().startswith("n"):
            print('EXITING')
            exit()
        else:
            print('INVALID INPUT --- TRY AGAIN.')
            continue
    
    
    
    FINAL_OUTPUT = []
    for title in tqdm(wiki_objs, desc = 'MINING PAGE DATA'):
        try:
            curpage = wikipedia.page(title)
        except:
            continue
        FINAL_OUTPUT.append({
            'title' : title,
            'text' : curpage.content,
            'kw' : parse_kw(title, curpage)
        })
       
    return FINAL_OUTPUT