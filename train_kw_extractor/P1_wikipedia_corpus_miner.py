#!/usr/bin/env python
# coding: utf-8

# In[1]:


import wikipedia
import pandas as pd
import wikipediaapi
wiki = wikipediaapi.Wikipedia('en')
import pickle
from bs4 import BeautifulSoup as bs


# In[2]:


def mine_page_titles(title, depth):
    final = []
    arr = wiki.page(title).categorymembers
    arr = list(arr.keys())
    if depth > 3: #recommended == 3
        return
    for i in arr:
        if 'Category:' in i:
            result = mine_page_titles(i, depth + 1)
            if result:
                final += result
        else:
            final.append(i)
    return final


# In[3]:


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


# In[4]:


def parse_text(title, curpage):
    try:
        page = curpage.content
        page = page.replace('\n','').replace('  ',' ').replace('==',' ')
        return page
    except:
        return


# In[14]:


def driver(wiki_catgory_page_title):

    wiki_objs = mine_page_titles(wiki_catgory_page_title,0)
    
    for page in wiki_objs:
        if page.find(':')!=-1 or page.find('List')!=-1:
            wiki_objs.remove(page)
    wiki_objs = list(set(wiki_objs))
    
    
    FINAL_OUTPUT = []
    count = 0
    for title in wiki_objs[0:5]: ###to be modified!!!
        curpage = wikipedia.page(title)
        FINAL_OUTPUT.append({
            'title' : title,
            'text' : parse_text(title, curpage),
            'kw' : parse_kw(title, curpage)
        })

        count += 1
        print(int(count/len(wiki_objs)*100), '% <--->', str(count)+'/'+str(len(wiki_objs)),  end='\r')
        
    return FINAL_OUTPUT


# In[15]:


# FINAL_OUTPUT = driver('Category:Artificial intelligence')
# with open('orig_wiki_corpus.pkl', 'wb') as f:
#     pickle.dump(FINAL_OUTPUT, f)

