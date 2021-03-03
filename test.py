import pickle
import wikipedia


infile = open('/home/jackragless/projects/data/DAIC_GLOGEN/wiki_orig_mined_dataframe.pkl','rb')
corpus = pickle.load(infile)[-30:]

titles = []
for i in corpus:
	if i['title']:
		titles.append(i['title'])




# pages = ['Augmented reality-assisted surgery','Universal Scene Description','Junaio','USens','ARCore','Adjusted mutual information','Algorithmic information theory','Anti-information','Ascendency','Asymptotic equipartition property']
for title in titles:
    text_file = open("text_files/{}.txt".format(title), "w")
    page = wikipedia.page(title,auto_suggest=False).content
    if page:
    	text_file.write(wikipedia.page(title,auto_suggest=False).content)
    	text_file.close()
