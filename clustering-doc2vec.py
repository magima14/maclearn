import gensim
import nltk
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, join
import pandas as pd     
import numpy as np
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

dataset = pd.read_csv("E:/MSc.BA/Semester 2/Smart Systems/uci-news-aggregator/uci-news-aggregator.csv")
dataset.head()


#data cleaning 
tokenizer = RegexpTokenizer(r'\w+')
stopword_set = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

# remove duplicate description columns
senlist = dataset.drop_duplicates('TITLE')
# remove rows with empty descriptions
senlist = senlist[~senlist['TITLE'].isnull()]

senlist['len'] = senlist['TITLE'].map(len)
senlist = senlist[senlist.len > 50]
senlist.reset_index(inplace=True)
senlist.drop('index', inplace=True, axis=1)
#converting pandas series to list 
sen_list = []
sen_list = senlist['TITLE']
sen_list = sen_list.tolist()
#sen_list = pd.Series(sen_list)
#sen_list = list(sen_list[~sen_list.isnull()])
#number_tokens = [re.sub(r'[\d]', ' ', i) for i in sen_list]

def nlp_clean(data):
   new_data = []
   for i in range(len(data)):
      new_str = data[i].lower()
      dlist = tokenizer.tokenize(new_str)
      dlist = list(set(dlist).difference(stopword_set))
      new_data.append(dlist)
   return new_data

data = nlp_clean(sen_list)

#doc_labels 
doc_labels = []
doclabels = dataset.index
doc_labels = doclabels.tolist() 
doc_labels = doc_labels[:len(data)]
doc_labels = [[i] for i in doc_labels]

sentences = [gensim.models.doc2vec.TaggedDocument(words=data[i],tags=doc_labels[i]) for i in range(len(doc_labels))]

model = gensim.models.Doc2Vec(vector_size=300, min_count=5, alpha=0.025, min_alpha=0.025, window=2,workers=5, dm=1)

#len(model.wv.vectors)
#a=list(doc)
#print(a[0])
model.build_vocab(sentences)
#training of model
#for epochs in range(40):
#    if epochs % 20 == 0:
#        print('Now training epoch %s' % epochs)
#    print ('iteration '+str(epochs+1))
model.train(sentences,total_examples=model.corpus_count,epochs=1)
#    model.alpha -= 0.002
#    model.min_alpha = model.alpha

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.cluster import MiniBatchKMeans

num_clusters = 30
kmeans_model = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', n_init=1, 
                         init_size=1000, batch_size=1000, verbose=False, max_iter=1000)
vz = model.docvecs.doctag_syn0
#vz = model.wv.vectors
kmeans = kmeans_model.fit(vz)
kmeans_clusters = kmeans.predict(vz)
kmeans_distances = kmeans.transform(vz)

from sklearn.manifold import TSNE

tsne_model = TSNE(n_components=2, verbose=1, random_state=0)

tsne_kmeans = tsne_model.fit_transform(kmeans_distances)
tsne_kmeans.shape 

import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_notebook
output_notebook()

import numpy as np

colormap = np.array(["#6d8dca", "#69de53", "#723bca", "#c3e14c", "#c84dc9", "#68af4e", "#6e6cd5",
"#e3be38", "#4e2d7c", "#5fdfa8", "#d34690", "#3f6d31", "#d44427", "#7fcdd8", "#cb4053", "#5e9981",
"#803a62", "#9b9e39", "#c88cca", "#e1c37b", "#34223b", "#bdd8a3", "#6e3326", "#cfbdce", "#d07d3c",
"#52697d", "#7d6d33", "#d27c88", "#36422b", "#b68f79"])

plot_kmeans = bp.figure(plot_width=700, plot_height=600, title="KMeans clustering of the news",
    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
    x_axis_type=None, y_axis_type=None, min_border=1)

kmeans_df = pd.DataFrame(tsne_kmeans, columns=['x', 'y'])
kmeans_df['cluster'] = kmeans_clusters
kmeans_df['TITLE'] = senlist['TITLE']
kmeans_df['category'] = senlist['CATEGORY']

plot_kmeans.scatter(x='x', y='y', 
                    color=colormap[kmeans_clusters], 
                    source=kmeans_df)
hover = plot_kmeans.select(dict(type=HoverTool))
hover.tooltips={"description": "@description", "category": "@category", "cluster":"@cluster"}
show(plot_kmeans)
