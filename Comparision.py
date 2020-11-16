# Install ElasticSearch on Local Machine and configure to port
# Hosting ElasticSearch sever
import os
from subprocess import Popen, PIPE, STDOUT

# Import searchAPI
from search import searchAPI

# Start and wait for server
server = Popen(['elasticsearch-7.8.1/bin/elasticsearch'], stdout=PIPE, stderr=STDOUT, preexec_fn=lambda: os.setuid(1))
from elasticsearch import Elasticsearch
es = Elasticsearch(hosts=["http://localhost:9200"], timeout=60, retry_on_timeout=True)
es.indices.create(index="environment", ignore=400)

lines=[]
with open('../input/combined/tvnews_corpus.tsv', 'r', errors='replace') as f:
    lines = f.readlines()
	
# Building Elastic Search Index
from tqdm import tqdm
import csv
counter = 1
finalMapping = dict()

#Replace with path for dataset
for dirname,_,filenames in os.walk('../input/environmental-news-nlp-dataset/TelevisionNews'):
    for filename in tqdm(filenames, "progress"):
        path = os.path.join(dirname, filename)
        with open(path, 'r', errors='replace') as file: 
            reader = csv.reader(file)
            dictForRow = dict()
            rowNum = 0
            for row in reader:
                if(rowNum!=0):
                    dictForRow['snippet'] = row[6]
                    strBuilder = str(filename) + "#" + str(rowNum)
                    finalMapping[counter] = strBuilder
                    es.index(index="environment", doc_type="env", id=strBuilder, body=dictForRow)
                    counter += 1
                rowNum+=1

#Testing a Search
res = es.search(index="environment", body={"from":0, "size":10000, "min_score":0, "query":{"match":{"snippet":"Global warming is a hoax"}}})

#Search API
import pickle
import operator
import time
from pywsd.utils import lemmatize_sentence
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(v1,v2):
    return dot(v1, v2)/(norm(v1)*norm(v2))

# The inverted index file contents are loaded into memory
# All queries will be analyzed with this, so it is fine to preload
with open("../input/indexing/inv_index_v2","rb") as f:
    posting_list = pickle.load(f)
    file_dict = pickle.load(f)
    vocab = pickle.load(f)

# The actual data is also loaded to be display the search results
fileobj = open('../input/combined/tvnews_corpus.tsv','r', errors='ignore')
lines = fileobj.readlines()
fileobj.close()

#Helper functions forRelevant Documents
def retrivetopkelastic(hits):
    topk=[]
    for i in range(0,len(hits)):
        topk.append(hits[i]['_id'])
    return topk

def retrievaAllElastic(line):
    total=[]
    data = es.search(index="environment",scroll='2m', body={"from":0, "size": 10000, "min_score":0, "query":{"match":{"snippet":line}}})
    sid = data['_scroll_id']
    scroll_size = len(data['hits']['hits'])
    while scroll_size > 0:
        scroll_size = len(data['hits']['hits'])
        total+=[id['_id'] for id in data['hits']['hits']]
        data = es.scroll(scroll_id=sid, scroll='2m')
        sid = data['_scroll_id']
    return total
	
#Run OurModel and ElasticSearch for TestCases
import time

testcasefile="../input/samplequeries/sample_queries.txt"
inputfile= open(testcasefile,"r")
timelist1=[]
timelist2=[]

retrieved=[]
relevant=[]

for line in tqdm((inputfile.read()).split('\n'), "progress"):
    
    #SearchAPI
    results = searchAPI(line)
    timelist1.append(results['Time'])
    retrieved.append(results['Documents'])
    
    #Elastic Search
    tic=time.time()
    relevant.append(retrievaAllElastic(line))
    toc=time.time()
    timelist2.append(toc-tic)
	
#Storing Results
with open("resultsfinalv1","wb") as f:
    pickle.dump(retrieved,f)
    pickle.dump(relevant, f)
    pickle.dump(timelist1,f)
    pickle.dump(timelist2,f)
	
#Also Available At https://drive.google.com/file/d/19wQG2DA6o_ArZRlsK3AFeZEwGsqkn4xX/view?usp=sharing	
#Retrieving Results
import pickle
with open("../input/resultsfinal/resultsfinalv1","rb") as f:
    retrieved = pickle.load(f)
    relevant = pickle.load(f)
    timelist1 = pickle.load(f)
    timelist2 = pickle.load(f)

#Calculating Percentage Change in Time between OurModel and ElasticSearch
perchange = []
for a, b in zip(timelist1, timelist2):
    perchange.append(100 * (a - b) / b)

#Plotting Percentage Time Change and Average Time Taken
import numpy as np
import random
from matplotlib import pyplot as plt

y = perchange

N = len(y)
x = np.arange(50)
width = 1 / 1.1
plt.figure(figsize=(8, 6), dpi=100)

fig, ax = plt.subplots(1, 1)
ax.grid(zorder=0)
positives = ['r'if(i>0) else'b' for i in y] 
bars = ax.barh(x, y, width, color=positives)


plt.title('Time Comparision with Elastic Search')
plt.ylabel('Query')
plt.xlabel('Percentage Change wrt Elastic Search')
plt.show()

print("Average Time Taken")
print("Our Model", sum(timelist1)/len(timelist1),"s")
print("Elastic Search", sum(timelist2)/len(timelist2),"s")

#Calculating Precision and Recall for Values of k i.e. top k results
from tqdm import tqdm
precisions = []
recalls = []

    
for k in tqdm(range(1,93000,50), 'progress'):
    precisionnum = 0
    precisionden = 0
    intermprecision = []
    intermrecall = []
    for i in range(50):
        precisionnum=len(set(retrieved[i][:k]).intersection(relevant[i][:int(0.2*len(relevant[i]))]))
        precisionden=len(set(retrieved[i][:k]))
        intermprecision.append(precisionnum/precisionden)
    precisions.append(sum(intermprecision)/50)
    recallnum = 0
    recallden = 0
    intermrecalls = []
    for i in range(50):
        recallnum=len(set(retrieved[i][:k]).intersection(relevant[i][:int(0.2*len(relevant[i]))]))
        recallden=len(set(relevant[i][:int(0.2*len(relevant[i]))]))
        intermrecalls.append(recallnum/recallden)
    recalls.append(sum(intermrecalls)/50)
	
# Top k Intersection
from tqdm import tqdm
topkintersection = []
for k in tqdm(list(range(0,105,5))[1:], 'progress'):
    intermk=[]
    for i in range(50):
        knum=len(set(retrieved[i][:k]).intersection(relevant[i][:k]))
        intermk.append(knum/k)
    topkintersection.append(sum(intermk)/50)
#Average k intersection value
print("Average k Interesection:", sum(topkintersection)/len(topkintersection))

#Plotting Top k Intersection
import matplotlib.pyplot as plt
import numpy as np

width=2
plt.figure(figsize=(5, 3), dpi=100)
plt.bar(list(range(0,105,5))[1:], topkintersection, width, label='Our Model')
plt.xlabel("k")
plt.ylabel('k intersection')
plt.title('Top k Intersection')
plt.show()

#Interpolating
indices = [0.0, 0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
recallprec = [(recalls[i], precisions[i]) for i in range(len(recalls))]

#Calculating Interpolated Precison for Recall Levels
recallprec.sort()
values=[]
for ind in tqdm(indices, 'Progress'):
    maxim=-1000
    for i in recallprec:
        if(i[0]>ind and i[1]>maxim):
            maxim=i[1]
    if(maxim!=-1000):
        values.append(maxim)
    else:
        values.append(values[len(values)-1])
		
#Plotting PR Curve and Interpolated PR Curve
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

plt.figure(figsize=(5, 3), dpi=100)
plt.plot(recalls, precisions, label = 'Original Curve')
plt.plot(indices,values, label = 'Interpolated')

plt.xlabel("Recall")
plt.title('Precision Recall Curve')
plt.ylabel("Precision")
plt.legend()
plt.show()

#MAP Calculation
print("Recall\tInterp. Precision")
for i in range(len(indices)):
    print(indices[i],'\t', values[i])
print("MEAN AVERAGE PRECISION:", sum(values)/len(values))
