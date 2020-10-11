'''
Search the corpus using a query
'''
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
with open("inv_index","rb") as f:
    posting_list = pickle.load(f)
    file_dict = pickle.load(f)
    vocab = pickle.load(f)

# The actual data is also loaded to be display the search results
fileobj = open('tvnews_corpus.tsv','r')
lines = fileobj.readlines()
fileobj.close()


# Using a fixed query for testing. Remove and use free input later
# query = input("Query:")
query = "donald trump accuses china of artificially creating climate change"
query_tokens = lemmatize_sentence(query) # lemmatize tokens to use as in vocabulary
print(query_tokens)

query_vector = []
for tok in query_tokens:
    try:
        query_vector.append(vocab.index(tok))
    except ValueError: # Token doesnt exist in vocab - ignored
        print(tok, "does not exist in the vocabulary")


print(query_vector)
query_length = len(query_vector)
start_time = time.time()

# First we obtain the list of all possible documents we actually need to search
# This is a union of the docs in each query term's posting list
# Not an intersection because we use cosine similarity and not boolean retrieval
possible_docs = set()
for q in query_vector:
    possible_docs = possible_docs.union(posting_list[q].keys())

# Run through each doc and generate the vector corresponding to the query terms
# Compute the cosine similarities of it vs a unit vector of the query
doc_scores = {}
for doc in possible_docs:
    doc_vector = []
    for q in query_vector:
        doc_vector.append(posting_list[q].get(doc,0))
    doc_scores[doc] = cosine_similarity(doc_vector,[1 for i in range(query_length)])

# Results are sorted
sorted_results = sorted(doc_scores.items(), key=operator.itemgetter(1), reverse=True)

end_time = time.time()
search_time = end_time - start_time

ct = 0
for i in sorted_results:
    print("Doc No:", i[0],"Score:",i[1])
    fname, rownum = file_dict[i[0]].split(' ')
    rownum = int(rownum[3:])
    print('\t',"CSV File:",fname," row:",rownum)
    search_res = lines[i[0]]
    search_res = search_res.split('\t')[2]
    print('\t',search_res)
    ct += 1
    if ct == 10:
        break
print()
print("Search Time:",end_time-start_time)
