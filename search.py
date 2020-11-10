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
with open("inv_index_v2","rb") as f:
    posting_list = pickle.load(f)
    file_dict = pickle.load(f)
    vocab = pickle.load(f)

# The actual data is also loaded to be display the search results
fileobj = open('tvnews_corpus.tsv','r')
lines = fileobj.readlines()
fileobj.close()

def searchAPI(searchterm):
    query = searchterm
    query_tokens = lemmatize_sentence(query) # lemmatize tokens to use as in vocabulary
    query_vector = []
    query_tf = {}
    total_query_vocab = 0
    for tok in query_tokens:
        try:
            indexvalue = vocab.index(tok)
            query_vector.append(indexvalue)
            query_tf[indexvalue] = 1 + query_tf.get(indexvalue,0)
            total_query_vocab += 1
        except ValueError: # Token doesnt exist in vocab - ignored
            print(tok, "does not exist in the vocabulary. - Ignoring")


    print("Query as vocab indices:", query_vector)
    print()
    start_time = time.time() # Timer starts

    # First we obtain the list of all possible documents we actually need to search
    # This is a union of the docs in each query term's posting list
    # Not an intersection because we use cosine similarity and not boolean retrieval
    possible_docs = set()
    query_tf_vector = []

    for q in query_vector:
        possible_docs = possible_docs.union(posting_list[q].keys())
        query_tf_vector.append(query_tf[q]/total_query_vocab)
        # We also generate a TDF vector for the query. Does not make sense to scale with IDF

    # Run through each doc and generate the vector corresponding to the query terms
    # Compute the cosine similarities of it vs the TF vector of the query
    # Ties are broken by the magnitude of the vector - note that this is obtained by only considering the query terms
    # Plus these query term weights were scaled with relative TF, so a higher magnitude means the terms were more important
    doc_scores = {}
    for doc in possible_docs:
        doc_vector = []
        for q in query_vector:
            doc_vector.append(posting_list[q].get(doc,0))
        doc_scores[doc] = (cosine_similarity(doc_vector,query_tf_vector), norm(doc_vector))

    # Results are sorted
    sorted_results = sorted(doc_scores.items(), key=operator.itemgetter(1), reverse=True)

    end_time = time.time() # Timer ends as search portion is complete
    search_time = end_time - start_time

    ct = 0
    print("-------------- SEARCH RESULTS --------------")
    results={}
    results['Documents']={}
    for i in sorted_results:
        fname, rownum = file_dict[i[0]].split(' ')
        rownum = int(rownum[3:])
        search_res = lines[i[0]]
        search_res = search_res.split('\t')[2]
        results['Documents'][i[0]]={'Name': fname, 'Row': rownum, 'Score': i[1], 'Results': search_res}
        ct += 1
        if ct == 10:
            break
    results['Time']=end_time-start_time
    
    return results