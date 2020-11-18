'''
Search the corpus using a query
'''
import pickle
import operator
import time
from pywsd.utils import lemmatize_sentence
from numpy import dot
from numpy.linalg import norm
import nltk
from nltk.corpus import wordnet 
from nltk.corpus import stopwords
from spellchecker import SpellChecker
spell = SpellChecker()

stop_words = set(stopwords.words('english'))


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


# Using a fixed query for testing. Remove and use free input later
# query = input("Query:")

#query = "Donald Trump accuses China of artificially creating climate change"
#query = "Climate change is very important"
#print("Input Query:", query)

while True:
    query = input("Query:")
    k = int(input("Enter K:"))
    query_tokens = lemmatize_sentence(query) # lemmatize tokens to use as in vocabulary
    query_vector = []
    query_tf = {}
    total_query_vocab = 0
    for i in range(len(query_tokens)):
        tok = query_tokens[i]
        try:
            indexvalue = vocab.index(tok)
            query_vector.append(indexvalue)
            query_tf[indexvalue] = 1 + query_tf.get(indexvalue,0)
            total_query_vocab += 1
        except ValueError: # Token doesnt exist in vocab - ignored
            #print(tok, "does not exist in the vocabulary. - Ignoring")
            if tok not in stop_words:
                misspelled = list(spell.unknown([tok]))
                print("invalid -> ",tok)
                if(len(misspelled)==0):
                    print("trying synonyms")
                    syn = list()
                    for synset in wordnet.synsets(tok):
                        for lemma in synset.lemmas():
                            syn.append(lemma.name())    #add the synonyms
                    #print('Synonyms: ' + str(list(set(syn))))
                    found_word = False
                    new_word = ''
                    for word in syn:
                        if word in vocab:
                            new_word = word
                            found_word = True
                            break
                    if found_word:
                        #print('Synonym present in vocab -> ',new_word)
                        indexvalue = vocab.index(new_word)
                        query_vector.append(indexvalue)
                        query_tf[indexvalue] = 1 + query_tf.get(indexvalue,0)
                        total_query_vocab += 1
                    else:
                        print("None of the synonyms present in the vocabulary")
                else:
                    print("trying spelling correction")
                    candidate_list = spell.candidates(misspelled[0])
                    found_word = False
                    corrected_tok = ''
                    #print("Candidates -> ",candidate_list)
                    for word in candidate_list:
                        new_query_tokens = query_tokens
                        new_query_tokens[i] = word
                        new_query = ''
                        for j in range(len(new_query_tokens)):
                            new_query += new_query_tokens[j]+' '
                        new_query = new_query[0:len(new_query)-1]
                        new_lem_query = lemmatize_sentence(new_query)
                        lem_word = new_lem_query[i]
                        #print("word -> ",word, ", present in vocab ->", word in vocab)
                        #print("lem_word -> ",lem_word, ", present in vocab ->", lem_word in vocab)
                        #print("new query -> ",new_query)
                        print("new lemmatized query -> ",new_lem_query)
                        if lem_word in vocab:
                            corrected_tok = lem_word
                            found_word = True
                            break
                    if found_word:
                        #print("corrected -> "+corrected_tok)
                        indexvalue = vocab.index(corrected_tok)
                        query_vector.append(indexvalue)
                        query_tf[indexvalue] = 1 + query_tf.get(indexvalue,0)
                        total_query_vocab += 1
                    else:
                        print("couldnt find any word")


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
    print("Total Relevant Documents", len(sorted_results))
    for i in sorted_results:
        print("# ", ct+1)
        print("Doc No:", i[0],"Score:",i[1])
        fname, rownum = file_dict[i[0]].split(' ')
        rownum = int(rownum[3:])
        print("CSV File:",fname," row:",rownum)
        search_res = lines[i[0]]
        search_res = search_res.split('\t')[2]
        print(search_res)
        print()
        print("##################")
        print()
        ct += 1
        if ct == k:
            break
    print()
    print("Search Time:",end_time-start_time)
