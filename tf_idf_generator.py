'''
Generates the inverted index for the corpus
Also generates an index to file name dictionary and a vocab index to word dictionary
'''

import nltk
from pywsd.utils import lemmatize_sentence # An all in one stemmer + lemmatizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from numpy import log10
import string
import pickle

# Words/Punctuators to ignore
stop_words = set(stopwords.words('english'))
punctuators = string.punctuation

# Computes the raw term frequency per word in a document
def document_tf(doc):
    term_counts = dict()
    sent_tokens = sent_tokenize(doc)
    for sent in sent_tokens:
        word_tokens = lemmatize_sentence(sent)
        for word in word_tokens:
            if (word not in stop_words) and (word not in punctuators):
                term_counts[word] = term_counts.get(word, 0) + 1 #dict.get returns default value specified if key not found

    return term_counts


# Generates the inverted index and assosciated mappings
def corpus_inv_index(corpus):
    tf_idf_list = []                    # List to keep the individual tf dictionaries
    file_name_dict ={}                  # Maps the index with the filename
    term_occurence = dict()             # Keeps track of how many documents the word occured in 
    vocab = set()                       # Vocabulary set
    
    corpus_file = open(corpus,'r')
    corpus_data = corpus_file.readlines()
    doc_count = len(corpus_data)
    print("Total Documents:",doc_count)
    counter = 0

    # First we generate a TF-IDF dictionary for each document
    # We also build the vocabulary and term occurence dictionary while doing this
    for doc_info in corpus_data:
        counter += 1
        print("Processing document :", counter)
        doc_details = doc_info.split('\t')
        document = doc_details[2]
        document = document[:len(document)-1] #Trim newline
        doc_tf = document_tf(document)
        for word in doc_tf:
            term_occurence[word] = term_occurence.get(word, 0) + 1
            vocab.add(word)
        tf_idf_list.append(doc_tf)
        file_name_dict[counter-1] = doc_details[0]+" Row"+doc_details[1]

    # We scale the Term frequencies with IDF
    print("Scaling with IDF.....")
    for doc_tf in tf_idf_list:
        for word in doc_tf:
            doc_tf[word] *= log10(doc_count/(1+term_occurence[word]))

    # Generate the mapping to convert each word into an index
    print("Generating vocabulary mapping.....")
    vocab = list(vocab)
    vocab.sort()
    vocab_length = len(vocab)

    # Finally we generate the inverted index
    # Inverted matrix : Each row is a vocab word's posting list dictionary
    # posting list dictionary is a doc index and it's tf-idf score
    print("Creating inverted index......")
    inverted_index = []
    for word_counter in range(vocab_length):
        posting_dict = {}
        for doc_counter in range(doc_count):
            tf_idf_score = tf_idf_list[doc_counter].get(vocab[word_counter],0)
            if tf_idf_score!=0:
                posting_dict[doc_counter] = tf_idf_score
        inverted_index.append(posting_dict)
        

    return inverted_index, file_name_dict, vocab


corpus_name = "tvnews_corpus.tsv" # Change appropriately
inverted_index, file_dict,vocab = corpus_inv_index(corpus_name)

# The processed corpus file contains three parts:
# Inverted index
# Doc index to csv file and row mapping
# Vocabulary mapping

with open("inv_index","wb") as f:
    pickle.dump(inverted_index,f)
    pickle.dump(file_dict, f)
    pickle.dump(vocab,f)
print()
print("Inverted index with assosciated mapping files has been generated")    
    
    
