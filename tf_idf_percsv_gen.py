'''
Generates the inverted index for the corpus
Also generates an index to file name dictionary and a vocab index to word dictionary

This version does IDF on a per csv file level
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

# Computes the term frequency per word in a document (read - row)
# Originally used raw count, but by using relative frequency, we can generate vectors of the same dimension as the query
# this is because now we have the info. about the length of the document in built into the embedding
def document_tf(doc):
    term_counts = dict()
    sent_tokens = sent_tokenize(doc)
    word_count = 0
    for sent in sent_tokens:
        word_tokens = lemmatize_sentence(sent)
        for word in word_tokens:
            if (word not in stop_words) and (word not in punctuators):
                word_count +=1 # keep track of total no. of words in the doc
                term_counts[word] = term_counts.get(word, 0) + 1 #dict.get returns default value specified if key not found
                
    for k in term_counts:
        term_counts[k] /= word_count # convert raw occurence count to relative frequency 
    return term_counts

# returns the text from a row
def text_from_row(csvrow):
    doc_text = ""
    data = csvrow.split(',',6) # Prevents splitting in snippet
    if len(data) == 7:
        url, date, station, show, show_ID, thumbnail, snippet = data #Unravel the data
        snippet = snippet[1:len(snippet)-2] # Removing the trailing newline and quotes
        snippet = snippet.replace('\n',' ') # Removes any additional newlines
        doc_text = date + " " + station + " " + show + " " + snippet
        return doc_text
    else:
        return None

def parse_csv(csvfile):
    csv_vocab = {} # store the vocabulary of the csv_file, with occurence count
    with open(csvfile, "r") as f:
        lines = f.readlines()
        lines = lines[1:] # strip header
        for row in lines:
            row_text = text_from_row(row)
            doc_tf = document_tf(row_text)
        for word in doc_tf:
            csv_vocab[word] = csv_vocab.get(word,0) + 1
        
            
        


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

    per_csv_occurence = {} # Akin to term occurence, but we reset this for each csv file

    current_csv = None
    last_index = 0

    # First we generate a TF-IDF dictionary for each document
    # We also build the vocabulary and term occurence dictionary while doing this
    for doc_info in corpus_data:
        doc_details = doc_info.split('\t')

        #If it's a different CSV file, do the IDF scaling and reset the occurence counter
        csv_name = doc_details[0]
        if csv_name != current_csv:
            print("New CSV:", csv_name)
            if current_csv is not None:
                print("Indexing over:", last_index, counter)
                for i in range(last_index, counter): # Loop through doc-tf vectors from previous file
                    post_list = tf_idf_list[i]
                    for word in post_list:
                        post_list[word] *= log10((counter-last_index)/(1+per_csv_occurence[word]))
                last_index = counter

            current_csv = csv_name
            per_csv_occurence = {}

        counter += 1            

            
        document = doc_details[2]
        document = document[:len(document)-1] #Trim newline
        doc_tf = document_tf(document)
        
        for word in doc_tf:
            per_csv_occurence[word] = per_csv_occurence.get(word, 0) + 1
            vocab.add(word)
        tf_idf_list.append(doc_tf)
        file_name_dict[counter-1] = doc_details[0]+" Row"+doc_details[1]

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

with open("inv_index_percsv_version","wb") as f:
    pickle.dump(inverted_index,f)
    pickle.dump(file_dict, f)
    pickle.dump(vocab,f)
print()
print("Inverted index with assosciated mapping files has been generated")    
    
    
