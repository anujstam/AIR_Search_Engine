'''
Creates a corpus from the entire dataset
Results in a total of 94858 total documents - where each doc is a row of a csv
'''
import glob

# Converts a csv row into a document
def doc_from_row(csvrow):
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

# Goes through all of the individual CSV files in a folder
# Considers each row to be a document which is a child of the csv
# Allows for search on a per row level
# Generates an intermediate file that's easier to process
def generate_news_corpus(base_dir, corpus_name="tvnews_corpus"):
    total_docs = 0
    counter = 1
    for file in glob.glob(base_dir+"//*.csv"):
        print("CSV Number:", counter)
        corpus_text = ""
        with open(file,'r',encoding='latin-1') as newsfile:
            news_items = newsfile.readlines()
            news_items = news_items[1:] # Remove header
            row_counter = 0
            for item in news_items:
                row_text = doc_from_row(item)
                if row_text:
                    total_docs +=1
                    corpus_text += file + '\t' + str(row_counter) + '\t' + row_text + '\n'
                row_counter +=1
        with open(corpus_name+".tsv","a",errors='ignore') as corpus_file:
            corpus_file.write(corpus_text)
        counter += 1
    print()
    print("###################")
    print("Corpus generated from all CSVs in directory")
    print("Total document(rows) count: ", total_docs)
    


base_dir = "TelevisionNews"
generate_news_corpus(base_dir)
