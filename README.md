# UE17CS412-AIR Assignment 
## Search Engine on Environmental News NLP Dataset
Rudimentary Search Engine on the Environmental news dataset

Download the dataset from here : https://www.kaggle.com/amritvirsinghx/environmental-news-nlp-dataset

Extract the contents, and clone the repo in the same folder

Get the inverted index here : https://drive.google.com/drive/folders/1i4ZZ_yaGzxvmyKGNoY92VDPfanD02OfE?usp=sharing
Get the results of Our Model vs Elastic Search here : https://drive.google.com/file/d/19wQG2DA6o_ArZRlsK3AFeZEwGsqkn4xX/view?usp=sharing
--

### First time setup:
Run corpus_generator.py to generate an intermediary csv file that is easier to process

Run tf_idf_generator.py to create the inverted index and other mapping files. This file can take a while to run due to the number of documents.

### Additional Details

The search.py offers an API like function to search for a given string using the index generated.

The search_vs_elastic_search.ipynb is a jupyter notebook initally used on Kaggle to compare the results between our model and ElasticSearch.

The testcases are present in sample_queries.txt which has been generated smartly keeping in mind a few corner cases.

Images of outputs are present in the Images/ folder
--

### To Search
Run AutomatedSearch.py which automatically takes each line as input from a test case file and prints the results for the same.
