import nltk
from nltk.corpus import wordnet   #Import wordnet from the NLTK
syn = list()
ant = list()
for synset in wordnet.synsets("Worse"):
	for lemma in synset.lemmas():
		syn.append(lemma.name())    #add the synonyms
		if lemma.antonyms():    #When antonyms are available, add them into the list
			ant.append(lemma.antonyms()[0].name())
print('Synonyms: ' + str(list(set(syn))))
print('Antonyms: ' + str(list(set(ant))))