from spellchecker import SpellChecker

spell = SpellChecker()

# find those words that may be misspelled
# inputfile= open("sample_queries.txt","r")
# for line in inputfile.readlines():
# 	tokens = line.split()
# 	print(line)
# 	misspelled = spell.unknown(tokens)
# 	for word in misspelled:
# 		print(word,spell.correction(word),spell.candidates(word))

misspelled = spell.unknown(['acuses'])
for word in misspelled:
	print(word,spell.correction(word),spell.candidates(word))