from search import searchAPI
import pandas as pd

testcasefile="TestCase1"
outputcsvfile=open("Results"+testcasefile+".csv", "w")
inputfile= open(testcasefile+".txt","r")
for line in inputfile.readlines():
    results = searchAPI(line)
    print(results)
    