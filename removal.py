import io
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# word_tokenize accepts a string as an input, not a file.

# Remove stopwords
to_remove = ['for', 'do', 'while']
stop_words = set(stopwords.words('english')).difference(to_remove)

file1 = open("pdfdata/outputPDF.text")
line = file1.read()# Use this to read file content as a stream:
words = line.split()
for r in words:
    if not r in stop_words:
        appendFile = open('pdfdata/StopWordRemovelPDF.text','a')
        appendFile.write(" "+r)
        appendFile.close()

# Stemming
# punctuation
# numbers
