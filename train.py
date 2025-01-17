import os
import sys
import bz2
import logging
import multiprocessing

import gensim

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH   = os.path.join(SCRIPT_PATH, 'pdfdata/')
MODEL_PATH  = os.path.join(SCRIPT_PATH, 'model/')

DICTIONARY_FILEPATH = os.path.join(DATA_PATH, 'english_wordids.txt.bz2')
#myDictionaryFILEPATH = os.path.join(DATA_PATH, 'outputPDF.text')
WIKI_DUMP_FILEPATH = os.path.join(DATA_PATH, 'StopWordRemovelPDF.text')

if __name__ == '__main__':

    # Check if the required files have been downloaded
    if not WIKI_DUMP_FILEPATH:
        print('Wikipedia articles dump could not be found..')
        sys.exit()

    # Get number of available cpus
    cores = multiprocessing.cpu_count()

    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    # Initialize logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    if not os.path.isfile(DICTIONARY_FILEPATH):
        logging.info('Dictionary has not been created yet..')

        # Construct corpus
        wiki = gensim.corpora.TextCorpus(WIKI_DUMP_FILEPATH)

        # Remove words occuring less than 20 times, and words occuring in more
        # than 10% of the documents. (keep_n is the vocabulary size)
        wiki.dictionary.filter_extremes(no_below=20, no_above=0.1, keep_n=10000)

        # Save dictionary to file
        wiki.dictionary.save_as_text(DICTIONARY_FILEPATH)
        del wiki

    # Load dictionary from file
    dictionary = gensim.corpora.Dictionary.load_from_text(DICTIONARY_FILEPATH)

    # Construct corpus using dictionary
    wiki = gensim.corpora.TextCorpus(WIKI_DUMP_FILEPATH, dictionary=dictionary)

    class SentencesIterator:
        def __init__(self, wiki):
            self.wiki = wiki

        def __iter__(self):
            for sentence in self.wiki.get_texts():
                yield list(map(lambda x: x.decode('utf-8'), sentence))

    # Initialize simple sentence iterator required for the Word2Vec model
    sentences = SentencesIterator(wiki)

    logging.info('Training word2vec model..')
    model = gensim.models.Word2Vec(sentences=sentences, size=300, min_count=1, window=5, workers=cores)

    # Save model
    logging.info('Saving model..')
    model.save(os.path.join(MODEL_PATH, 'word2vec.model'))
    logging.info('Done training word2vec model!')
