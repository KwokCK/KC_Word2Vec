from __future__ import print_function

import logging
import os
import sys
import multiprocessing
import gensim

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# step 1    : output pdf into txt
#           : Command in python cosole  : textract pdfdata/Thinking_in_Java_4th_edition.pdf > pdfdata/outputPDF.text
# step 2    : Stop word removel and remove punctuation
# step 3    : train PC
# step 4    : test2.py
 
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 4:
        print("Useing: python train_word2vec_model.py input_text output_gensim_model output_word_vector")
        sys.exit(1)
    inp, outp1, outp2 = sys.argv[1:4]

    #SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
    #DATA_PATH = os.path.join(SCRIPT_PATH, '')
    #WIKI_DUMP_FILEPATH = os.path.join(DATA_PATH, 'enwiki-20180801-pages-articles.xml.bz2')
    #wiki = gensim.corpora.WikiCorpus(WIKI_DUMP_FILEPATH)
    #wiki.dictionary.filter_extremes(no_below=20, no_above=0.1)

    model = Word2Vec(LineSentence(inp),
                     size=200,
                     window=5,
                     min_count=5,
                     workers=multiprocessing.cpu_count()
                     )

    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)
