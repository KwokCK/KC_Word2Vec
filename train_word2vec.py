#! /usr/bin/env python

"""
Hallo

"""

import sys
import gensim

#from gensim.models.doc2vec import Word2Vec
from gensim.models import Word2Vec

def read_file(doc_path):

    sentences = []
    with open(doc_path, "r") as f:
        for line in f:
            sentences.append(line.encode('utf-8').strip("\n").split(" "))

    return sentences


def train_model(doc_path, output_path, dim=100):
    """
        python train_word2vec.py test_data/test.txt test_data/test.model
    """
    print ("Reading a file ...")
    sentences = read_file(doc_path)
    print ("Training a model ...")
    model = Word2Vec(sentences, min_count=0, size=dim, window=10)
    print ("Saving the moles ...")
    model.save(output_path)
    print ("Done.")


def main(doc_path, output_path, dim=100):
    """
    Main function
    """
    train_model(doc_path, output_path, dim)


if __name__ == "__main__":
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print (__doc__)
    else:
        if len(sys.argv) == 3:
            main(sys.argv[1], sys.argv[2])
        else:
            main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
