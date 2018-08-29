# -*- coding: utf-8 -*
import os

import gensim

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH  = os.path.join(SCRIPT_PATH, 'model/')

if __name__ == '__main__':

    # Load word2vec model
    model = gensim.models.Word2Vec.load(os.path.join(MODEL_PATH, 'word2vec.model'), mmap='r')

while True:
    try:
        query = input("Entry word with ' ' \n")
        query_list = query.split(",")
        print len(query_list)

        if len(query_list) == 1:
            print("The top 10 similiar word is...:")
            res = model.most_similar(query_list[0], topn=10)
            for item in res:
                print(item[0] + "," + str(item[1]))
        elif len(query_list) == 2:
            print("The similiarity of these two words are...:")
            res = model.similarity(query_list[0], query_list[1])
            print(res)
        else:
            print("%s = %s，%s = ..." % (query_list[0], query_list[1], query_list[2]))
            res = model.most_similar(positive=[query_list[2], query_list[1]], negative=[query_list[0]], topn=5)
            for item in res:
                print(item[0] + "," + str(item[1]))
    except Exception as e:
        print("Error:" + repr(e))
