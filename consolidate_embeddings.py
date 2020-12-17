import os
import string
import pickle
from helper_funcs import clear_dir
def data_mapping():
    clear_dir("processed_data/")
    vocabfile = open("elmo-base-vocab.txt","r", encoding="utf-8")
    words = vocabfile.read().split("\n")
        # for t in tokens:
        #     if len(t)<2:
        #         tokens.remove(t)
    embeddings = []
    num_files = len(os.listdir("embeddingfs/"))
    for i in range(num_files):
        print("file"+str(i)+".p")
        embedding_chunk = pickle.load(open("embeddingfs/file"+str(i)+".p","rb"))
        for embedding in embedding_chunk:
            embeddings.append(embedding[0])
    pickle.dump(words,open("processed_data/words.p","wb"))
    pickle.dump(embeddings,open("processed_data/embeddings.p","wb"))
