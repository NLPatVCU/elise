from bert_embedding import BertEmbedding
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pickle
import threading
from helper_funcs import clear_dir
class VocabProcessorBert:
    num_threads = 2
    def __init__(self, num_threads):
        self.num_threads = num_threads+1
        clear_dir("embeddingfs/")
        vocabfile = open("elmo-base-vocab.txt","r",encoding="utf-8")
        tokens = vocabfile.read().split("\n")
        batches = [tokens[i * 100:(i + 1) * 100] for i in range((len(tokens) + 100 - 1) // 100)]
        threadbatch = int(len(batches)/self.num_threads)
        threads = list()
        for i in range(self.num_threads):
            if i != 0:
                x = threading.Thread(target=self.generate_embeddings, args=(i, threadbatch, batches))
                threads.append(x)
                x.start()
        for index, thread in enumerate(threads):
            thread.join()
    def generate_embeddings(self, threadnum, batchsize, batches):
        bert_embedding = BertEmbedding(model='bert_12_768_12',dataset_name="book_corpus_wiki_en_uncased")
        if threadnum == self.num_threads-1:
            for i in range((threadnum-1)*batchsize, len(batches)):
                print("Thread: "+str(threadnum)+" sentence:"+str(i)+"/"+str(len(batches)-1))
                batch=batches[i]
                batch_embeddings = bert_embedding(batch)
                output = []
                for emb in batch_embeddings:
                    output.append(emb[1][0])
                pickle.dump(output, open("embeddingfs/file"+str(i)+".p","wb"))
        else:
            for i in range((threadnum-1)*batchsize,threadnum*batchsize):
                print("Thread: "+str(threadnum)+" sentence: "+str(i)+"/"+str(threadnum*batchsize-1))
                batch=batches[i]
                batch_embeddings = bert_embedding(batch)
                output=[]
                for emb in batch_embeddings:
                    output.append(emb[1][0])
                pickle.dump(output, open("embeddingfs/file"+str(i)+".p","wb"))
                        

