import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import re
import string
import tensorflow_hub as hub
import pickle
import threading
from helper_funcs import clear_dir
class VocabProcessorBase:
    num_threads = 2
    def __init__(self, num_threads):
        self.num_threads = num_threads+1
        clear_dir("embeddingfs/")
        vocabfile = open("elmo-base-vocab.txt","r", encoding="utf-8")
        tokens = vocabfile.read().split("\n")
        labels =[]
        batches = [tokens[i * 100:(i + 1) * 100] for i in range((len(tokens) + 100 - 1) // 100)]
        threadbatch = int(len(batches)/self.num_threads)
        threads = list()
        for i in range(self.num_threads):
            if i!=0:
                x = threading.Thread(target=self.generate_embeddings, args=(i,threadbatch,batches))
                threads.append(x)
                x.start()
        for index, thread in enumerate(threads):
            thread.join()

    def generate_embeddings(self, threadnum, batchsize, batches):
        elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
        if threadnum == self.num_threads-1:
            for i in range((threadnum-1)*batchsize,len(batches)):
                print("Thread: "+str(threadnum)+" sentence: "+str(i)+"/"+str(len(batches)-1))
                batch=batches[i]
                batch_embeddings =elmo(batch, as_dict=True, signature='default')["elmo"]
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.tables_initializer())
                    batch_embeddings = sess.run(batch_embeddings)
                    pickle.dump(batch_embeddings,open("embeddingfs/file"+str(i)+".p","wb"))
        else:
            for i in range((threadnum-1)*batchsize,threadnum*batchsize):
                print("Thread: "+str(threadnum)+" sentence: "+str(i)+"/"+str(threadnum*batchsize-1))
                batch=batches[i]
                batch_embeddings = elmo(batch, as_dict=True, signature='default')["elmo"]
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.tables_initializer())
                    batch_embeddings = sess.run(batch_embeddings)
                    pickle.dump(batch_embeddings,open("embeddingfs/file"+str(i)+".p","wb"))
    
