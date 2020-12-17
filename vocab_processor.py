import re
import string
import tensorflow as tf
import tensorflow_hub as hub
import pickle
import threading

from allennlp.commands.elmo import ElmoEmbedder


#test = ["5mg","ibuprofen"]
elmogang =ElmoEmbedder(options_file="config-files/pubmed.json", weight_file="config-files/pubmed-weights.hdf5")
#bruhbruh = list(elmogang.embed_sentences(test))
#print(bruhbruh)
#exit()
num_threads = 2

def func(threadnum,batchsize):
    
   #elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
    if threadnum == num_threads-1:
        for i in range((threadnum-1)*batchsize,len(batches)):
            print("Thread: "+str(threadnum)+" sentence: "+str(i)+"/"+str(len(batches)-1))
            b=batches[i]
            elmo_input = []
            for y in range(len(b)):
                elmo_input.append( [b[y]])
            #thing =elmo(b, as_dict=True, signature='default')["elmo"]
            #with tf.Session() as sess:
            #    sess.run(tf.global_variables_initializer())
            #    sess.run(tf.tables_initializer())
            #    thing = sess.run(thing)
    #print(token_embeddings)
            thing = list(elmogang.embed_sentences(elmo_input))
            elmo_output =[]
            for y in range(len(thing)):
                temp = []
                for t in range(1024):
                    temp.append((thing[y][0][0][t]+thing[y][1][0][t]+thing[y][2][0][t])/3)

                elmo_output.append(temp)
            pickle.dump(elmo_output,open("embeddingfs/file"+str(i)+".p","wb"))
    else:
        for i in range((threadnum-1)*batchsize,threadnum*batchsize):
            print("Thread: "+str(threadnum)+" sentence: "+str(i)+"/"+str(threadnum*batchsize-1))
            b=batches[i]
            thing =elmo(b, as_dict=True, signature='default')["elmo"]
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.tables_initializer())
                thing = sess.run(thing)
    #print(token_embeddings)
                pickle.dump(thing,open("embeddingfs/file"+str(i)+".p","wb"))

vocabfile = open("vocabulary.txt","r", encoding="utf-8")
tokens = vocabfile.read().split("\n")
labels =[]

##only for specific dataset
#with open("entity-labels.txt","r") as f:
#    labels = f.read().split("\n")

tokens = tokens[3:]
for i in range(len(tokens)):
    word = tokens[i]
    word = word.translate(str.maketrans('', '', string.punctuation))
    word = word.replace(" ","")
    tokens[i] = word
entities={}
##only if we're being specific with data
#for i in range(len(tokens)):
#    if tokens[i] not in entities:
#        entities[tokens[i]]=labels[i]
#tokens = list(entities.keys())

bruh = []
for t in tokens: #otherwise uncomment this
     if len(t)<2 or t in bruh:
         tokens.remove(t)
     else:
         bruh.append(t)
#pickle.dump(entities,open("processed_data/entities.p","wb"))
print(len(tokens))
#exit()
batches = [tokens[i * 100:(i + 1) * 100] for i in range((len(tokens) + 100 - 1) // 100)]
threadbatch = int(len(batches)/num_threads)
threads = list()
for i in range(num_threads):
    if i!=0:
        x = threading.Thread(target=func, args=(i,threadbatch))
        threads.append(x)
        x.start()
for index, thread in enumerate(threads):
    thread.join()

