import pickle
import string
import os
from scipy import spatial
import threading
from helper_funcs import clear_dir
num_threads = 2

def pivot_func(clusters, sim_matrix, threadnum):
	
	batchsize = int(len(clusters) /num_threads)
	if threadnum == num_threads - 1:
		print(threadnum)
		pivot_list=[]
		progress = 1
		end = len(clusters)-(threadnum-1)*batchsize-1
		i=0

		for i in range((threadnum - 1) * batchsize, len(clusters)):
			c = list(clusters.keys())[i]
			print("Thread: "+str(threadnum)+" "+str(progress)+"/"+str(end))
			progress += 1
			node_strengths = []
			for t in clusters[c]:
				connections = []
				for ti in clusters[c]:
					connections.append(sim_matrix[t][ti])
				node_strengths.append(sum(connections)/len(connections))
			max = 0.0
			maxw = ""
			for x in range(len(node_strengths)):
            	#print(thing[i])

				if node_strengths[x]>max:
					max = node_strengths[x]
					maxw = clusters[c][x]
			pivot_list.append(maxw)
		pickle.dump(pivot_list, open("pivot_lists/list"+str(threadnum)+".p","wb"))
		print("we made it")
	#else:
	#	pivot_list=[]
	#	progress = 1
	#	end = batchsize
	#	i=0
#
#		for i in range((threadnum - 1) * batchsize, threadnum*batchsize):
#			c = list(clusters.keys())[i]
#			print("Thread: "+ str(threadnum)+" "+str(progress)+"/"+str(end))
#			progress += 1
#			node_strengths = []
#			for t in clusters[c]:
#				connections = []
#				for ti in clusters[c]:
#					connections.append(sim_matrix[t][ti])
#				node_strengths.append(sum(connections)/len(connections))
#			max = 0.0
#			maxw = ""
#			for x in range(len(node_strengths)):
            	#print(thing[i])

#				if node_strengths[x]>max:
#					max = node_strengths[x]
#					maxw = clusters[c][x]
#			pivot_list.append(maxw)
		
#		pickle.dump(pivot_list, open("pivot_lists/list"+str(threadnum)+".p","wb"))

	
def cluster_func(vo, c, sm, threadnum):
	batchsize = int(len(vo) / num_threads)
	if threadnum == num_threads - 1:
		for i in range((threadnum - 1) * batchsize, len(vo)):
			v = list(vo.keys())[i]
			print("Thread: " + str(threadnum) + " " + str(i) + "/" + str(len(vo)))
			cluster = []
			sim = {}
			for vv in vo:
				result = 1 - spatial.distance.cosine(vo[v], vo[vv])
				sim[vv] = result
				if (result >= .5):
					cluster.append(vv)
			c[v] = cluster
			sm[v] = sim
	else:
		for i in range((threadnum - 1) * batchsize, threadnum * batchsize):
			v = list(vo.keys())[i]
			print("Thread: " + str(threadnum) + " " + str(i) + "/" + str(threadnum*batchsize-1))
			cluster = []
			sim = {}
			for vv in vo:
				result = 1 - spatial.distance.cosine(vo[v], vo[vv])
				sim[vv] = result
				if (result >= .5):
					cluster.append(vv)
			c[v] = cluster
			sm[v] = sim
    # print(len(os.listdir("embeddingfs/")))
def generate_similarity_matrix(thread_num):
	global num_threads
	num_threads = thread_num
	words = pickle.load(open("processed_data/words.p","rb"))
	embeddings = pickle.load(open("processed_data/embeddings.p","rb"))
    # print(len(embeddings))
	clear_dir("clusters_and_similarity/")
        
	vocab = {}
	clusters = {}
	similarity_matrix = {}
	progress = 1
	for i in range(len(words)):
		vocab[words[i]]=embeddings[i]
	threads = list()
	for i in range(num_threads):
		if i==189:
			x = threading.Thread(target=cluster_func, args=(vocab,clusters,similarity_matrix,i))
			threads.append(x)
			x.start()
	for index, thread in enumerate(threads):
		thread.join()
	pickle.dump(clusters,open("clusters_and_similarity/clusters.p","wb"))
	pickle.dump(similarity_matrix,open("clusters_and_similarity/similarity.p","wb"))
    
def pivot(thread_num):
	global num_threads
	num_threads = thread_num
	clusters = pickle.load(open("clusters_and_similarity/clusters.p","rb"))
	sim_matrix = pickle.load(open("clusters_and_similarity/similarity.p","rb"))
	threads = list()
	for i in range(num_threads):
        	if i!=0:
            		x = threading.Thread(target=pivot_func, args=(clusters,sim_matrix,i))
            		threads.append(x)
            		x.start()
	for index, thread in enumerate(threads):
        	thread.join()
def merge_clusters(thread_num):
	clusters = pickle.load(open("clusters_and_similarity/clusters.p","rb"))
	condensed_clusters ={}
	for i in range(1,thread_num):
		print("Progress: " + str(i) + "/" + str(thread_num-1))
		pivot_list=pickle.load(open("pivot_lists/list"+str(i)+".p","rb"))
		for p in pivot_list:
			print("Total Clusters: " + str(len(condensed_clusters)))
			if p not in condensed_clusters:
				condensed_clusters[p] = list(clusters.values())[i]
			else:
				for token in list(clusters.values())[i]:
					if token not in condensed_clusters[p]:
						condensed_clusters[p].append(token)
	pickle.dump(condensed_clusters, open("clusters_and_similarity/condensed_clusters.p", "wb"))
