from parse import create_vocab_file
from vocab_processor_base import VocabProcessorBase
from vocab_processor_bert import VocabProcessorBert
from consolidate_embeddings import data_mapping
from clustering_algorithm import generate_similarity_matrix
from clustering_algorithm import pivot
from clustering_algorithm import merge_clusters
import argparse
class ParameterError(Exception):
    def __init__(self, message):
        self.message=message
        super().__init__(self.message)

num_threads=1
parser = argparse.ArgumentParser(description="Run ELISE")
parser.add_argument("--first_run", help="Initiate preprocessing-step.",action="store_true")
parser.add_argument("emb", help="Which model to use for generating embeddings. 'BERT' or 'BASE_ELMO'.")
parser.add_argument("--train", help="Path to training corpus. Required if --first_run")
parser.add_argument("--threads", type=int,help="Number of threads to use. Default is 1.")
args = parser.parse_args()
bert = args.emb=="BERT"
base_elmo= args.emb=="BASE_ELMO"
first_run=args.first_run
train_data_path = args.train
if args.threads != None:
    num_threads = args.threads
if (bert==base_elmo):
    raise ParameterError("emb flag should be either 'BERT' or 'BASE_ELMO'.")
if num_threads<1:
    raise ParameterError("--threads should be greater than 0.")
if first_run:
    if train_data_path==None:
        raise ParameterError("If --first_run, then --train is required.")
    print("Creating vocab file...")
    #create_vocab_file(train_data_path)
    print("Done!")
    if base_elmo:
        print("Generating ELMo [base] embeddings...")
        #VocabProcessorBase(num_threads)
    elif bert:
        print("Generating BERT embeddings...")
        #VocabProcessorBert(num_threads)
    print("Done!")
    print("Data mapping...")
    #data_mapping()
    print("Done!")
#    generate_similarity_matrix(num_threads)
    #pivot(num_threads)
    # merge_clusters(num_threads)

