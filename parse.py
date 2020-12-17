import operator
import os
import re

# Function that generates a vocab file for ELISE preprocessing code given some input directory with
# .txt files. It extracts all the tokens in these files to build the vocab file. The tokens in the 
# vocab file are ordered by frequency. Name of the vocab file outputted is "elmo-base-vocab.txt".
def create_vocab_file(input_folder):
    vocab_freq_dict = {} #where we will write the words
    directory_list = list(os.listdir(input_folder))
    for file_name in directory_list:
        if file_name.endswith(".txt"): #we are pulling ALL the words from the actual files
            with open(input_folder+file_name, "r") as txt_file:
                lines = txt_file.read().split("\n")
                for line in lines:
                    tokens = line.split(" ")
                    for token in tokens:
                        clean_token = re.sub(r'[^\w\s]', '', token) #remove punctuation from tokens
                        if len(clean_token)>1: #tokens should be at least length 2, rest we ignore
                            if clean_token not in vocab_freq_dict: #add it to the dict, 
                                                                    
                                vocab_freq_dict[clean_token] = 1
                            else: #increment if it's already there
                                vocab_freq_dict[clean_token] += 1
    #Sort by frequency
    sorted_vocab_freq_tuples = sorted(vocab_freq_dict.items(), key=operator.itemgetter(1), reverse=True) 
    with open("elmo-base-vocab.txt", "w") as output_file:
        for token_tuple in sorted_vocab_freq_tuples: #finally, output the tokens to file
            output_file.write(token_tuple[0])
            output_file.write("\n")
