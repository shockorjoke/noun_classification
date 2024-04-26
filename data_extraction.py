import pickle
import spacy
from datasets import load_dataset
#from numba import jit, cuda
from timeit import default_timer as timer   
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
spacy.prefer_gpu()
dataset = load_dataset("bookcorpus")
nlp = spacy.load("en_core_web_trf")
"""Extracts the sentences of a noun from Brown corpus. Saves the sentences as a list of str.

"""
#with open(f"selected 100 sents.txt", "r") as ds:
#    dataset = ds.readlines()

#@jit(target_backend='cuda') 
def extract_noun_sentences(target_nouns, dataset):
    """Extract the sentences of a noun from Brown corpus.
    
    Returns:
    list: sentences (str) that contain the target noun."""
    #Edit between here
    sentence_with_directed_noun = {}
    seen = []
    count=0
    for split in dataset.keys():
        data = dataset[split]
        #use this for small data
        sentences = data["text"]
        #sentences = data["text"]
        for sentence in sentences:
            count+=1
            if count%1000000 == 0:
                print(count)
            for target_noun in target_nouns:
                if target_noun in sentence:
                    if sentence not in seen:
                        doc = nlp(sentence)
                        for token in doc:
                            if token.text == target_noun and token.dep_ == "dobj":
                                seen.append(sentence)
                                if target_noun not in sentence_with_directed_noun:
                                    sentence_with_directed_noun[target_noun]=[sentence]
                                else:
                                    sentence_with_directed_noun[target_noun].append(sentence)
                    else: pass
                else: pass
    return sentence_with_directed_noun

target_nouns = ["dinner","breakfast","feast","meal","buffet","speech","response","submission","lecture","conversation","brochure","textbook","novel","diary","summary","boat","classroom","cave","apartment","tower"]




start = timer()
sentence_with_directed_noun = extract_noun_sentences(target_nouns, dataset) 
for target_noun, sentences in sentence_with_directed_noun.items():
    if len(sentences)>0:
        with open(f"sentences/{target_noun}.pickle", "wb") as sentence_file:
            pickle.dump(sentences, sentence_file)
print("with GPU:", timer()-start)
