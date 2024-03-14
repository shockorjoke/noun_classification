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

Your Tasks:
- Write the method extract_noun_sentences()
"""
#@jit(target_backend='cuda') 
def extract_noun_sentences(target_nouns, dataset):
    """Extract the sentences of a noun from Brown corpus.
    
    Returns:
    list: sentences (str) that contain the target noun."""
    
    sentence_with_directed_noun = {}
    for split in dataset.keys():
        data = dataset[split]
        #use this for small data
        sentences = data["text"]
        #sentences = data["text"]
        for sentence in sentences:
            for target_noun in target_nouns:
                if target_noun in sentence:
                    doc = nlp(sentence)
                    for token in doc:
                        if token.text == target_noun and token.dep_ == "dobj":
                            if target_noun not in sentence_with_directed_noun:
                                sentence_with_directed_noun[target_noun]=[sentence]
                            else:
                                if sentence not in sentence_with_directed_noun[target_noun]: #this statement was insert because it seem that there are some sentences which appear 2 time in corpus
                                    sentence_with_directed_noun[target_noun].append(sentence)
    return sentence_with_directed_noun

    
target_nouns = ["dinner","breakfast","feast","meal","buffet","speech","response","submission","lecture","conversation","brochure","textbook","novel","diary","summary","boat","classroom","cave","apartment","tower"]



start = timer()
sentence_with_directed_noun = extract_noun_sentences(target_nouns, dataset) 
for target_noun, sentences in sentence_with_directed_noun.items():
    if len(sentences)>0:
        with open(f"sentences/{target_noun}.pickle", "wb") as sentence_file:
            pickle.dump(sentences, sentence_file)
print("Time: ", timer()-start)
