import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModel
import spacy
import os
from timeit import default_timer as timer  
import re

"""Classifies the verb-obj pairs in the sentences of a noun.

1. Parse each sentence to determine the verb-direct object pairs in the sentences: get_verbobj().
2. Calculate the relation vector for the verb-direct object pairs in each sentence:  calculate_relation_vector(), get_averaged_vector().
3. Classify the relation vector with the relevant type classifier.
4. Print out the results in the required form.


Your Tasks:
- Fill out the parts for the steps 1 and 4. 
**The step 2 and 3 are already written and ready to use.
"""

spacy.prefer_gpu()
def get_verbobj(parse_sentence, target_noun):
    """Parses the sentence and returns the verb-direct object pair. The target noun should be the direct object of the verb in the sentence.
    Returns:
    tuple: the verb and the target_noun as strings (verb, target_noun), e.g. ("ate", "dinner")
    """
    # Edit between here
    #parse_sentence = nlp(sentence)
    for word in parse_sentence:
        if word.dep_=="dobj" and word.pos_ == "NOUN" and word.lemma_ == target_noun:
            return(word.head.text,word.text) 
        
def get_adjobj(parse_sentence, target_noun):
    #parse_sentence = nlp(sentence)
    for word in parse_sentence:
        if word.dep_=="amod" and word.pos_ == "ADJ" and word.head.text == target_noun and word.head.dep_=="dobj":
            #print(word.head.dep_)
            return(word.head.text,word.text) 
    # Edit between here


def get_averaged_vector(tokenizer, model, sentence, word):
    """Extracts the embedding of the verb and the object from the pre-trained model."""
    inputs = tokenizer(sentence, truncation=True, return_tensors="pt")
    word_char = sentence.index(word)
    token_id = inputs.char_to_token(word_char)
    output = model(**inputs, output_hidden_states=True).hidden_states
    vectors = [
        output[layer_no][:, token_id, :].detach().numpy() for layer_no in range(8, 12)
    ]
    return np.average(vectors, axis=0)


def calculate_relation_vector(tokenizer, model, sentence, items):
    """Calculates the relation vector for the verb and the object."""
    if len(sentence.split(" ")) <= 512:
        item1_vector = get_averaged_vector(tokenizer, model, sentence, items[0])
        item2_vector = get_averaged_vector(tokenizer, model, sentence, items[1])
        if item1_vector is not None and item2_vector is not None:
            return np.concatenate((item1_vector, item2_vector), axis=1)


tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModel.from_pretrained("xlm-roberta-base")
nlp = spacy.load("en_core_web_trf")

#target_nouns = ["dinner","breakfast","feast","meal","buffet","speech","response","submission","lecture","conversation","brochure","textbook","novel","diary","summary","boat","classroom","cave","apartment","tower"]  # Change the noun when necessary
#use code below for smalle test
#target_nouns=["dinner", "breakfast"]
type1 = "Artifact"  # Change the type according to the noun
type2 = "Event"  # Change the type according to the noun
type3 = "Human"
type4 = "Information"
type5 = "Location"
type_adj1 = "Artifact_adj"
type_adj2 = "Event_adj"
datas = []

target_nouns=[]
folder_path = 'sentences'  
files = os.listdir(folder_path)
for file in files:
    if os.path.isfile(os.path.join(folder_path, file)):
        file_name = os.path.splitext(file)[0]
        target_nouns.append(file_name)




with open(f"classifiers/{type1}_clf.pickle", "rb") as classifier_file1:
    type1_clf = pickle.load(classifier_file1)
with open(f"classifiers/{type2}_clf.pickle", "rb") as classifier_file2:
    type2_clf = pickle.load(classifier_file2)
with open(f"classifiers/{type3}_clf.pickle", "rb") as classifier_file3:
    type3_clf = pickle.load(classifier_file3)
with open(f"classifiers/{type4}_clf.pickle", "rb") as classifier_file4:
    type4_clf = pickle.load(classifier_file4)
with open(f"classifiers/{type5}_clf.pickle", "rb") as classifier_file5:
    type5_clf = pickle.load(classifier_file5)
with open(f"classifiers/{type_adj1}_clf.pickle", "rb") as classifier_file5:
    type_adj1_clf = pickle.load(classifier_file5)
with open(f"classifiers/{type_adj2}_clf.pickle", "rb") as classifier_file5:
    type_adj2_clf = pickle.load(classifier_file5)


#data = [
#    "I ate the funny dinner.",
#    "She threws the dinner.",
#    "I attended a delicious dinner.",
#]  # an Example, delete the line to run with the real data

#results = {
#    type1: {"sentences": [], "items": []},
#    type2: {"sentences": [], "items": []},
#}  # This can change

results = {
    type1: {},
    type2: {},
    type3: {},
    type4: {},
    type5: {},
    type_adj1 : {},
    type_adj2 : {},

}  # I change it so that sentences and paars will become a dict where key=sentence and value = paar because I don't know how to paaring
#both sentences and (verb,noun) from 2 different dicts without indexing it which take extra time
num_of_nouns = len(datas)
pattern = r"``|''|\""

import csv
start = timer()
def classify_noun(target_nouns):
    for target_noun in target_nouns:
        sentenceID=0
        token_index=0
        token_span=0
        with open(f"sentences/{target_noun}.pickle", "rb") as sentence_file:
            datas = pickle.load(sentence_file)
        
        with open(f'copred_sentences/copred_bookcorpus_{target_noun}.tsv', 'wt', newline="") as out_file:
            heads = """#FORMAT=WebAnno TSV 3.3
#T_SP=custom.Span|label
#T_RL=custom.Relation|label|BT_custom.Span
            """
            out_file.write(heads+"\n\n")

            print("current noun is "+target_noun)
            for sen in datas:
                sentence = ' '.join(re.sub(pattern,'',sen).replace('-',' ').split())
            #for sentence in ["he buys you drinks , buys you dinner .", "would you like to have dinner with me ?", "he came to the house once to bring me dinner and check on me when i was on bed rest , and then he took me to the opera .","she had seemed a little off the night before when they had dinner together in the cafeteria ."]:
                tokenID=1
                # Parse the sentence
                parse_sentence = nlp(sentence)
                verbobj_pair = get_verbobj(parse_sentence, target_noun)  # Output: tuple: ("ate", "dinner")  

                adjnoun_pair= get_adjobj(parse_sentence, target_noun)
                noun =""
                verb=""
                adj=""
                type_and_paar = ()
                if verbobj_pair:  # If the noun is the direct object of a verb in the sentence
                    # Calculate the Relation Vector
                    
                    relation_vector = calculate_relation_vector(
                        tokenizer, model, sentence, verbobj_pair
                    )
                    if relation_vector is not None:
                        # Prediction
                        prediction_type1 = type1_clf.predict(
                            np.asarray(relation_vector, "float64").reshape(1, -1)
                        )  # Output: [1]
                        prediction_type2 = type2_clf.predict(
                            np.asarray(relation_vector, "float64").reshape(1, -1)
                        )  # Output: [0]
                        prediction_type3 = type3_clf.predict(
                            np.asarray(relation_vector, "float64").reshape(1, -1)
                        )
                        prediction_type4 = type4_clf.predict(
                            np.asarray(relation_vector, "float64").reshape(1, -1)
                        )
                        prediction_type5 = type5_clf.predict(
                            np.asarray(relation_vector, "float64").reshape(1, -1)
                        )
                        prediction_type_adj1 = type_adj1_clf.predict(
                            np.asarray(relation_vector, "float64").reshape(1, -1)
                        )
                        prediction_type_adj2 = type_adj2_clf.predict(
                            np.asarray(relation_vector, "float64").reshape(1, -1)
                        )
                        # Store the sentence and the verb-object pair of each sentence
                        if prediction_type1[0] == 1:
                            results[type1][sentence]=verbobj_pair
                            type_and_paar = (type1,verbobj_pair) #save (type, paar) for backtrackign type
                        if prediction_type2[0] == 1:
                            results[type2][sentence]=verbobj_pair
                            type_and_paar = (type2,verbobj_pair)
                        if prediction_type3[0] == 1:
                            results[type3][sentence]=verbobj_pair
                            type_and_paar = (type3,verbobj_pair)
                        if prediction_type4[0] == 1:
                            results[type4][sentence]=verbobj_pair
                            type_and_paar = (type4,verbobj_pair)
                        if prediction_type5[0] == 1:
                            results[type5][sentence]=verbobj_pair
                            type_and_paar = (type5,verbobj_pair)
                    verb,noun = verbobj_pair
                    type_and_paarAdj = ()
                    
                    if len(type_and_paar)>0:
                        sentenceID+=1
                        if sentenceID>1:
                            out_file.write("\n\n")
                            if sentenceID>19999:
                                sentenceID=1
                                token_index=0
                                token_span=0
                                tokenID=1
                         #this will cause output too have 2 more line between head and sentence so after output  just go the file and delete those 2 empyty lines
                        out_file.write(f"#Text={sentence.lstrip()}")
                        if adjnoun_pair:  # If the adj_noun pair exuist
                            # Calculate the Relation Vector
                            relation_vector = calculate_relation_vector(
                                tokenizer, model, sentence, adjnoun_pair
                            )
                            if relation_vector is not None:
                                prediction_type_adj1 = type_adj1_clf.predict(
                                    np.asarray(relation_vector, "float64").reshape(1, -1)
                                )
                                prediction_type_adj2 = type_adj2_clf.predict(
                                    np.asarray(relation_vector, "float64").reshape(1, -1)
                                )
                                # Store the sentence and the adj-object pair of each sentence
                                if prediction_type_adj1[0] == 1:
                                    results[type_adj1][sentence]=adjnoun_pair
                                    type_and_paarAdj = (type_adj1,adjnoun_pair)
                                if prediction_type_adj2[0] == 1:
                                    results[type_adj2][sentence]=adjnoun_pair
                                    type_and_paarAdj = (type_adj2,adjnoun_pair)
                            noun,adj=adjnoun_pair
                                                
                        
                        for token in parse_sentence:  #output 
                            token_index+=token_span
                            #print(token.idx)
                            token_span = len(token)
                            head="_"
                            clfType = "_"
                            clfTypeNoun="_"
                            clfTypeAdj="_"
                            token_pos ="_"
                            if len(type_and_paar) > 0:
                                if token.text == verb and noun in [t.text for t in token.children]: #only relevant verb with child = noun  
                                    #print(token.text, noun)
                                    for child in token.children:
                                        #dependency_colum="_"
                                        #print(tokenID,token.text+"-"+child.text+"-"+str(child.i+1))
                                        if (token.text,child.text) == type_and_paar[1]:
                                            #print("When paar",tokenID,token.text+"-"+child.text+"-"+str(child.i+1))
                                            clfTypeNoun=type_and_paar[0]
                                            token_pos = token.pos_ 
                                            child_id = child.i+1
                                            #dependency_colum= str(sentenceID)+"-"+str(child.i+1) 
                                    #print("print(dependency_colum) ",dependency_colum)
                                    out_file.write("\n")
                                    out_file.write('\t'.join([str(sentenceID)+"-"+str(tokenID),str(token_index)+"-"+str(token_index+token_span), token.text, token_pos,clfTypeNoun,str(sentenceID)+"-"+str(child_id) ]))
                                        #print("When not paar",tokenID,token.text+"-"+child.text+"-"+str(child.i+1))
                                    

            
                                elif token.text == noun and token.head.text == verb:
                                    token_pos = token.pos_
                                    head = token.head.text
                                    out_file.write("\n")
                                    out_file.write('\t'.join([str(sentenceID)+"-"+str(tokenID),str(token_index)+"-"+str(token_index+token_span), token.text, token_pos,clfType,"_"]))
                                elif token.text == adj and token.head.text == noun and len(type_and_paarAdj)>0 and token.head.head.text == verb:
                                    token_pos = token.pos_
                                    head = token.head.text
                                    if (head,token.text) == type_and_paarAdj[1]:
                                        clfTypeAdj=type_and_paarAdj[0]
                                    out_file.write("\n")
                                    out_file.write('\t'.join([str(sentenceID)+"-"+str(tokenID),str(token_index)+"-"+str(token_index+token_span), token.text, token_pos,clfTypeAdj,str(sentenceID)+"-"+str(token.head.i+1)]))
                                else:
                                    out_file.write("\n")
                                    out_file.write('\t'.join([str(sentenceID)+"-"+str(tokenID),str(token_index)+"-"+str(token_index+token_span), token.text, token_pos,clfType,"_"]))
                            token_index+=1
                            tokenID +=1  
            print(target_noun+" is finish")

classify_noun(target_nouns)
print("Time: ", timer()-start)
