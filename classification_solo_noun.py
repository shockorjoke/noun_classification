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

#target_nouns = ["apartment","meal","dinner","breakfast","response","conversation","boat"]  
target_nouns =["dinner","breakfast","feast","meal","buffet","speech","response","submission","lecture","conversation","brochure","textbook","novel","diary","summary","boat","classroom","cave","apartment","tower"]


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

#Optional filter to make the corpus fit the site reader. 
def replace_symbol(input_sentence):
    patterns = {
        r"``|''|\"|_|(?<!\w)\*(?!\w)" : r"",
        r"-" : r" ",
        r".-" : r" ",
        r"(?<=\w)/(?=\w)": r" , ",
        r"(['\"])(?=\w)" : r" \1 ",
        r" ' m\b": r" am",
        r" ' ve\b": r" have",
        r" ' ll\b" : r" will",
        r"\bwont\b" : r"will not",
        r"\bdont\b" : r"do not",
        r"\bdoesnt\b" : r"does not",
        r"\bdidnt\b" : r"did not",
        r"\bisnt\b" : r"is not",
        r"\bwasnt\b" : r"was not",
        r"\bwerent\b" : r"were not",
        r"\bhavent\b" : r"have not",
        r"\bhaven\b" : r"have not",
        r"\bhasnt\b" : r"has not",
        r"\bhadnt\b" : r"had not",
        r"\barent\b" : r"are not",
        r"\bwouldnt\b" : r"would not",
        r"\bshouldnt\b" : r"should not",
        r"\bcouldnt\b" : r"could not",
        r"\bwouldve\b" : r"would have",
        r"\bcouldve\b" : r"could have",
        r"\bshouldve\b" : r"should have",
        r"\bwillve\b" : r"will have\b",
        r"\bdontve\b" : r"do not have\b",
        r"\bdoesntve\b" : r"does not have",
        r"\bdidntve\b" : r"did not have",
        r"\byoull\b" : r"you will",
        r"\btheyll\b" : r"they will",
        r" ' d\b": r" would",
        r" ' s\b": r" 's",
        r" ' re\b": r" are",
        r" n ' t\b": r" n't",
        r"\bhed\b": r"he would",
        r"\bwhod\b": r"who would",
        r"\bshed\b": r"she would",
        r"\bwed\b": r"we would",
        r"\bim\b": r"i am",
        r"\bitd\b": r"it would",
        r"\bid\b": r"i would",
        r"\bive\b": r"i have",
        r"\btheyd\b": r"they would",
        r"\bweve\b": r"we have",
        r"\bwhove\b": r"who have",
        r"\btheyve\b": r"they have",
        r"\btheyre\b" : r"they are",
        r"\byoure\b" : r"you are",
        r"\bhes\b" : r"he is",
        r"\bshes\b" : r"she is",
        r"\bmr\.": r"mr",
        r"\bms\.": r"ms",
        r"\bst\.": r"st",
        r"\bmrs\.": r"mrs",
        r"\bdr\.": r"dr",
        r"\bprof\.": r"prof",
        r"\btearin '\b":r"tearing",
        r"\bthats\b":r"that is",
        r"\btheres\b":r"there is",
        r"\b'hey , '": r"hey , ",
        r"\bcant\b": r"can not",
        r"'could ": r"' could ",
        r"'please ": r"' please ",
        r"'well ": r"' well ",
        r"'it ": r"' it ",
        r" r ": r" are ",
        r"'i ": r"' i ",
        r"'dear ": r"' dear ",
        r"'oh ": r"' oh ",
        r"'the ": r"' the ",
        r"'but ": r"' but ",
        r"'can ": r"' can ",
        r"\bits\b": r"it is",
        r"([.,;:!?-])": r" \1 ",
        r" p m ": r" pm ",
        r" ca n't ": r" can not ",
        
    }
    new_sentence = input_sentence
    for patt, replacement in patterns.items():
        new_sentence = re.sub(patt,replacement,new_sentence)
    return new_sentence

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
            #for sen in datas[5808:5810]:
            for sen in datas:    
                #print("SEN: ", sen, "\n")
                #pre_sentence = ' '.join(re.sub(pattern,'',sen).replace('-',' ').split())
                pre_sentence=replace_symbol(sen)
                sentence = " ".join(pre_sentence.split())
                #print("SENTENCE_TEST: ",sentnee_test, "\n")
                #print("SENTENCE: ",sentence, "\n")
                with open(f"compare.txt", "a") as sentence_file:
                    sentence_file.write("SENTENCE_TEST: " +sentence)
                    sentence_file.write("\n")
                    sentence_file.write("SENTENCE: "  +pre_sentence)
                    sentence_file.write("\n")
            #for sentence in ["he buys you drinks , buys you dinner .", "would you like to have dinner with me ?", "he came to the house once to bring me dinner and check on me when i was on bed rest , and then he took me to the opera .","she had seemed a little off the night before when they had dinner together in the cafeteria ."]:
                tokenID=1
                # Parse the sentence
                parse_sentence = nlp(sentence)
                #print(sentenceID,parse_sentence, "\n")
                verbobj_pair = get_verbobj(parse_sentence, target_noun)  # Output: tuple: ("ate", "dinner")  

                adjnoun_pair= get_adjobj(parse_sentence, target_noun)
                noun =""
                verb=""
                adj=""
                type_and_paar = {}
                if verbobj_pair:  # If the noun is the direct object of a verb in the sentence
                    # Calculate the Relation Vector
                    type_and_paar[verbobj_pair]=[]
                    
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
                            type_and_paar[verbobj_pair].append(type1) #save (paar, [typed]) for backtrackign type
                        if prediction_type2[0] == 1:
                            results[type2][sentence]=verbobj_pair
                            type_and_paar[verbobj_pair].append(type2)
                        if prediction_type3[0] == 1:
                            results[type3][sentence]=verbobj_pair
                            type_and_paar[verbobj_pair].append(type3)
                        if prediction_type4[0] == 1:
                            results[type4][sentence]=verbobj_pair
                            type_and_paar[verbobj_pair].append(type4)
                        if prediction_type5[0] == 1:
                            results[type5][sentence]=verbobj_pair
                            type_and_paar[verbobj_pair].append(type5)
                    verb,noun = verbobj_pair
                    #print("TYOEAND OAAR",type_and_paar[verbobj_pair])
                    #print(type_and_paar)
                    type_and_paarAdj = {}
                    
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
                        #print(sentence)
                        if adjnoun_pair:  # If the adj_noun pair exuist
                            type_and_paarAdj[adjnoun_pair]=[]
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
                                    type_and_paarAdj[adjnoun_pair].append(type_adj1)
                                if prediction_type_adj2[0] == 1:
                                    results[type_adj2][sentence]=adjnoun_pair
                                    type_and_paarAdj[adjnoun_pair].append(type_adj2)
                            noun,adj=adjnoun_pair
                                                
                        
                        for token in parse_sentence:  #output 
                            #print(token)
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
                                        if (token.text,child.text) == list(type_and_paar.keys())[0]:
                                            clfTypeNoun="_".join(list(type_and_paar.values())[0])
                                            token_pos = token.pos_ 
                                            child_id = child.i+1

                                    out_file.write("\n")
                                    out_file.write('\t'.join([str(sentenceID)+"-"+str(tokenID),str(token_index)+"-"+str(token_index+token_span), token.text, token_pos,clfTypeNoun,str(sentenceID)+"-"+str(child_id) ]))                                    
                                elif token.text == noun and token.head.text == verb:
                                    token_pos = token.pos_
                                    head = token.head.text
                                    out_file.write("\n")
                                    out_file.write('\t'.join([str(sentenceID)+"-"+str(tokenID),str(token_index)+"-"+str(token_index+token_span), token.text, token_pos,clfType,"_"]))
                                elif token.text == adj and token.head.text == noun and len(type_and_paarAdj)>0 and token.head.head.text == verb:
                                    token_pos = token.pos_
                                    head = token.head.text
                                    if (head,token.text) == list(type_and_paarAdj.keys())[0]:
                                        clfTypeAdj="_".join(list(type_and_paarAdj.values())[0])
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

