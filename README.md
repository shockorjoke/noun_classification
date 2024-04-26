# noun_classification
Prepare data for cocorpus, (verb,noun) and (adj,noun) pairs are classified with "xlm-roberta-base" models https://huggingface.co/FacebookAI/xlm-roberta-base

raw data come from "bookcorpus" https://huggingface.co/datasets/bookcorpus

### requirement
pip install -r requirement.txt

### data_extraction.py
extract directed object target nouns from corpus and save in sentences folder

### classification.py
annotating noun, its verb and adj with classifier and parsing into copred_sentences
