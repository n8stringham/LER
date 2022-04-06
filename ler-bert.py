# This is the demo script for LER-bert
from transformers import pipeline
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
import torch


def annotate_str(start, end, tag, text):
    '''
    return a new string with an entity annotation made
    '''
    return text[:start] + '[' + text[start:end] + f']\\{tag}' + text[end:]



# Model to make predictions with
model_checkpoint = "models/bert-finetuned-ner/checkpoint-2625"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# tokenize the input
token_classifier = pipeline(
    "token-classification", model=model_checkpoint, tokenizer=tokenizer, aggregation_strategy="simple"
)

query = True
while query:
    text = input('Please type a literary sentence you would like tagged: ')
    result = token_classifier(text)
#print("result=",result)
    ents = []
    new_str = text
    offset = 0
    for r in result:
        start = r['start'] + offset
        end = r['end'] + offset
        tag = r['entity_group']
        new_str = annotate_str(start, end, tag, new_str)
        offset += 6


################## 
# Format the Output
# #################
    print()
    print('--------------------------------')
    print('Original Input Text')
    print('--------------------------------')
    print()
    print(text)
    print()
    print('--------------------------------')
    print('Tagged Version of the Input Text')
    print('--------------------------------')
    print()
    print(new_str)
    print('--------------------------------')
    print('Next Query or Exit')
    print('--------------------------------')
    nxt = input('Next query? y or n: ')
    if nxt != 'y':
        query = False

