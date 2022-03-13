# a program that can tag an arbitrary piece of text using the model
# and display the output to the user
import predictions
from nltk.tokenize import word_tokenize

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

# needs to run once
#import nltk
#nltk.download('punkt')

text = input('Please type a literary sentence you would like tagged: ')


# tokenize the input
tokenized = word_tokenize(text)

if args.debug:
    print("tokenized=",tokenized)

# create the feature df
ents = predictions.demo_tag_ents(text, tokenized)

# annotate the original text to include the tagged entities
# initialize original
annotated_tokens = [t for t in tokenized]
for e in ents:
    start_idx = e[2][0]
    end_idx = e[2][1] - 1
    #print("idx=",idx)
    to_annotate = annotated_tokens[start_idx:end_idx]
    if start_idx == end_idx:
        to_annotate = annotated_tokens[end_idx]
    if args.debug:
        print("e=",e)
        #print("to_annotate=",to_annotate)
    # add the annotations at appropriate indices
    #annotated_tokens[idx] = to_annotate + '/' + e[1]
    annotated_tokens[start_idx] = '{' + annotated_tokens[start_idx]
    annotated_tokens[end_idx] = annotated_tokens[end_idx] + '}/' + e[1]

################## 
# Format the Output
# #################
print()
print('--------------------------------')
print('Original Input Text')
print('--------------------------------')
print()
print(text)
output = ' '.join(annotated_tokens)
print()
print('--------------------------------')
print('Tagged Version of the Input Text')
print('--------------------------------')
print()
print(output)
