# This script will be used to extract features for the training and development sets
# It will create the feature files as a csv
# Each row contains a word/token. Each column represents a feature or the gold label

##########
# Imports
# ########
import argparse

from collections import Counter, defaultdict

import pandas as pd

from nltk import pos_tag
# needed to run once
#nltk.download('averaged_perceptron_tagger')


#############


def _merge_sent(words):
    '''
    helper function to merge tokenized words back into a sentence
    - need a sequence of words to get part of speech tags

    input: list of strings and merge them together
    return: merged string representing a sentence
    '''
    # some words need to be combined e.g. john 's -> john's
   # idx_to_combine = [(i, i-1) for i in range(len(words)) if words[i][0] in ["'", ".", ",", ";"]]
   # print("idx_to_combine=",idx_to_combine)
   # for i, idx in enumerate(idx_to_combine):
   #     offset = i
   #     print("words=",words)
   #     words[idx[1] - offset] = words[idx[1] - offset] + words[idx[0] - offset] #     del words[idx[0] - offset]
   #     # update indicies for other idx_to_combine

    #sent = ' '.join(words)
    return sent


def get_pos(sent):
    '''
    tag each of the sentences in the book
    input: tokenized sentence
    output: POS tags for the tokens
    '''
    #print("pos_tag(sent)=",pos_tag(sent))
    return pos_tag(sent)


def build_feature_df(bookpath):
    '''
    input: bookpath- path to the data for a single book
    output: feature dataframe for each word in that book
    '''
    # Training DF
    with open(bookpath, 'r') as f:
        words = []
        # labels for 0th nesting level-will need to expand to the other levels eventually
        labels0 = []
        # list of lists. Each list is a 
        sentences = []
        sent = []

        for line in f:
            #print("line=",line)
            if line == '\n':
                # merge the words into a sentence
                #merged = _merge_sent(sent)
                # add to sentences list
                #sentences.append(merged)
                sentences.append(sent)
                # starting a new sentence
                sent = [] 
            if line != '\n':
                #word, l0, l1, l2, l3 = line.split()
                components = line.split()
                word = components[0]
                l0 = components[1]
                sent.append(word)
                words.append(word)
                labels0.append(l0)


    # get the part of speech tags for each sentence 
    pos_tags = []
    for s in sentences:
        new_tags = [tag[1] for tag in get_pos(s)]
        pos_tags += new_tags
        

    # POS-1 POS+1 WORD-1 WORD+1
    # currently these cross sentence boundaries, will have to fix 
    pos_prev = []
    pos_2prev = []
    pos_next = []
    pos_2next = []

    word_prev = []
    word_2prev = []
    word_next = []
    word_2next = []
    for i in range(len(pos_tags)):
        #if i == 5:
            #asd
        if i == 0:
            pos_prev.append('PHI')
            word_prev.append('PHI')
            word_2prev.append('PHI')
            pos_2prev.append('PHI')
        elif i == 1:
            word_2prev.append('PHI')
            pos_2prev.append('PHI')
            word_prev.append(words[i - 1])
            pos_prev.append(pos_tags[i - 1])

        elif i == len(pos_tags) - 2:
            word_2next.append('OMEGA')
            pos_2next.append('OMEGA')
            word_next.append(words[i + 1])
            pos_next.append(pos_tags[i + 1])

        elif i == len(pos_tags) - 1:
            pos_next.append('OMEGA')
            word_next.append('OMEGA')
            word_2next.append('OMEGA')
            pos_2next.append('OMEGA')
        else:
            pos_2prev.append(pos_tags[i - 2])
            pos_prev.append(pos_tags[i - 1])
            pos_next.append(pos_tags[i - 1])
            pos_2next.append(pos_tags[i])

            word_2prev.append(words[i - 2])
            word_prev.append(words[i - 1])
            word_next.append(words[i - 1])
            word_2next.append(words[i])

       # print("i=",i)
       # print("word_2prev=",word_2prev)
       # print("word_prev=",word_prev)
       # print("word_next=",word_next)
       # print("word_2next=",word_2next)


    #print("len(pos_tags)=",len(pos_tags))
    #print("len(words)=",len(words))


# Create a pandas df - each row is a word and columns are features/label
    cols = ['WORD', 'POS', 'WORD-2', 'WORD-1','WORD+1', 'WORD+2', 'POS-2', 'POS-1','POS+1','POS+2', 'LABEL']
    feature_df = pd.DataFrame(list(zip(words, pos_tags,word_2prev, word_prev, word_next, word_2next, pos_2prev, pos_prev, pos_next, pos_2next, labels0)), columns=cols)

    return feature_df

def write_feature_csv(book_paths, filename):
    '''
    input: book_paths
           filename
           no_write: if false then we just return the concatenated dataframe
    output: features written to filename
    '''
    # build a feature df for each book in the set of training paths
    frames = []
    for path in book_paths:
        frame = build_feature_df(path)
        frames.append(frame)
    #concatenate all of the training frames together into single df 
    # currently keeping the separate index for each dataframe - this shouldn't affect ml.py
    # since we drop the index anyway. If need to have 1 index then set ignore_index=True in pd.concat()
    df = pd.concat(frames, ignore_index=True)
    #print("train_df=",train_df)
        
    #if no_write:
        #return df
    #else:
    # Convert DataFrame into a CSV
    df.to_csv(filename, index=False)
    print(f'a feature csv has been written to {filename}')


# ###################
# Feature Extraction
# ###################


if __name__ == '__main__':
    # only use argparse when this program is run directly
    parser = argparse.ArgumentParser()
    # path to the booklist to be tested against
    parser.add_argument('--train')
    parser.add_argument('--dev')


    args = parser.parse_args()
    # get the list of book paths for the training set
    with open(args.train, 'r') as f:
        train_paths = [line.split()[0] for line in f if line != '\n']
        write_feature_csv(train_paths, 'features/train_features.csv')

    if args.dev is not None:
        # get the list of book paths for the test set
        with open(args.dev, 'r') as f:
            dev_paths = [line.split()[0] for line in f if line != '\n']
            write_feature_csv(dev_paths, 'features/dev_features.csv')

    # create the train and dev feature csv files
    #write_feature_csv(train_paths, 'features/train_features.csv')
    #write_feature_csv(dev_paths, 'features/dev_features.csv')

