# use the model for inference

import pickle
import features

#import argparse

import pandas as pd
from sklearn.feature_extraction import DictVectorizer

from nltk import pos_tag
# needed to run once
#nltk.download('averaged_perceptron_tagger')

#parser = argparse.ArgumentParser()
# path to the booklist to be tested against
#parser.add_argument('data')


#args = parser.parse_args()

def vectorize_data(data, dict_vectorizer):
    '''
    returns: vectorized data according to pre-trained dict_vectorizer
    '''
    # Initialize loaded vectorizer
    v = dict_vectorizer
    # We fit_transform in featurized train data (first transform the pandas dataframe into a list of dictionaries key = column name, value = entry in row)
    X = v.transform(data.to_dict('records'))
    # return vectorized train and test data
    return X


def make_preds(df, dict_vectorizer, model):
    '''
    inputs: feature dataframe
    outputs: predictions
    '''
    # separating the label column from the data
    #features=['WORD', 'POS']
    X = df.drop('LABEL', axis=1)
    #X = X[features]
    # getting the labels of the data 
    y = df.LABEL.values

    # the vectorized data 
    vectorized_data = vectorize_data(X, dict_vectorizer)
    #print("vectorized_data.shape=",vectorized_data.shape)

    preds = model.predict(vectorized_data)
    #print("preds=",preds)
    #print("len(preds)=",len(preds))
    return preds


def extract_ents(preds, df):
    '''
    input: predictions, feature df for a single book
    output: entity list - tuple of (entity, tag, span)
    '''
    #Apply the predictions and merge to extract entities
    words = df['WORD'].tolist()

    #print("preds=",preds)
    #print("len(words)=",len(words))

    ents = []
    is_entity = False
    for i, (word, label) in enumerate(zip(words, preds)):
        # if we find a B or I label then we want to extract that as an entity
        if label[0] in ['B', 'I']:
            #len(tab_separated[i]) > 1 and 
            is_entity = True
            counter = i
            start = i
            # checking the next entities
            while is_entity:
                ent_type = preds[counter][2:]
                #print("ent_type=",ent_type)
                #print("preds[counter]=",preds[counter])
                counter += 1

                # we've reached the end of the words so that is automatically
                # the end of the entity span
                if counter == len(words):
                    is_entity = False
                    entity_str = ' '.join(words[start:counter])
                    #print("entity_str=",entity_str)

                    #first tag is the tag for the whole string - remove (B- or I-) - this has 
                    # effect of treating B and I labels as the same which is fine for now
                    entity_tag = preds[start][2:]
                    #print("entity_tag=",entity_tag)
                    # which tokens does this entity span? [start, end)
                    entity_span = (start, counter)
                    entity_tuple = (entity_str, entity_tag, entity_span) 
                    #print("entity_tuple=",entity_tuple)

                    # collect all gold entities
                    ents.append(entity_tuple)

                # we've found a new B tag or a new O so the current entity is finished
                elif preds[counter][2:] != ent_type or preds[counter] == 'O':
                #elif preds[counter] == 'B' or preds[counter] == 'O':
                    is_entity = False
                    #print("start=",start)
                    #print("counter=",counter)
                    entity_str = ' '.join(words[start:counter])
                    #print("entity_str=",entity_str)

                    #first tag is the tag for the whole string - remove (B- or I-) - this has 
                    # effect of treating B and I labels as the same which is fine for now
                    entity_tag = preds[start][2:]
                    #print("entity_tag=",entity_tag)
                    # which tokens does this entity span? [start, end)
                    entity_span = (start, counter)
                    entity_tuple = (entity_str, entity_tag, entity_span) 
                    #print("entity_tuple=",entity_tuple)

                    # collect all gold entities
                    ents.append(entity_tuple)
    return ents


#if __name__ == '__main__':
def get_pred_ents(booklist):
    '''
    input: the booklist you want to get entities for
    returns: dictionary of entities extracted for each book
    '''

    # storing the predictions for each book
    preds_dict = {}

    # load up the trained classifier
    #filename = 'logistic_regression_model.joblib'
    filename = 'logistic_regression_model_all.joblib'
    with open(filename, 'rb') as f:
        model = pickle.load(f)

    # load up the vectorizer
    #dict_file = 'DictVectorizer.joblib'
    dict_file = 'DictVectorizer_all.joblib'
    with open(dict_file, 'rb') as f:
        dict_vectorizer = pickle.load(f)

    # read in the data paths
    #with open(args.data, 'r') as f:
    #with open(booklist, 'r') as f:
        #paths = [line.split()[0] for line in f if line != '\n']
    paths = booklist

    # create the train and dev feature csv files
    num_feature_vecs = 0
    for bookpath in paths:
        book_id = bookpath.split('/')[2].split('_')[0]
        #print("book_id=",book_id)

        df = features.build_feature_df(bookpath)
        num_feature_vecs += df.shape[0]
        # prepare df to be vectorized
        assert (df.isnull().values.any() == False), 'There are null entries in the data'
        # assert that given features are columns in the csv
        #assert (set(features).issubset(set(df.columns.tolist()))), 'Some of the features selected are not columns in the input csv file'

        # make_preds
        preds = make_preds(df, dict_vectorizer, model)
        #print("preds=",preds)

        # extract ents
        ents = extract_ents(preds, df)
        #print("ents=",ents)
        #print("len(ents)=",len(ents))

        # update ents dict for the current book
        preds_dict.update({book_id : ents})        
    #print("preds_dict=",preds_dict)
    #print("preds_dict.keys()=",preds_dict.keys())
    print("num_feature_vecs=",num_feature_vecs)

    return preds_dict


def demo_tag_ents(user_input, tokens):
    '''
    This function should only be used for the demo
    input: the tokenized input from a user 
    returns: the entities that were tagged 
    '''
    # load up the trained classifier
    filename = 'logistic_regression_model_all.joblib'
    with open(filename, 'rb') as f:
        model = pickle.load(f)

    # load up the vectorizer
    dict_file = 'DictVectorizer_all.joblib'
    with open(dict_file, 'rb') as f:
        dict_vectorizer = pickle.load(f)

    # create the train and dev feature csv files
    df = demo_feature_df(user_input, tokens)
    #print("df=",df)

    # prepare df to be vectorized
    assert (df.isnull().values.any() == False), 'There are null entries in the data'
    # assert that given features are columns in the csv
    #assert (set(features).issubset(set(df.columns.tolist()))), 'Some of the features selected are not columns in the input csv file'

    # make_preds
    preds = make_preds(df, dict_vectorizer, model)
    #print("preds=",preds)

    # extract ents
    ents = extract_ents(preds, df)
    #print("ents=",ents)
    #print("len(ents)=",len(ents))

    return ents 


def demo_feature_df(user_input, tokens):
    '''
    build feature df for user provided input - function only used with demo
    output: feature dataframe for each word in user_input
    '''
    # assume that user passes in complete sentences that end in punct
    words = tokens
    labels0 = ['NA' for t in tokens]
    pos_tags = [tag[1] for tag in features.get_pos(tokens)]

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

    #print("len(pos_tags)=",len(pos_tags))
    #print("len(words)=",len(words))


# Create a pandas df - each row is a word and columns are features/label
    #cols = ['WORD', 'POS', 'LABEL']
    #feature_df = pd.DataFrame(list(zip(words, pos_tags, labels0)), columns=cols)
    cols = ['WORD', 'POS', 'WORD-2', 'WORD-1','WORD+1', 'WORD+2', 'POS-2', 'POS-1','POS+1','POS+2', 'LABEL']
    feature_df = pd.DataFrame(list(zip(words, pos_tags,word_2prev, word_prev, word_next, word_2next, pos_2prev, pos_prev, pos_next, pos_2next, labels0)), columns=cols)

    return feature_df
