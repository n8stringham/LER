# inference with fine_tuned Bert 

from transformers import pipeline
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
import torch

import re
import pickle

import argparse

from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('--predict', action='store_true')
parser.add_argument('--score', action='store_true')

args = parser.parse_args()

def get_tokens_and_tags(books):
    '''
    input: list of paths to books
    output: [...[tokens for sentence i]...], [...[tags for sentence i]...]
    '''
    #sentence_tokens = []
    #ner_tags = {'0' : [], '1': [], '2': [], '3': []} 
    # need the labels found in the dataset
    book_dict = {}
    labels = set()
    for  b in books:
        sentence_tokens = []
        ner_tags = {'0' : [], '1': [], '2': [], '3': []} 
        book_id = b.split('/')[2].split('_')[0]
        #print("book_id=",book_id)
        #tab_separated = []
        with open(b, 'r') as f:
            tokens = []
            tags0 = []
            tags1 = []
            tags2 = []
            tags3 = []
            for line in f:
                tab_separated = line.split('\t')
                if line != '\n':
                    tokens.append(tab_separated[0])
                    tags0.append(tab_separated[1])
                    tags1.append(tab_separated[2])
                    tags2.append(tab_separated[3])
                    tags3.append(tab_separated[4])
                    labels.add(tab_separated[1])
                    
                # end of sentence
                else:
                    sentence_tokens.append(tokens)
                    ner_tags['0'].append(tags0)
                    ner_tags['1'].append(tags1)
                    ner_tags['2'].append(tags2)
                    ner_tags['3'].append(tags3)

                    # reset
                    tokens = []
                    tags0 = []
                    tags1 = []
                    tags2 = []
                    tags3 = []
        # add sentence tokens and ner_tags to dict
        book_dict[book_id] = (sentence_tokens, ner_tags)

    #return sentence_tokens, ner_tags, labels
    return book_dict 

def add_space(s, idx, not_s=False):
    '''
    add a space at specified location in a string
    '''
    #return s[:idx] + ' ' + s[idx:]
    if not_s:
        return s[:idx] + ' ' + s[idx] + ' ' + s[idx+1:]
    else:
        return s[:idx] + ' ' + s[idx:]



#def get_token_index(sent, results, token_count):
#    '''
#    by defaul BERT outputs the character span of the entities
#    Our scoring script uses the word positions to calculate
#    metrics. This takes in the character indices and returns word
#    indices relative to the sentence
#    '''
#    # revert sent string to list of tokens by splitting on whitespace
#    sent_tokens = sent.split()
#    print("sent_tokens  =",sent_tokens  )
#    # if we have duplicate words we need to find the index of
#    # the correct occurrence. only search indices after last occurence
#    search_start = None
#    # keep track of latest indx of seen entities
#    seen = {} 
#    for r in results:
#        print("r=",r)
#        # handle apostrophes that should be split by adding a space
#        # to the string in r['word']
#        apostrophe = re.finditer("'s",r['word']) 
#        a_idxs = [m.span()[0] for m in apostrophe]
#        print("a_idxs=",a_idxs)
#        if a_idxs:
#            for i in a_idxs:
#                r['word'] = add_space(r['word'], i)
#        apostrophe_not_s = re.finditer("'(?!s)", r['word'])
#        not_s_idxs = [m.span()[0] for m in apostrophe_not_s]
#        # non-s apostrophes
#        if not_s_idxs:
#            for i in not_s_idxs:
#                r['word'] = add_space(r['word'], i, not_s=True)
#
#        # handle commas - separate them from tokens
#        comma = re.finditer(',', r['word'])
#        c_idxs = [m.span()[0] for m in comma]
#        if c_idxs:
#            for i, c in enumerate(c_idxs):
#                r['word'] = add_space(r['word'], c + i)
#
#
#        # handle hypenated words - if we find it we need to join words
#        # together
#        if r['word'].find('-') != -1:
#            words  = r['word'].split()
#            hyphen = words.index('-')
#            # check if hyphen is last word
#            if len(words) > hyphen + 1:
#                new_word = words[hyphen - 1] + words[hyphen] + words[hyphen+1]
#                del words[hyphen-1:hyphen+2]
#            else:
#                new_word = words[hyphen - 1] + words[hyphen]
#                del words[hyphen-1:hyphen+1]
#            words.insert(hyphen -1, new_word)
#            print("words=",words)
#            r['word'] = ' '.join(words)
#
#            
#        
#        
#
#        # split up multi word ents
#        ent = r['word'].split()
#        span = [] 
#        for e in ent:
#            print("e=",e)
#            if e not in seen:
#                idx = sent_tokens.index(e)
#                print("idx=",idx)
#                span.append(idx)
#                seen[e] = idx
#            else:
#                idx = sent_tokens.index(e, seen[e] + 1)
#                print("idx=",idx)
#                span.append(idx)
#                
#        # update dict with token level span of the ent
#        if len(span) > 1:
#            r['token_span'] = (span[0] + token_count, span[-1] + token_count + 1)
#        else:
#            r['token_span'] = (span[0] + token_count, span[0] + token_count + 1)
#    #test = results
#    #print("test=",test)
#    return results
#
#def get_gold_ents():
#    '''
#    helpful to look at
#    '''
#    # For each book we need to combine BIO tags to get the full entities
#        tab_separated = [line.split('\t') for line in f if line != '\n']
#        #print("tab_separated=",tab_separated)
#        #print("len(tab_separated)=",len(tab_separated))
#        is_entity = False
#        entities = []
#        for i in range(len(tab_separated)):
#            # start keeping track if we find a 'B' token)
#            if tab_separated[i][level][0] not in ['O', 'I']:
#                #len(tab_separated[i]) > 1 and 
#                is_entity = True
#                counter = i
#                start = i
#                # checking the next entities
#                while is_entity:
#                    counter += 1
#                    # we've found a new B tag or a new O so the current entity is finished
#                    if tab_separated[counter][level][0] == 'B' or tab_separated[counter][level] == 'O':
#                        is_entity = False
#                        #print("start=",start)
#                        #print("counter=",counter)
#                        entity_str = ' '.join([s[0] for s in tab_separated[start:counter]])
#                        
#                        #first tag is the tag for the whole string - remove (B- or I-)
#                        entity_tag = tab_separated[start][level][2:]
#                        # which tokens does this entity span? [start, end)
#                        entity_span = (start, counter)
#                        entity_tuple = (entity_str, entity_tag, entity_span) 
#                        #print("entity_tuple=",entity_tuple)
#
#                        # collect all gold entities
#                        entities.append(entity_tuple)


def get_gold_char_spans(sent, tags):
    '''
    get the character level spans for each entity
    The other approach was not really working
    Here I need to just map each gold entity to its character index
    in the sentence. Then I can keep track of the character count
    to make the spans relative to the whole document.
    '''
    #print("sent=",sent)
    #print("tags=",tags)
    joined = ' '.join(sent)
    #print("joined=",joined)
    #print("len(joined)=",len(joined))
    is_entity = False
    entities = []
    seen = []
    joined_pos = 0
    #print("joined_pos=",joined_pos)
    # where we are in the string
    end_word_idx = 0
    for i, (word, tag) in enumerate(zip(sent, tags)):
        if i >= end_word_idx:
            # start keeping track if we find a 'B' token)
            if tag[0][0] not in ['O', 'I']:
                #len(tab_separated[i]) > 1 and 
                is_entity = True
                matches = [m.span() for m in re.finditer(word, joined)]
                # filter matches so that we only look at matches in front
                #print("joined_pos=",joined_pos)
                matches = [m for m in matches if m[0] >= joined_pos]
                #print("matches=",matches)
                char_start = matches[0][0]
                #print("char_start=",char_start)
                word_start = i
                counter = i 
                # update pos in sentence 
                joined_pos += len(word) + 1

                while is_entity:
                    # we've found a new B tag or a new O so the current entity is finished

                    # update index counter
                    counter += 1
                    if tags[counter][0] == 'B' or tags[counter][0] == 'O' or counter == len(sent) - 1:
                        is_entity = False
                        end_matches = [m.span() for m in  re.finditer(re.escape(sent[counter]), joined)]
                        #print("end_matches=",end_matches)
                        #print("joined_pos=",joined_pos)
                        end_matches = [m for m in end_matches if m[0] >= joined_pos]
                        #print("end_matches=",end_matches)

                        char_end = end_matches[0][0] - 1
                        #print("char_end=",char_end)
                        end_word_idx = counter
                        entity_str = ' '.join([s for s in sent[word_start:counter]])
                        #first tag is the tag for the whole string - remove (B- or I-)
                        entity_tag = tags[word_start][2:]
                        # which tokens does this entity span? [start, end)
                        entity_span = (char_start, char_end)
                        entity_tuple = (entity_str, entity_tag, entity_span) 
                        #print("entity_tuple=",entity_tuple)

                        # collect all gold entities
                        entities.append(entity_tuple)
                        #print("entities=",entities)
            else:
                # move marker to end of word + 1 for space
                joined_pos += len(word) + 1
        else:
            #print('I passed')
            pass
    return entities    

def make_preds(sent):
    '''
    get the predictions for a sentence in a book
    '''
    # combine the words in a sent with whitespace - now sents is a
    # list of strings where each string is a sentence
    #sent = [' '.join(sent) for sent in sents]
    sent = ' '.join(sent)
    #print("sents=",sents)

    # setup the token classifier pipeline
    # decoding strategy is "simple
    token_classifier = pipeline(
        "token-classification", model=model_checkpoint, tokenizer=tokenizer, aggregation_strategy="simple"
    )
    result = token_classifier(sent)
    print("result=",result)
    ents = []
    for r in result:
        entity_tup = (r['word'], r['entity_group'], (r['start'], r['end']))
        ents.append(entity_tup)
    return ents

def compute_metrics(gold, preds):
    '''
    Compute precision, recall, and f-score w.r.t. 6 entity types
        - PER - FAC - GPE - LOC - VEH - ORG
    '''
    ent_types_pred = Counter()
    ent_types_gold = Counter()

    # track correct by class
    ent_correct_pred = Counter()

    # keep track of incorrect answers
    missed = [] 

    # for each book count the total entity types in gold and in preds
    for book, ents in gold.items():
        #print("book=",book)
        # counts for each entity type in gold
        for ent in ents:
            ent_types_gold[ent[1]] += 1

            # count correct
            if ent in preds[book]:
                ent_correct_pred[ent[1]] += 1

            # track which examples we miss
            else:
                missed.append(ent)
            
        # count for each type in pred
        #print("preds[book]=",preds[book])
        for pred in preds[book]:
            ent_types_pred[pred[1]] += 1

    # record the total number of each type of entity in gold
    print("ent_types_gold.most_common()=",ent_types_gold.most_common())
    print("ent_types_gold.total()=",ent_types_gold.total())
    print("ent_types_pred.most_common()=",ent_types_pred.most_common())
    print("ent_types_pred.total()=",ent_types_pred.total())

    print("ent_correct_pred.most_common()=",ent_correct_pred.most_common())

    #print("missed=",missed)


    # Compute R,P,F for each class
    #classes = ent_types_gold.keys()
    classes = ['PER', 'FAC', 'GPE', 'LOC', 'VEH', 'ORG']
    class_dict = {}

    # total
    for c in classes:
        print("c=",c)
        if ent_types_gold[c] != 0:
            recall = ent_correct_pred[c] / ent_types_gold[c]
        else:
            recall = 0

        if ent_types_pred[c] != 0:
            precision = ent_correct_pred[c] / ent_types_pred[c]
        else:
            precision = 0

        if precision == 0 and recall == 0:
            f_score = 0
        else:
            f_score = 2*precision*recall / (precision + recall)

        # store in dictionary for macro averaging
        rpf_tuple = (recall, precision, f_score)
        class_dict[c] = rpf_tuple

        # format 2 decimal
        precision = f'{precision:.2f}'
        recall = f'{recall:.2f}'
        f_score = f'{f_score:.2f}'

        print("precision=",precision)
        print("recall=",recall)
        print("f_score=",f_score)

    # Micro-averaged scores
    rec_micro = ent_correct_pred.total() / ent_types_gold.total()
    prec_micro = ent_correct_pred.total() / ent_types_pred.total()
    if prec_micro + rec_micro != 0:
        f_micro = 2*prec_micro*rec_micro / (prec_micro + rec_micro)
    else:
        f_micro = 0

    # format 2 decimals
    prec_micro = f'{prec_micro:.2f}'
    rec_micro = f'{rec_micro:.2f}'
    f_micro = f'{f_micro:.2f}'

    print("prec_micro=",prec_micro)
    print("rec_micro=",rec_micro)
    print("f_micro=",f_micro)


    # Macro-averaged scores 
    rec_macro = 0
    prec_macro = 0
    f_macro = 0
    for c in classes:
        rec_macro += class_dict[c][0]
        prec_macro += class_dict[c][1]
        f_macro += class_dict[c][2]

    rec_macro = rec_macro / 6
    prec_macro = prec_macro / 6
    f_macro = f_macro / 6

    # format 2 decimals
    rec_macro = f'{rec_macro:.2f}'
    prec_macro = f'{prec_macro:.2f}'
    f_macro = f'{f_macro:.2f}'

    print("rec_macro=",rec_macro)
    print("prec_macro=",prec_macro)
    print("f_macro=",f_macro)




def macro_average(scores):
    '''
    macro-average the score over the classes
    arithmetic mean of the score for each class
    '''
    return sum(scores) / len(scores)

def micro_average(ent_correct_pred, ent_types_pred):
    '''
    micro-average the score over the classes
    sum of the correct answers from all classes divided by 
    '''

def f_score(r, p):
    '''
    harmonic mean of recall and precision
    '''
    return 2*r*p / (r + p)


if __name__ == '__main__':

    if args.predict:

        # Model to make predictions with
        model_checkpoint = "models/bert-finetuned-ner/checkpoint-2625"
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


        # get list of books to score
        with open('data/train/books.txt', 'r') as f:
            train_books = [b.strip() for b in f]

        with open('data/dev/books.txt', 'r') as f:
            dev_books = [b.strip() for b in f]

        with open('data/test/books.txt', 'r') as f:
            test_books = [b.strip() for b in f]

        datasets = [train_books, dev_books, test_books]
        filenames = ['train_books', 'dev_books', 'test_books']


        #train_book_dict = get_tokens_and_tags(train_books)
        #dev_book_dict = get_tokens_and_tags(dev_books)
        #test_book_dict = get_tokens_and_tags(test_books)
        
        for d, f in zip(datasets, filenames):
            # Extract the gold entity tuples for each dataset
            all_ents = {}
            all_preds = {}
            # set books, and book dict according to current dataset
            books = d
            book_dict = get_tokens_and_tags(d)
            gold_filename = f + '_ents.pkl'
            preds_filename = f + '_preds.pkl'
            #print("gold_filename=",gold_filename)
            #print("preds_filename=",preds_filename)
            for b in books:
                # extract book id to use as key
                book_id = b.split('/')[2].split('_')[0]
                #print("book_id=",book_id)
                book_ents = []
                preds = []
                sents = book_dict[book_id][0]
                #for l in ['0', '1', '2', '3']:
                tags0 = book_dict[book_id][1]['0']
                tags1 = book_dict[book_id][1]['1']
                tags2 = book_dict[book_id][1]['2']
                tags3 = book_dict[book_id][1]['3']
                #tags = tags1 + tags2 + tags3
                # loop over all sentences in a book
                for s, t, t1, t2, t3 in zip(sents, tags0, tags1, tags2, tags3):
                    #print("s=",s)
                    es0 = get_gold_char_spans(s, t)
                    es1 = get_gold_char_spans(s, t1)
                    es2 = get_gold_char_spans(s, t2)
                    es3 = get_gold_char_spans(s, t3)
                    # combine gold ents from all nesting levels
                    entities = es0 + es1 + es2 + es3
                    print("entities=",entities)
                    book_ents += entities

                    # make preds using model on the sentence
                    preds += make_preds(s)
                    #print("book_ents=",book_ents)
                    #print("preds=",preds)


                # update dictionaries 
                #print("len(book_ents)=",len(book_ents))
                all_ents[book_id] = book_ents
                all_preds[book_id] = preds
                #print("all_ents[book_id]=",all_ents[book_id])
                #print("all_preds[book_id]=",all_preds[book_id])
                #print("len(all_ents[book_id])=",len(all_ents[book_id]))
                #print("len(all_preds[book_id])=",len(all_preds[book_id]))
            #print("len(all_ents.keys())=",len(all_ents.keys()))

            print('Pickling' + gold_filename + ' and ' + preds_filename)
            # save predictions and gold for each data split
            with open('keys/'+gold_filename, 'wb') as p:
                pickle.dump(all_ents, p)

            with open('keys/'+preds_filename, 'wb') as p:
                pickle.dump(all_preds, p)

    if args.score:
        # load the gold and preds

        #datasets = [train_books, dev_books, test_books]
        
        # train split
        with open('keys/train_books_ents.pkl', 'rb') as p:
            train_gold = pickle.load(p)

        with open('keys/train_books_preds.pkl', 'rb') as p:
            train_preds = pickle.load(p)

        # dev split
        with open('keys/dev_books_ents.pkl', 'rb') as p:
            dev_gold = pickle.load(p)

        with open('keys/dev_books_preds.pkl', 'rb') as p:
            dev_preds = pickle.load(p)

        # test split
        with open('keys/test_books_ents.pkl', 'rb') as p:
            test_gold = pickle.load(p)

        with open('keys/test_books_preds.pkl', 'rb') as p:
            test_preds = pickle.load(p)

       # print("test_gold['1023']=",test_gold['1023'])
       # print("len(test_gold['1023']=",len(test_gold['1023']))
       # print("len(test_preds['1023'])=",len(test_preds['1023']))
        #print("test_gold['829']=",test_gold['829'])
        #print('-----')
        #print("test_preds['829']=",test_preds['829'])
        # look for what we predicted as ORG
       # for b in train_preds.keys():
       #     for p in train_preds[b]:
       #         if p[1] == 'ORG':
       #             print("p  =",p  )

        #asd

        pairs = [(train_gold, train_preds), (dev_gold, dev_preds), (test_gold, test_preds)]

        # compute metrics for each data split
        for i, pair in enumerate(pairs):
            print('-----------------------------')
            print('Computing Metrics for pair ' + str(i))
            print('-----------------------------')
            compute_metrics(pair[0], pair[1])






#tokenized_inputs = tokenizer(sents, truncation=True, is_split_into_words=True, return_tensors="pt", padding=True)
#
##test = tokenized_inputs['input_ids'][0]
#model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)
#
#outputs = model(**tokenized_inputs)
#print("outputs.logits.shape=",outputs.logits.shape)
#predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
#print("predictions=",predictions)
#
#print("model.config.id2label=",model.config.id2label)
#asd












