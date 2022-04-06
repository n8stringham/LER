# Scoring Script for the Entity Recognizer 
# Script can be run from the command line python3 score.py --<path to answers>
#
# Score each book one at a time

import argparse
from collections import Counter, defaultdict

# stuff needed to import my module that's in a parent directory
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
#print("currentdir=",currentdir)
#print("parentdir=",parentdir)
sys.path.insert(0, parentdir) 

import predictions
############

parser = argparse.ArgumentParser()
# path to the booklist to be tested against
parser.add_argument('booklist')


args = parser.parse_args()
print("args.booklist=",args.booklist)



def get_gold_ents(bookpath, level, visualize=False):
    '''
    Input: path to LitBank gold annotations for a book

    merges together the BIO tags from LitBank into full entity.
    currently only does this for first nesting level

    visualize: writes the merged gold entities to a file so they can be browsed.

    Return: dictionary with key=bookname and val=list of merged gold entity tuples
    '''
    # level conversions - 0 nesting level needs to index to 1 in tab_separated
    level += 1

    print("bookpath=",bookpath)
    bookname = bookpath.split('/')[2:][0]
    print("bookname=",bookname)
    with open(bookpath, 'r') as f:
    # For each book we need to combine BIO tags to get the full entities
        tab_separated = [line.split('\t') for line in f if line != '\n']
        #print("tab_separated=",tab_separated)
        #print("len(tab_separated)=",len(tab_separated))
        is_entity = False
        entities = []
        for i in range(len(tab_separated)):
            # start keeping track if we find a 'B' token)
            if tab_separated[i][level][0] not in ['O', 'I']:
                #len(tab_separated[i]) > 1 and 
                is_entity = True
                counter = i
                start = i
                # checking the next entities
                while is_entity:
                    counter += 1
                    # we've found a new B tag or a new O so the current entity is finished
                    if tab_separated[counter][level][0] == 'B' or tab_separated[counter][level] == 'O':
                        is_entity = False
                        #print("start=",start)
                        #print("counter=",counter)
                        entity_str = ' '.join([s[0] for s in tab_separated[start:counter]])
                        
                        #first tag is the tag for the whole string - remove (B- or I-)
                        entity_tag = tab_separated[start][level][2:]
                        # which tokens does this entity span? [start, end)
                        entity_span = (start, counter)
                        entity_tuple = (entity_str, entity_tag, entity_span) 
                        #print("entity_tuple=",entity_tuple)

                        # collect all gold entities
                        entities.append(entity_tuple)

    #print("entities=",entities)
    print("len(entities)=",len(entities))
    if visualize:
        with open(bookname+'-0.gold', 'w') as f:
            for e in entities:
                f.write(str(e) + '\n')
    
    return entities


def compute_metrics(preds, gold):
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

    print("prec_macro=",prec_macro)
    print("rec_macro=",rec_macro)
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

    # open the booklist and get paths to books 
    with open(args.booklist, 'r') as f:
        books = [b.strip() for b in f]
        #books = [books[5]]
    
    # get gold entities for each book and add to dictionary
    all_ents = {}
    for book in books:
        book_id = book.split('/')[2].split('_')[0]
        print("book_id=",book_id)
        #ents = get_gold_ents(book, level=0)

        # ###########################################
        # new stuff
        # get all of the entities from 4 nesting levels [0-3]
        ents0 = get_gold_ents(book, level=0)
        ents1 = get_gold_ents(book, level=1)
        ents2 = get_gold_ents(book, level=2)
        ents3 = get_gold_ents(book, level=3)

        combined_ents = ents0 + ents1 + ents2 + ents3
        all_ents.update({book_id : combined_ents})
        #############################################
        #ents = get_gold_ents(book, visualize=True)
        #all_ents.update({book_id : ents})
        print("len(all_ents)=",len(all_ents))
    # get the dictionary with {book_id : entity}
    preds = predictions.get_pred_ents(books)

    # this is just meant to be run for one book to do a sanity check - delete or comment out
    #bookname = books[0].split('/')[2:][0]
    #with open(bookname+'-0.pred', 'w') as f:
        #book_id = books[0].split('/')[2].split('_')[0]
        #for e in preds[book_id]:
            #f.write(str(e) + '\n')
    compute_metrics(preds, all_ents)
