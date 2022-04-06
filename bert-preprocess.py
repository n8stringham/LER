# This script is where we preprocess LitBank Training set so we 
# can use if for fine-tuning with BERT. This script also does
# the fine-tuning with HuggingFace Trainer

def tokenize_and_align_labels(sents, tags):
    '''
    This function takes in the list of sentences we want to tokenize
    It tokenizes them and aligns the labels to the subwords
    '''
    data_points = {}
    for i, (sent, tag) in enumerate(zip(sents, tags)):
        # tokenize a single sentence
        tokenized_inputs = tokenizer(sent, truncation=True, is_split_into_words=True)
        #word_ids = tokenized_inputs.word_ids(i)
        word_ids = tokenized_inputs.word_ids()
        # convert string tags to ints
        tags_num = label_to_int(labels_dict, tag)
        #new_labels.append(align_labels(tags_num, word_ids))
        tokenized_inputs['labels'] = align_labels(tags_num, word_ids)
        #tokenized_inputs['ner_tags'] = tag
        #tokenized_inputs['toks'] = sent
        data_points[i] = tokenized_inputs


    #new_labels = []
    #for i, tag in enumerate(tags):
    #    word_ids = tokenized_inputs.word_ids(i)
    #    # convert string tags to ints
    #    tags_num = label_to_int(labels_dict, tags)
    #    new_labels.append(align_labels(tags_num, word_ids))
    return data_points


def align_labels(labels_num, word_ids):
    '''
    align labels to sub-word tokenized sentence
    '''
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of word
            current_word = word_id
            # set special tokens to have label that will be ignored
            # during training
            if word_id is None:
                label = -100
            # same label as original
            else:
                label = labels_num[word_id]

            new_labels.append(label)
        
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels_num[word_id]
            # I labels are odds
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 0 and label !=12:
                label += 1
            new_labels.append(label)

    return new_labels    

def label_to_int(mapping, labels):
    '''
    takes a set of string labels and maps them to ints
    '''
    return [mapping[l] for l in labels]


def get_tokens_and_tags(books):
    '''
    input: list of paths to books
    output: [...[tokens for sentence i]...], [...[tags for sentence i]...]
    '''
    sentence_tokens = []
    ner_tags = {'0' : [], '1': [], '2': [], '3': []} 
    # need the labels found in the dataset
    labels = set()
    for  b in books:
        tab_separated = []
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

    return sentence_tokens, ner_tags, labels


def compute_metrics(eval_preds):
    print("eval_preds=",eval_preds)
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    
    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[int_to_label_dict[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [int_to_label_dict[p] for (p, l) in zip(prediction, label) if l != -100]
    #true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    #true_predictions = [
    #    [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }


from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments
from transformers import Trainer

from transformers import DataCollatorForTokenClassification
from datasets import load_metric
import numpy as np

import torch

if __name__ == '__main__':
    # open the training booklist and get paths to books 
    with open('./data/train/books.txt', 'r') as f:
        train_books = [b.strip() for b in f]

    # open the dev booklist and get paths
    with open('./data/dev/books.txt', 'r') as f:
        dev_books = [b.strip() for b in f]
    
    # open the test booklist and get paths
    with open('./data/test/books.txt', 'r') as f:
        test_books = [b.strip() for b in f]

   # print("train_books=",train_books)
   # print("dev_books=",dev_books)
   # print("test_books=",test_books)

    train_sents, train_tags, train_labels = get_tokens_and_tags(train_books)
    dev_sents, dev_tags, dev_labels = get_tokens_and_tags(dev_books)

    # sort the labels so that evens are B-, odds are I, and O is last
    ls = list(train_labels)
    ls.sort()
    ls = sorted(ls[:-1], key = lambda x : x[2]) + list(ls[-1])
    # dictionary to map labels to ints
    labels_dict = dict(zip(ls, range(13)))
    int_to_label_dict = {val : key for key, val in labels_dict.items()}
    print("labels_dict=",labels_dict)
    print("int_to_label_dict.items()=",int_to_label_dict.items())

    #############
    # Tokenization
    # ###########
    
    model_checkpoint = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # tokenize and align
    tokenized_sents = tokenize_and_align_labels(train_sents, train_tags['0'])

    dev_tokenized_sents = tokenize_and_align_labels(dev_sents, dev_tags['0'])
    
    print("len(tokenized_sents)=",len(tokenized_sents))
    print("len(dev_tokenized_sents)=",len(dev_tokenized_sents))

    print("tokenized_sents[0]['label']=",tokenized_sents[0]['labels'])


    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # test out a few examples - sanity check X
   
    #batch = data_collator([tokenized_sents[i] for i in range(2)])
    #print("batch['labels']=",batch['labels'])

    #for i in range(2):
        #print("tokenized_sents[i]['labels']=",tokenized_sents[i]['labels'])
   #     print("tokenizer.decode(tokenized_sents[i]['input_ids'])=",tokenizer.decode(tokenized_sents[i]['input_ids']))
   #     print("tokenizer.tokenize(train_sents[i])=",tokenizer.tokenize(' '.join(train_sents[i])))
   
    metric = load_metric("seqeval")

   # # test the metric
   # ls = train_tags['0'][0]
   # print("ls=",ls)
   # predictions = ls.copy()
   # predictions[3] = "O"
   # m = metric.compute(predictions=[predictions], references=[ls])
   # print("m=",m)

    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=13
        #id2label=int_to_label_dict,
        #label2id=labels_dict,
    )


    num = model.config.num_labels
    print("num=",num)


    args = TrainingArguments(
        "./models/bert-finetuned-ner",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        #push_to_hub=True,
    )


    # define the trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_sents,
        eval_dataset=dev_tokenized_sents,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    print('Make sure to uncomment trainer.train')

    # Fine-Tune it!
    #trainer.train()

    # push to the hub
    #trainer.push_to_hub(commit_message="Training complete")






