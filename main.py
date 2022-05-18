# https://github.com/otuncelli/turkish-stemmer-python is used for stemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()

import csv
import pandas as pd
import re
import regex
from matplotlib import pyplot as plt
import string
import numpy as np
from TurkishStemmer import TurkishStemmer
import json

stemmer = TurkishStemmer()

PAD_tkn = 0
Strt_tkn = 1

tr_stop_words = []

with open("turkce-stop-words.txt", "r", encoding="utf-8") as stop_words:
    for line in stop_words:
        tr_stop_words.append(line)

labels = [
    "labeleconomics",
    "labelhealth",
    "labelsports",
    "labeltechnology",
    "labellife"
]

files = [
    "economics.txt",
    "health.txt",
    "sports.txt",
    "technology.txt",
    "life.txt"
]


wordsToIndex = {}
wordsToCount = {}
indexToWord = {PAD_tkn : "PAD", Strt_tkn: "SRT"}
num_words = 0


def containsLetterAndNumber(input):
    has_letter = False
    has_number = False
    for x in input:
        if x.isalpha():
            has_letter = True
        elif x.isnumeric():
            has_number = True
        if has_letter and has_number:
            return True
    return False

def add_word(word):
    global num_words
    #if word in labels:
    #    return
    if word not in wordsToIndex:
        wordsToIndex[word] = num_words
        wordsToCount[word] = 1
        indexToWord[num_words] = word
        num_words += 1
    else:
        wordsToCount[word] += 1




data = []
labels = []
data_test = []
labels_test = []
# eco_train_labels
# line_counter = 0
num_of_lines = 0

def order_dict():
    wordsToCount2 = json.load(open("vocab_count.txt"))
    wordsToCount2_sorted = {k: v for k, v in sorted(wordsToCount2.items(), key=lambda item: item[-1], reverse=True)}
    json.dump(wordsToCount2_sorted, open("vocab_count.txt", "w"))

#order_dict()


def preprocess(filename):
    with open(filename, "r", encoding='utf8') as fp:
        num_of_lines = sum(1 for line in fp)

    with open(filename, "r", encoding='utf8') as fp:
        print(num_of_lines)
        line_counter = 0
        for line in fp:
            line = line.translate(str.maketrans("","",string.punctuation+'’'))
            line = np.array(line.split())
            line_new = []
            line_new.append(Strt_tkn)
            for word in line[1:]:
                if containsLetterAndNumber(word):
                    word = word.translate(str.maketrans('','', string.digits))
                word_list = re.findall('[a-zA-Z][^A-Z]*', word)
                for words in word_list:
                    words = words.replace('\xad', '')
                    words = words.replace('\x92', '')
                    words = stemmer.stem(words.lower())
                    if words not in tr_stop_words:
                        add_word(words)
                        line_new.append(wordsToIndex[words])

            if line_counter < 0.6 * num_of_lines:
                labels.append(line[0])
                data.append(line_new)
                line_counter += 1
            else:
                labels_test.append(line[0])
                data_test.append(line_new)
                line_counter += 1



for file in files:
    preprocess(file)

def write_to_csv():
    with open("datas.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)
    with open("labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(labels)
    with open("data_test.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data_test)
    with open("test_labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(labels_test)


write_to_csv()

def write_vocab():
    json.dump(wordsToIndex, open("vocab_index.txt", "w"))
    json.dump(wordsToCount, open("vocab_count.txt", "w"))
    json.dump(indexToWord, open("index_vocab.txt", "w"))


write_vocab()

datas_read = []
label_read = []
data_test_read = []
label_test_read = []
def read_from_csv():
    with open('datas.csv', 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        datas_read = list(csv_reader)
    with open("labels.csv", "r") as f:
        csv_reader = csv.reader(f)
        list_a = list(csv_reader)
        label_read = list_a[0]
    with open('data_test.csv', 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        data_test_read = list(csv_reader)
    with open("test_labels.csv", "r") as f:
        csv_reader = csv.reader(f)
        list_a = list(csv_reader)
        label_test_read = list_a[0]

read_from_csv()

lengths = [len(i) for i in data]
def averageLen(lst):
    lengths = [len(i) for i in lst]
    return 0 if len(lengths) == 0 else (float(sum(lengths)) / len(lengths))





#model = make_pipeline(TfidfVectorizer(), MultinomialNB())
#model.fit(datas_read, label_test_read)
#labels_matrix = model.predict(data_test_read)


#from sklearn.metrics import confusion_matrix
#mat = confusion_matrix(labels_test_read, labels_matrix)
#sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
#            xticklabels=train.target_names, yticklabels=train.target_names)
#plt.xlabel('true label')
#plt.ylabel('predicted label');

