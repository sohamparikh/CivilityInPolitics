import spacy
import string
import random
import numpy as np
from tqdm import tqdm
from sklearn.naive_bayes import MultinomialNB
from IPython import embed
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class ClassifierUtils:

    def __init__(self):
        self.nlp_light = spacy.load('en', disable=['tagger', 'parser', 'ner'])
        self.nlp = spacy.load('en')
        self.stop_words = spacy.lang.en.stop_words.STOP_WORDS
        self.alphabet = string.ascii_lowercase 
        self.load_names()


    def load_names(self):
        self.hannity_names = list()
        self.maddow_names = list()
        self.pbs_names = list()
        with open('../data/names/Hannity_names.txt') as f:
            for line in f:
                for token in line.split():
                    self.hannity_names.append(token.lower())

        with open('../data/names/Maddow_names.txt') as f:
            for line in f:
                for token in line.split():
                    self.maddow_names.append(token.lower())

        with open('../data/names/PBS_names.txt') as f:
            for line in f:
                for token in line.split():
                    self.pbs_names.append(token.lower())

        self.host_names = self.hannity_names + self.maddow_names + self.pbs_names


    def forbidden(self, token):

        if(token.lower() in self.stop_words or token.lower() in self.host_names):
            return True
        else:
            for character in token.lower():
                if(character in self.alphabet):
                    return False
            return True

        return False


    def forbidden_sw(self, token):

        if(token.lower() in self.host_names):
            return True
        else:
            for character in token.lower():
                if(character in self.alphabet):
                    return False
            return True

        return False


    def prepare_for_input(self, documents, label, vectorizer='count'):

        y = [label]*len(documents)
        if(vectorizer == 'count'):
            X = self.count_vectorizer(documents)
        elif(vectorizer == 'tfidf'):
            X = self.tfidf_vectorizer(documents)

        return X, y


    def form_splits(self, shows, labels, num_splits=5):

        indices = np.arange(len(shows))
        random.shuffle(indices)
        X_shuffled = [shows[i] for i in indices]
        y_shuffled = [labels[i] for i in indices]
        splits = list()

        for idx in range(num_splits):
            current_split = dict()
            start_index = int(idx*(len(X_shuffled)/num_splits))
            end_index = int((idx+1)*(len(X_shuffled)/num_splits))
            if(idx == num_splits - 1):
                current_split['test_docs'] = X_shuffled[start_index:max(end_index, len(X_shuffled))]
                current_split['y_test'] = y_shuffled[start_index:max(end_index, len(y_shuffled))]
                current_split['train_docs'] = X_shuffled[:start_index]
                current_split['y_train'] = y_shuffled[:start_index]
            else:
                current_split['test_docs'] = X_shuffled[start_index:end_index]
                current_split['y_test'] = y_shuffled[start_index:end_index]
                current_split['train_docs'] = X_shuffled[:start_index] + X_shuffled[end_index:]
                current_split['y_train'] = y_shuffled[:start_index] + y_shuffled[end_index:]

            splits.append(current_split)

        return splits


    def get_sentences_labels(self, shows, labels):

        sentences = list()
        return_labels = list()
        for idx, show in enumerate(shows):
            current_label = labels[idx]
            for part in show[1]:
                for dialogue in part:
                    for sentence in self.nlp(dialogue[1]).sents:
                        sentences.append(sentence.text)
                        return_labels.append(current_label)

        return sentences, return_labels


    def get_dialogues_labels(self, shows, labels):

        dialogues = list()
        return_labels = list()
        for idx, show in enumerate(shows):
            current_label = labels[idx]
            for part in show[1]:
                for dialogue in part:
                    dialogues.append(dialogue[1])
                    return_labels.append(current_label)

        return dialogues, return_labels


    def cross_validate_sentences_nonoverlapping(self, shows, labels, clf_type='NB', multi_class='ovr'):

        splits = self.form_splits(shows, labels)
        accuracies = list()
        for split in tqdm(splits):
            test_sentences, test_labels = self.get_sentences_labels(split['test_docs'], split['y_test'])
            train_sentences, train_labels = self.get_sentences_labels(split['train_docs'], split['y_train'])
            accuracy = self.evaluate(train_sentences, train_labels, test_sentences, test_labels, clf_type=clf_type, multi_class=multi_class)
            accuracies.append(accuracy)

        return accuracies


    def cross_validate_dialogues_nonoverlapping(self, shows, labels, clf_type='NB', multi_class='ovr'):

        splits = self.form_splits(shows, labels)
        accuracies = list()
        for split in tqdm(splits):
            test_dialogues, test_labels = self.get_dialogues_labels(split['test_docs'], split['y_test'])
            train_dialogues, train_labels = self.get_dialogues_labels(split['train_docs'], split['y_train'])
            accuracy = self.evaluate(train_dialogues, train_labels, test_dialogues, test_labels, clf_type=clf_type, multi_class=multi_class)
            accuracies.append(accuracy)

        return accuracies



    def cross_validate(self, documents, labels, clf_type='NB'):

        splits = self.form_splits(documents, labels)
        accuracies = list()
        for split in tqdm(splits):
            accuracy = self.evaluate(split['train_docs'], split['y_train'], split['test_docs'], split['y_test'], clf_type=clf_type)
            accuracies.append(accuracy)

        return accuracies


    def evaluate(self, train_docs, y_train, test_docs, y_test, clf_type='NB', multi_class='ovr'):

        if(clf_type == 'NB'):
            clf = MultinomialNB()
        elif(clf_type == 'LR'):
            if(multi_class == 'ovr'):
                clf = LogisticRegression()
            else:
                clf = LogisticRegression(multi_class='multinomial', solver='saga')
        count_vectorizer = CountVectorizer(tokenizer=self.tokenizer)
        count_vectorizer.fit(train_docs)
        X_train = count_vectorizer.transform(train_docs)
        X_test = count_vectorizer.transform(test_docs)
        clf.fit(X_train, y_train)
        y_predicted = clf.predict(X_test)
        metrics = classification_report(y_test, y_predicted, output_dict=True)
        metrics['accuracy'] = accuracy_score(y_test, y_predicted)

        return metrics


    def get_nb_features(self, train_docs, labels):

        clf = MultinomialNB()
        count_vectorizer = CountVectorizer(tokenizer=self.tokenizer)
        count_vectorizer.fit(train_docs)
        X_train = count_vectorizer.transform(train_docs)
        clf.fit(X_train, labels)

        features = count_vectorizer.get_feature_names()
        ordered_features = [[features[idx] for idx in array[::-1]] for array in np.argsort(clf.coef_, axis=1)]

        return ordered_features


    def get_lr_features(self, train_docs, labels, multi_class='ovr'):

        if(multi_class == 'ovr'):
            clf = LogisticRegression()
        else:
            clf = LogisticRegression(multi_class='multinomial', solver='saga')
        count_vectorizer = CountVectorizer(tokenizer=self.tokenizer)
        count_vectorizer.fit(train_docs)
        X_train = count_vectorizer.transform(train_docs)
        clf.fit(X_train, labels)

        features = count_vectorizer.get_feature_names()
        ordered_features = [[features[idx] for idx in array[::-1]] for array in np.argsort(clf.coef_, axis=1)]

        return ordered_features


    def tokenizer(self, text):
        return [token.text for token in self.nlp_light(text) if (not self.forbidden(token.text))]

    def tokenizer_sw(self, text):
        return [token.text for token in self.nlp_light(text) if (not self.forbidden_sw(token.text))]

    def compute_count_vectorizer(self, documents):

        self.count_vectorizer = CountVectorizer(tokenizer=self.tokenizer)
        self.count_vectorizer.fit(documents)

    def compute_tfidf_vectorizer(self, documents):

        self.tfidf_vectorizer = TfidfVectorizer(tokenizer=self.tokenizer)
        self.tfidf_vectorizer.fit(documents)

