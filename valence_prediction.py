import os
import csv
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ValencePredictionUtils:

    def __init__(self):

        self.load_data()


    def load_data(self):

        df = pd.read_csv('../data/BRM-emot-submit.csv')
        self.valences = dict()
        self.arousals = dict()
        self.dominances = dict()
        for idx, word in enumerate(df.Word):
            self.valences[word] = df['V.Mean.Sum'][idx]
            self.arousals[word] = df['A.Mean.Sum'][idx]
            self.dominances[word] = df['D.Mean.Sum'][idx]


    def form_splits(self, x, y, num_splits):
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        shuffled_x = [x[idx] for idx in indices]
        shuffled_y = [y[idx] for idx in indices]
        folds = list()

        for idx in range(num_splits):
            start_idx = int(idx*(len(x)/num_splits))
            end_idx = int((idx + 1)*(len(x)/num_splits))
            folds.append((shuffled_x[start_idx:end_idx], shuffled_y[start_idx:end_idx]))

        return folds


    def save_embeddings(self, filepath, outpath, emb_size=300):

        unseen = dict()
        with open(filepath) as f:
            counter = 0
            for line in tqdm(f):
                split_line = line.split()
                n = len(split_line)
                word = ' '.join(split_line[0:(n-emb_size)]).strip()
                embedding = [float(x) for x in split_line[-emb_size:]]
                if(word not in self.valences):
                    unseen[word] = embedding
                if(len(unseen) == 200000):
                    with open(os.path.join(outpath, str(counter) + '.pkl'), 'wb') as fdump:
                        pkl.dump(unseen, fdump)
                    counter += 1
                    unseen = dict()
        with open(os.path.join(outpath, str(counter) + '.pkl'), 'wb') as fdump:
            pkl.dump(unseen, fdump)



    def predict_unseen(self, embedding_path, pkl_dir, out_dir, emb_size, normalization=False, target='valence'):

        if(normalization):
            model_path = os.path.join(out_dir, 'model_normalized' + target + '.pkl')
        else:
            model_path = os.path.join(out_dir, 'model' + target + '.pkl')
        if(not os.path.exists(model_path)):
            features = dict()
            with open(embedding_path) as f:
                for line in tqdm(f):
                    split_line = line.split()
                    n = len(split_line)
                    word = ' '.join(split_line[0:(n-emb_size)]).strip()
                    embedding = [float(x) for x in split_line[-emb_size:]]
                    if(word in self.valences):
                        features[word] = embedding

            X_train = np.array([features[x] for x in features])
            if(target == 'valence'):
                y_train = np.array([self.valences[x] for x in features])
            elif(target == 'arousal'):
                y_train = np.array([self.arousals[x] for x in features])
            elif(target == 'dominance'):
                y_train = np.array([self.dominances[x] for x in features])
            if(normalization):
                y_train = (y_train - min(y_train))/(max(y_train) - min(y_train))

            model = LinearRegression()
            model.fit(X_train, y_train)
            with open(model_path, 'wb') as f:
                pkl.dump(model, f)
        else:
            with open(model_path, 'rb') as f:
                model = pkl.load(f)

        for file in tqdm(os.listdir(pkl_dir)):
            with open(os.path.join(pkl_dir, file), 'rb') as f, open(os.path.join(out_dir, file), 'wb') as fw:
                data = pkl.load(f)
                embeddings = [data[x] for x in data]
                words = [x for x in data]
                predictions = model.predict(embeddings)
                predicted_targets = dict()
                for idx, prediction in enumerate(predictions):
                    predicted_targets[words[idx]] = prediction
                pkl.dump(predicted_targets, fw)


    def cross_validate(self, embeddings_path, emb_size, model_type, num_splits, normalization=False, target='valence'):

        features = dict()
        with open(embeddings_path) as f:
            print("Loading word vectors")
            for line in tqdm(f):
                split_line = line.split()
                n = len(split_line)
                word = ' '.join(split_line[0:(n-emb_size)]).strip()
                embedding = [float(x) for x in split_line[-emb_size:]]
                if(word in self.valences):
                    features[word] = embedding


        X_train = np.array([features[x] for x in features])
        if(target == 'valence'):
            y_train = np.array([self.valences[x] for x in features])
        elif(target == 'arousal'):
            y_train = np.array([self.arousals[x] for x in features])
        elif(target == 'dominance'):
            y_train = np.array([self.dominances[x] for x in features])
        if(normalization):
            y_train = (y_train - min(y_train))/(max(y_train) - min(y_train))

        mses = list()
        maes = list()
        folds = self.form_splits(X_train, y_train, num_splits)
        for idx in range(num_splits):
            train_x = list()
            train_y = list()
            test_x = folds[idx][0]
            test_y = folds[idx][1]
            for idx2, fold in enumerate(folds):
                if(idx2 == idx):
                    continue
                else:
                    train_x += folds[idx2][0]
                    train_y += folds[idx2][1]
            if(model_type == 'lr'):
                model = LinearRegression()
            elif(model_type == 'ridge'):
                model = Ridge()
            elif(model_type == 'lasso'):
                model = Lasso()
            model.fit(train_x, train_y)
            mse = mean_squared_error(test_y, model.predict(test_x))
            mae = mean_absolute_error(test_y, model.predict(test_x))
            mses.append(mse)
            maes.append(mae)

        return mses, maes


    def get_word_counts(self, wc_dir, valence_dir, out_dir, normalization=False):

        if(normalization):
            norm_dir = 'normalized'
        else:
            norm_dir = 'unnormalized'
        all_word_counts = dict()
        for file in os.listdir(wc_dir):
            with open(os.path.join(wc_dir, file), 'rb') as f:
                all_word_counts[file.split('_')[0]] = pkl.load(f)

        valence_counts = dict()
        for folder in os.listdir(valence_dir):
            for file in os.listdir(os.path.join(valence_dir, folder, norm_dir)):
                with open(os.path.join(valence_dir, folder, norm_dir, file), 'rb') as f:
                    valences = pkl.load(f)
                    if(type(valences) != dict):
                        continue
                    for word in tqdm(valences):
                        if(word not in valence_counts):
                            valence_counts[word] = dict()
                            valence_counts[word]['counts'] = dict()
                            valence_counts[word]['valences'] = dict()
                            total_count = 0
                            for source in all_word_counts:
                                valence_counts[word]['counts'][source] = all_word_counts[source][word]
                                total_count += all_word_counts[source][word]
                            valence_counts[word]['total_count'] = total_count
                        valence_counts[word]['valences'][folder] = valences[word]

        with open(os.path.join(out_dir, 'valence_counts.pkl'), 'wb') as f:
            pkl.dump(valence_counts, f)

    def get_overlap(self, target='valence', threshold=3):

        with open('../outputs/valence_counts.pkl', 'rb') as f:
            valence_counts = pkl.load(f)
        overlap = dict()
        overlap['twitter'] = dict()
        overlap['wiki'] = dict()
        overlap['cc'] = dict()
        overlap['twitter_wiki'] = dict()
        overlap['twitter_cc'] = dict()
        overlap['wiki_cc'] = dict()
        overlap['twitter_wiki_cc'] = dict()
        for word in tqdm(valence_counts):
            key = ''
            if('twitter' in valence_counts[word]['valences']):
                if(valence_counts[word]['valences']['twitter'] < threshold):
                    key = 'twitter'
            if('wiki' in valence_counts[word]['valences']):
                if(valence_counts[word]['valences']['wiki'] < threshold):
                    if(key):
                        key += '_wiki'
                    else:
                        key = 'wiki'
            if('cc' in valence_counts[word]['valences']):
                if(valence_counts[word]['valences']['cc'] < threshold):
                    if(key):
                        key += '_cc'
                    else:
                        key = 'cc'
            if(key):
                overlap[key][word] = dict()
                overlap[key][word]['valences'] = valence_counts[word]['valences']
                overlap[key][word]['counts'] = valence_counts[word]['counts']
                overlap[key][word]['total_count'] = valence_counts[word]['total_count']

        with open('../outputs/overlap.pkl', 'wb') as f:
            pkl.dump(overlap, f)

        for key in overlap:
            with open(os.path.join('../outputs/overlap_csvs', key + '.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['word', 'Common Crawl', 'Wikipedia', 'Twitter', 'Total Count'])
                for word in overlap[key]:
                    if('twitter' in overlap[key][word]['valences']):
                        twitter_valence = overlap[key][word]['valences']['twitter']
                    else:
                        twitter_valence = -1
                    if('wiki' in overlap[key][word]['valences']):
                        wiki_valence = overlap[key][word]['valences']['wiki']
                    else:
                        wiki_valence = -1
                    if('cc' in overlap[key][word]['valences']):
                        cc_valence = overlap[key][word]['valences']['cc']
                    else:
                        cc_valence = -1
                    writer.writerow([word, round(cc_valence, 2), round(wiki_valence, 2), round(twitter_valence, 2), 
                        overlap[key][word]['total_count']])









