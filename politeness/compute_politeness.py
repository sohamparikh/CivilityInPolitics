import sys
import os
import cPickle as pkl
import csv
from IPython import embed
import json
import numpy as np
import scipy
from scipy.sparse import csr_matrix
import sklearn
import nltk
from features.vectorizer import PolitenessFeatureVectorizer
from tqdm import tqdm
from matplotlib import pyplot as plt

MODEL_FILENAME = os.path.join(os.path.split(__file__)[0], 'politeness-svm.p')

clf = pkl.load(open(MODEL_FILENAME))
vectorizer = PolitenessFeatureVectorizer()

def score(request):
    """
    :param request - The request document to score
    :type request - dict with 'sentences' and 'parses' field
        sample (taken from test_documents.py)--
        {
            'sentences': [
                "Have you found the answer for your question?", 
                "If yes would you please share it?"
            ],
            'parses': [
                ["csubj(found-3, Have-1)", "dobj(Have-1, you-2)", "root(ROOT-0, found-3)", "det(answer-5, the-4)", "dobj(found-3, answer-5)", "poss(question-8, your-7)", "prep_for(found-3, question-8)"], 
                ["prep_if(would-3, yes-2)", "root(ROOT-0, would-3)", "nsubj(would-3, you-4)", "ccomp(would-3, please-5)", "nsubj(it-7, share-6)", "xcomp(please-5, it-7)"]
            ]
        } 

    returns class probabilities as a dict
        {
            'polite': float, 
            'impolite': float
        }
    """
    # vectorizer returns {feature-name: value} dict
    features = vectorizer.features(request)
    fv = [features[f] for f in sorted(features.iterkeys())]
    # Single-row sparse matrix
    X = csr_matrix(np.asarray([fv]))
    probs = clf.predict_proba(X)
    # Massage return format
    probs = {"polite": probs[0][1], "impolite": probs[0][0]}
    return probs


def main():
    # with open('../outputs/minute_parses.pkl', 'rb') as f, open('../minute_politeness.csv', 'w') as f2:
    #     rows = cPickle.load(f)
    #     writer = csv.writer(f2)
    #     new_rows = list()
    #     for row in tqdm(rows):
    #         impoliteness = score(row)['impolite']
    #         writer.writerow([impoliteness])
    # impolitenesses = list()
    # toxicities = list()
    # with open('../outputs/h_toxicity_parse.pkl', 'rb') as f, open('../outputs/h_toxicities.csv', 'r') as f2:
    #     reader = csv.reader(f2)
    #     parses = pkl.load(f)
    #     for row in reader:
    #         toxicities.append(float(row[1]))
    #     for parse in tqdm(parses):
    #         impoliteness = score(parse)['impolite']
    #         impolitenesses.append(impoliteness)
    # with open('../outputs/m_toxicity_parse.pkl', 'rb') as f, open('../outputs/m_toxicities.csv', 'r') as f2:
    #     reader = csv.reader(f2)
    #     parses = pkl.load(f)
    #     for row in reader:
    #         toxicities.append(float(row[1]))
    #     for parse in tqdm(parses):
    #         impoliteness = score(parse)['impolite']
    #         impolitenesses.append(impoliteness)
    # with open('../outputs/p_toxicity_parse.pkl', 'rb') as f, open('../outputs/p_toxicities.csv', 'r') as f2:
    #     reader = csv.reader(f2)
    #     parses = pkl.load(f)
    #     for row in reader:
    #         toxicities.append(float(row[1]))
    #     for parse in tqdm(parses):
    #         impoliteness = score(parse)['impolite']
    #         impolitenesses.append(impoliteness)
    # embed()
    with open('../outputs/request_data.pkl', 'rb') as f, open('../outputs/requests.csv', 'w') as f2:
        data = pkl.load(f)
        writer = csv.writer(f2)
        documents = data['documents']
        toxicities = data['toxicities']
        impolitenesses = list()
        for idx, document in tqdm(enumerate(documents)):
            impoliteness = score(document)['impolite']
            impolitenesses.append(impoliteness)
            writer.writerow([document['sentences'][0].encode('utf-8'), impoliteness, toxicities[idx]])
        # plt.hist(impolitenesses)
        # plt.xlabel('Impoliteness')
        # plt.ylabel('Number of sentences')
        # plt.show()
        plt.scatter()
        embed()
    # with open('../outputs/parsed_h.pkl', 'rb') as f, open('../outputs/h_politeness.csv', 'w') as f2:
    #     rows = cPickle.load(f)
    #     writer = csv.writer(f2)
    #     for row in tqdm(rows):
    #         impoliteness = score(row[4])['impolite']
    #         writer.writerow([row[0].encode('utf-8'), impoliteness])
    # with open('../outputs/parsed_m.pkl', 'rb') as f, open('../outputs/m_politeness.csv', 'w') as f2:
    #     rows = cPickle.load(f)
    #     writer = csv.writer(f2)
    #     for row in tqdm(rows):
    #         impoliteness = score(row[4])['impolite']
    #         writer.writerow([row[0].encode('utf-8'), impoliteness])
    # with open('../outputs/parsed_p.pkl', 'rb') as f, open('../outputs/p_politeness.csv', 'w') as f2:
    #     rows = cPickle.load(f)
    #     writer = csv.writer(f2)
    #     for row in tqdm(rows):
    #         impoliteness = score(row[4])['impolite']
    #         writer.writerow([row[0].encode('utf-8'), impoliteness])
    # with open('../outputs/nic_comments.pkl', 'rb') as f, open('../outputs/nic_politeness.csv', 'w') as f2:
    #     rows = cPickle.load(f)
    #     writer = csv.writer(f2)
    #     for row in tqdm(rows[1:]):
    #         impoliteness = score(row[3])['impolite']
    #         writer.writerow([row[0], row[1], impoliteness])
    # with open('../outputs/combined_politeness_d.pkl', 'rb') as f, open('../outputs/combined_politeness_d.csv', 'w') as f2:
    #     rows = cPickle.load(f)
    #     writer = csv.writer(f2)
    #     for row in rows:
    #         row[3] =  score(row[3])['impolite']
    #         writer.writerow(row)
#     with open('../outputs/debate_toxicity.json') as f:
#         parts = json.load(f)
#     for part in parts:
#         for dialogue in tqdm(parts[part]['dialogues']):
#             dialogue['politeness'] = score(dialogue['parse'])
        # for speaker in tqdm(parts[part]['dialogue_toxicities']):
        #     for dialogue in parts[part]['dialogue_toxicities'][speaker]:
        #         dialogue['politeness'] = score(dialogue['parse'])
        # for speaker in tqdm(parts[part]['sentence_toxicities']):
        #     for dialogue in parts[part]['sentence_toxicities'][speaker]:
        #         dialogue['politeness'] = score(dialogue['parse'])
    # with open('../outputs/debate_politeness.pkl', 'wb') as f:
    #     cPickle.dump(parts, f, cPickle.HIGHEST_PROTOCOL)



    # with open('../outputs/h_toxicities_2.pkl', 'rb') as f, open('../outputs/h_politeness.pkl', 'wb') as f2:
    #     toxicities = cPickle.load(f)
    #     for dialogue in tqdm(toxicities):
    #         dialogue['politeness'] = score(dialogue['parse'])
    #         for sentence in dialogue['sentences']:
    #             sentence['politeness'] = score(sentence['parse'])

    #     cPickle.dump(toxicities, f2, cPickle.HIGHEST_PROTOCOL)

    # with open('../outputs/m_toxicities_2.pkl', 'rb') as f, open('../outputs/m_politeness.pkl', 'wb') as f2:
    #     toxicities = cPickle.load(f)
    #     for dialogue in tqdm(toxicities):
    #         dialogue['politeness'] = score(dialogue['parse'])
    #         for sentence in dialogue['sentences']:
    #             sentence['politeness'] = score(sentence['parse'])

    #     cPickle.dump(toxicities, f2, cPickle.HIGHEST_PROTOCOL)

    # with open('../outputs/p_toxicities_2.pkl', 'rb') as f, open('../outputs/p_politeness.pkl', 'wb') as f2:
    #     toxicities = cPickle.load(f)
    #     for dialogue in tqdm(toxicities):
    #         dialogue['politeness'] = score(dialogue['parse'])
    #         for sentence in dialogue['sentences']:
    #             sentence['politeness'] = score(sentence['parse'])

    #     cPickle.dump(toxicities, f2, cPickle.HIGHEST_PROTOCOL)



    

if __name__ == '__main__':
    main()

