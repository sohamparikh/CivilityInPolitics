import numpy as np
import sys
import os
# import cPickle
import csv
import pickle as pkl
import scipy
from scipy.sparse import csr_matrix
import sklearn
import nltk
import spacy
from data_utils import DebateParser
from IPython import embed
from tqdm import tqdm
import json

nlp = spacy.load('en')

def get_parse(doc):

    parse = dict()
    parse['parse'] = list()
    parse['sentences'] = list()
    for sent in nlp(doc).sents:
        sent_parse = list()
        for token in sent:
            dep = token.dep_
            current_id = token.i
            if(token.head == token):
                parent_id = 0
                parent = "ROOT"
            else:
                parent = token.head.text
                parent_id = token.head.i
            token_parse = dep + '(' + parent + '-' + str(parent_id) + ', ' + token.text + '-' + str(current_id) + ')'
            sent_parse.append(token_parse)
        parse['parse'].append(sent_parse)
        parse['sentences'].append(sent.text)

    return parse

def main():
    # with open('../data/comments_forPoliteness - comments_forPoliteness.csv') as f, open('../outputs/nic_comments.pkl', 'wb') as f2:
    #     reader = csv.reader(f)
    #     new_list = list()
    #     for row in tqdm(reader):
    #         parse = get_parse(row[1])
    #         row.append(parse)
    #         new_list.append(row)
    #     pkl.dump(new_list, f2, 2)
    # with open('../outputs/minute_toxicity.pkl', 'rb') as f, open('../outputs/minute_parses.pkl', 'wb') as f2:
    #     data = pkl.load(f)
    #     new_rows = list()
    #     for row in tqdm(data):
    #         parse = get_parse(row[0])
    #         new_rows.append(parse)
    #     pkl.dump(new_rows, f2, 2)
    with open('../outputs/h_toxicities.csv') as f, open('../outputs/h_toxicity_parse.pkl', 'wb') as f2:
        reader = csv.reader(f)
        parses = list()
        for row in tqdm(reader):
            parse = get_parse(row[0])
            parses.append(parse)
        pkl.dump(parses, f2, 2)
    with open('../outputs/m_toxicities.csv') as f, open('../outputs/m_toxicity_parse.pkl', 'wb') as f2:
        reader = csv.reader(f)
        parses = list()
        for row in tqdm(reader):
            parse = get_parse(row[0])
            parses.append(parse)
        pkl.dump(parses, f2, 2)
    with open('../outputs/p_toxicities.csv') as f, open('../outputs/p_toxicity_parse.pkl', 'wb') as f2:
        reader = csv.reader(f)
        parses = list()
        for row in tqdm(reader):
            parse = get_parse(row[0])
            parses.append(parse)
        pkl.dump(parses, f2, 2)
    # with open('../outputs/h_sentences.csv') as f, open('../outputs/parsed_h.pkl', 'wb') as f2:
    #     reader = csv.reader(f)
    #     new_list = list()
    #     for row in tqdm(reader):
    #         parse = get_parse(row[0])
    #         row.append(parse)
    #         new_list.append(row)
    #     pkl.dump(new_list, f2, 2)
    # with open('../outputs/p_sentences.csv') as f, open('../outputs/parsed_p.pkl', 'wb') as f2:
    #     reader = csv.reader(f)
    #     new_list = list()
    #     for row in tqdm(reader):
    #         parse = get_parse(row[0])
    #         row.append(parse)
    #         new_list.append(row)
    #     pkl.dump(new_list, f2, 2)
    # with open('../outputs/m_sentences.csv') as f, open('../outputs/parsed_m.pkl', 'wb') as f2:
    #     reader = csv.reader(f)
    #     new_list = list()
    #     for row in tqdm(reader):
    #         parse = get_parse(row[0])
    #         row.append(parse)
    #         new_list.append(row)
    #     pkl.dump(new_list, f2, 2)

    # with open('../outputs/combined_tuples_d.csv', 'r') as f, open('../outputs/combined_politeness_d.pkl', 'wb') as f2:
    #     reader = csv.reader(f)
    #     new_list = list()
    #     for row in tqdm(reader):
    #         parse = get_parse(row[0])
    #         row.append(parse)
    #         new_list.append(row)
    #     pkl.dump(new_list, f2, 2)

    # with open('../outputs/h_toxicities.pkl', 'rb') as f, open('h_toxicities_2.pkl', 'wb') as f2:
    #     toxicities = pkl.load(f)
    #     for dialogue in tqdm(toxicities):
    #         dialogue['parse'] = get_parse(dialogue['text'])
    #         dialogue['sentences'] = list()
    #         sentences = list(nlp(dialogue['text']).sents)
    #         if(len(sentences) != len(dialogue['sentence_toxicities'])):
    #             continue
    #         for idx, sentence in enumerate(sentences):
    #             sentence_dict = dict()
    #             sentence_dict['text'] = sentence.text
    #             sentence_dict['toxicity'] = dialogue['sentence_toxicities'][idx]
    #             sentence_dict['parse'] = get_parse(sentence.text)
    #             dialogue['sentences'].append(sentence_dict)

    #     pkl.dump(toxicities, f2, 2)

    # with open('../outputs/m_toxicities.pkl', 'rb') as f, open('m_toxicities_2.pkl', 'wb') as f2:
    #     toxicities = pkl.load(f)
    #     for dialogue in tqdm(toxicities):
    #         dialogue['parse'] = get_parse(dialogue['text'])
    #         dialogue['sentences'] = list()
    #         sentences = list(nlp(dialogue['text']).sents)
    #         if(len(sentences) != len(dialogue['sentence_toxicities'])):
    #             continue
    #         for idx, sentence in enumerate(nlp(dialogue['text']).sents):
    #             sentence_dict = dict()
    #             sentence_dict['text'] = sentence.text
    #             sentence_dict['toxicity'] = dialogue['sentence_toxicities'][idx]
    #             sentence_dict['parse'] = get_parse(sentence.text)
    #             dialogue['sentences'].append(sentence_dict)

    #     pkl.dump(toxicities, f2, 2)

    # with open('../outputs/p_toxicities.pkl', 'rb') as f, open('p_toxicities_2.pkl', 'wb') as f2:
    #     toxicities = pkl.load(f)
    #     for dialogue in tqdm(toxicities):
    #         dialogue['parse'] = get_parse(dialogue['text'])
    #         dialogue['sentences'] = list()
    #         sentences = list(nlp(dialogue['text']).sents)
    #         if(len(sentences) != len(dialogue['sentence_toxicities'])):
    #             continue
    #         for idx, sentence in enumerate(nlp(dialogue['text']).sents):
    #             sentence_dict = dict()
    #             sentence_dict['text'] = sentence.text
    #             sentence_dict['toxicity'] = dialogue['sentence_toxicities'][idx]
    #             sentence_dict['parse'] = get_parse(sentence.text)
    #             dialogue['sentences'].append(sentence_dict)

    #     pkl.dump(toxicities, f2, 2)
        


if __name__ == '__main__':
    main()






