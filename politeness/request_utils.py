

from features.politeness_strategies import check_elems_for_strategy, initial_polar, aux_polar
import cPickle as pkl
from IPython import embed
import csv
from tqdm import tqdm

def check_is_request(document):
    """
    Heuristic to determine whether a document
    looks like a request

    :param document- pre-processed document
        that might be a request
    :type document- dict with fields 
        'sentences' and 'parses', as
        in other parts of the system
    """
    for sentence, parse in zip(document['sentences'], document['parse']):
        if "?" in sentence:
            return True
        if check_elems_for_strategy(parse, initial_polar) or check_elems_for_strategy(parse, aux_polar):
            return True
    return False




if __name__ == "__main__":


    # from test_documents import TEST_DOCUMENTS
    # embed()
    # for doc in TEST_DOCUMENTS:
    #     print "\nText: ", doc['text']
    #     print "Is request: ", check_is_request(doc)

    # print "\n"
    is_requests = list()    
    documents = list()
    toxicities = list()
    request_toxicities = list()
    request_documents = list()
    with open('../outputs/h_toxicity_parse.pkl', 'rb') as f, open('../outputs/h_toxicities.csv') as f2:
        data = pkl.load(f)
        reader = csv.reader(f2)
        for idx, row in enumerate(reader):
            documents.append(data[idx])
            toxicities.append(float(row[1]))
    with open('../outputs/m_toxicity_parse.pkl', 'rb') as f, open('../outputs/m_toxicities.csv') as f2:
        data = pkl.load(f)
        reader = csv.reader(f2)
        for idx, row in enumerate(reader):
            documents.append(data[idx])
            toxicities.append(float(row[1]))
    with open('../outputs/p_toxicity_parse.pkl', 'rb') as f, open('../outputs/p_toxicities.csv') as f2:
        data = pkl.load(f)
        reader = csv.reader(f2)
        for idx, row in enumerate(reader):
            documents.append(data[idx])
            toxicities.append(float(row[1]))
    for idx, document in tqdm(enumerate(documents)):
        if(check_is_request(document)):
            is_requests.append(True)
            request_toxicities.append(toxicities[idx])
            request_documents.append(documents[idx])
        else:
            is_requests.append(False)
    with open('../outputs/request_data.pkl', 'wb') as f:
        pkl.dump({'documents': request_documents, 'toxicities': request_toxicities}, f)
    embed()    



