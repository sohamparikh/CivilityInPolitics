import time
import spacy
import string
from tqdm import tqdm
from IPython import embed
from nltk import ngrams
from apiclient import discovery
from collections import Counter
from spacy.lang.en.stop_words import STOP_WORDS
from allennlp.predictors import Predictor

class ShowAnalyzer:

    def __init__(self):

        self.nlp = spacy.load('en')
        self.nlp_light = spacy.load('en', disable=['parser', 'tagger', 'ner'])
        self.nlp_light.add_pipe(self.nlp_light.create_pipe('sentencizer'))
        self.API_KEYS = ['AIzaSyDGOeosoFs7i6WWfxvaUyF2Ow5X2fT2AjY', 'AIzaSyBO3kOcvtGPcn51Xy2I-8GFp_7P2eGC_nI', 'AIzaSyBbS4-2w9U3M-PKtGWZqlk9hXrXtkdQzro']
        self.service1 = discovery.build('commentanalyzer', 'v1alpha1', developerKey=self.API_KEYS[0])
        self.service2 = discovery.build('commentanalyzer', 'v1alpha1', developerKey=self.API_KEYS[1])
        self.service3 = discovery.build('commentanalyzer', 'v1alpha1', developerKey=self.API_KEYS[2])
        self.alphabet = string.ascii_lowercase
        #self.ner_model = Predictor.from_path('/Users/sohamp/Documents/Projects/models/allennlp/ner-model-2018.12.18.tar.gz')


    def get_dialogue_length(self, dialogue):

        return len(self.nlp_light(dialogue))

    def get_average_show_length(self, shows):

        total_length = 0
        for show in shows:
            total_length += self.get_show_length(show[1])

        return total_length/len(shows)


    def get_dialogue_length_per_speaker(self, speaker, shows):

        total_dialogue_length = 0
        num_dialogues = 0
        for show in shows:
            for part in show[1]:
                for dialogue in part:
                    if(speaker in dialogue[0].lower()):
                        dialogue_length = self.get_dialogue_length(dialogue[1])
                        total_dialogue_length += dialogue_length
                        num_dialogues += 1

        return total_dialogue_length/num_dialogues


    def get_dialogue_length_per_show(self, shows):

        total_dialogue_length = 0
        num_dialogues = 0
        for show in shows:
            for part in show[1]:
                for dialogue in part:
                    dialogue_length = self.get_dialogue_length(dialogue[1])
                    total_dialogue_length += dialogue_length
                    num_dialogues += 1

        return total_dialogue_length/num_dialogues


    def get_dialogue_length_per_show_no_host(self, speaker, shows):

        total_dialogue_length = 0
        num_dialogues = 0
        for show in shows:
            for part in show[1]:
                for dialogue in part:
                    if(speaker not in dialogue[0].lower()):
                        dialogue_length = self.get_dialogue_length(dialogue[1])
                        total_dialogue_length += dialogue_length
                        num_dialogues += 1

        return total_dialogue_length/num_dialogues



    def get_show_length(self, show):

        show_length = 0
        for part in show:
            for dialogue in part:
                dialogue_length = self.get_dialogue_length(dialogue[1])
                show_length += dialogue_length

        return show_length


    def get_show_toxicity(self, shows, service=None):

        dialogues = list()
        for show in shows:
            for part in show[1]:
                for dialogue in tqdm(part):
                    toxicities = dict()
                    dialogue_toxicity, span_scores = self.get_toxicity(dialogue[1])
                    toxicities['dialogue'] = {'text': dialogue[1], 'in_video': dialogue[2], 'toxicity': dialogue_toxicity, 'span_scores': span_scores}
                    sentences = list(self.nlp_light(dialogue[1]).sents)
                    toxicities['sentences'] = list()
                    if(len(sentences) > 1):
                        for sentence in sentences:
                            if(service is None):
                                sentence_toxicity, span_scores = self.get_toxicity(sentence.text)
                            else:
                                sentence_toxicity, span_scores = self.get_toxicity(sentence.text, service)
                            toxicities['sentences'].append({'text': sentence.text, 'in_video': dialogue[2], 'toxicity':sentence_toxicity, 'span_scores': span_scores})
                    else:
                        toxicities['sentences'] = [{'text': dialogue[1], 'in_video': dialogue[2], 'toxicity': dialogue_toxicity, 'span_scores': span_scores}]
                    dialogues.append(toxicities)

        return dialogues


    def get_toxicity(self, text, service=None):

        comment = dict()
        comment['text'] = text
        analyze_request = dict()
        analyze_request['comment'] = comment
        # analyze_request['spanAnnotations'] = True
        analyze_request['requestedAttributes'] = {'TOXICITY': {}}
        try:
            if(service is None):
                response = self.service1.comments().analyze(body=analyze_request).execute()
            else:
                response = service.comments().analyze(body=analyze_request).execute()
            value = response['attributeScores']['TOXICITY']['summaryScore']['value']
            # span_scores = response['attributeScores']['TOXICITY']['spanScores']
        except:
            embed()
            value = None
        time.sleep(1)

        return value


    def get_common_words(self, shows, n=1):

        counter = Counter()
        for show in tqdm(shows):
            for part in show[1]:
                for dialogue in part:
                    for token in self.nlp_light(dialogue[1]):
                        if(not self.forbidden(token.text)):
                            counter[token.text.lower()] += 1

        return counter


    def get_common_ngrams(self, shows, max_n = 4):

        ngram_dict = dict()
        for n in range(1, max_n+1):
            ngram_dict[n] = Counter()
        for show in shows:
            for part in show[1]:
                for dialogue in part:
                    text = dialogue[1]
                    for n in range(1, max_n+1):
                        token_list = [token.text for token in self.nlp_light(text.strip())]
                        ngram_list = ngrams(token_list, n)
                        for ngram in ngram_list:
                            ngram_dict[n][ngram] += 1

        return ngram_dict


    def get_noun_chunks(self, shows):

        noun_chunk_counter = Counter()
        for show in shows:
            for part in show[1]:
                for dialogue in part:
                    text = dialogue[1]
                    for noun_chunk in self.nlp(text).noun_chunks:
                        noun_chunk_counter[noun_chunk.text.lower()] += 1

        return noun_chunk_counter


    def get_named_entities_text(self, text):

        named_entities = list()
        try:
            response = self.ner_model.predict(text.strip())
        except:
            return list()
        prev_tag = 'O'
        curr_ne = list()
        for idx, tag in enumerate(response['tags']):
            if(prev_tag == 'O'):
                if(tag == 'O'):
                    continue
                else:
                    prev_tag = tag
                    curr_ne.append(response['words'][idx])
            else:
                if(tag == 'O'):
                    named_entities.append(' '.join(curr_ne))
                    curr_ne = list()
                else:
                    curr_ne.append(response['words'][idx])
                prev_tag = tag


        if(prev_tag != 'O'):
            named_entities.append(' '.join(curr_ne))

        return named_entities 





    def get_named_entities_shows(self, shows):

        named_entities = Counter()
        for show in shows:
            for part in show[1]:
                for dialogue in part:
                    text = dialogue[1]
                    for named_entity in self.get_named_entities_text(text.strip()):
                        named_entities[named_entity.lower()] += 1

        return named_entities


    def forbidden_chunk(self, noun_chunk):

        for token in self.nlp_light(noun_chunk.strip()):
            if(not self.forbidden(token.text)):
                return False

        return True


    def forbidden(self, token):

        if(token.lower() in STOP_WORDS):
            return True
        else:
            for character in token.lower():
                if(character in self.alphabet):
                    return False
            return True

        return False



class DebateAnalyzer:

    def __init__(self):

        self.nlp = spacy.load('en')
        self.nlp_light = spacy.load('en', disable=['parser', 'tagger', 'ner'])
        self.stopwords = STOP_WORDS
        self.API_KEYS = ['AIzaSyDGOeosoFs7i6WWfxvaUyF2Ow5X2fT2AjY', 'AIzaSyBO3kOcvtGPcn51Xy2I-8GFp_7P2eGC_nI', 'AIzaSyBbS4-2w9U3M-PKtGWZqlk9hXrXtkdQzro']
        self.service1 = discovery.build('commentanalyzer', 'v1alpha1', developerKey=self.API_KEYS[0])
        self.service2 = discovery.build('commentanalyzer', 'v1alpha1', developerKey=self.API_KEYS[1])
        self.service3 = discovery.build('commentanalyzer', 'v1alpha1', developerKey=self.API_KEYS[2])
        self.alphabet = string.ascii_lowercase


    def get_dialogue_length(self, dialogue):

        return len(self.nlp_light(dialogue))

    def get_toxicity(self, text, service=None):

        comment = dict()
        comment['text'] = text
        analyze_request = dict()
        analyze_request['comment'] = comment
        analyze_request['requestedAttributes'] = {'TOXICITY': {}}
        analyze_request['spanAnnotations'] = True
        try:
            if(service is None):
                response = self.service1.comments().analyze(body=analyze_request).execute()
            else:
                response = service.comments().analyze(body=analyze_request).execute()
            value = response['attributeScores']['TOXICITY']['summaryScore']['value']
            span_scores = response['attributeScores']['TOXICITY']['spanScores']
        except:
            value = None
            span_scores = None
        time.sleep(1)

        return value, span_scores


    def forbidden(self, token):

        if(token.lower() in STOP_WORDS):
            return True
        else:
            for character in token.lower():
                if(character in self.alphabet):
                    return False
            return True

        return False



    def get_sentence_toxicities(self, dialogues):

        sentence_toxicities = dict()
        for dialogue in tqdm(dialogues):
            if(dialogue['speaker'] not in sentence_toxicities):
                sentence_toxicities[dialogue['speaker']] = list()
            for sentence in self.nlp(dialogue['text']).sents:
                toxicity = self.get_toxicity(sentence.text)
                dialogue['toxicity'] = toxicity
                sentence_toxicities[dialogue['speaker']].append({'text': sentence.text, 'toxicity': toxicity})

        return sentence_toxicities, dialogues




    def get_dialogue_toxicities(self, dialogues):

        toxicities = dict()
        for dialogue in tqdm(dialogues):
            if(dialogue['speaker'] not in toxicities):
                toxicities[dialogue['speaker']] = list()
            toxicity = self.get_toxicity(dialogue['text'])
            dialogue['toxicity'] = toxicity
            toxicities[dialogue['speaker']].append({'text': dialogue['text'], 'toxicity': toxicity})

        return toxicities


