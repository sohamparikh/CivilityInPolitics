import os
import re
import spacy
import pdftotext
from tqdm import tqdm
from IPython import embed

class PdfExtractor:



    def __init__(self):
        self.nlp = spacy.load('en')
        self.nlp_light = spacy.load('en', disable=['parser', 'tagger', 'ner'])
        self.commercial_sentinel = "(COMMERCIAL BREAK)"
        self.begin_clip = "(BEGIN VIDEO CLIP)"
        self.end_clip = "(END VIDEO CLIP)"
        self.load_names()

    def extract_single_show(self, filepath):

        with open(filepath, 'rb') as f:
            full_pdf = ""
            pdf = pdftotext.PDF(f)
            for page in pdf:
                full_pdf += " " + page

        full_pdf = full_pdf.replace("T ", "T")
        full_pdf = full_pdf.replace("BEGN", "BEGIN")
        full_pdf = full_pdf.replace("(CROSSTALK)", "")
        full_pdf = full_pdf.replace("(VIDEO CLIP PLAYS)", "")
        full_pdf = full_pdf.replace("(BEEP)", "")
        full_pdf = full_pdf.replace("(BEGIN VIDEO CLIP)", "(BEGIN VIDEO CLIP)\n")
        full_pdf = full_pdf.replace("(BEGIN VIDOE CLIP)", "(BEGIN VIDEO CLIP)\n")
        full_pdf = full_pdf.replace("(BEGIN VIDOEO CLIP)", "(BEGIN VIDEO CLIP)\n")
        full_pdf = full_pdf.replace("(END VIDEO CLIP)", "(END VIDEO CLIP)\n")
        full_pdf = full_pdf.replace("(COMMERCIAL BREAK)", "(COMMERCIAL BREAK)\n")
        full_pdf = full_pdf.replace("(BEGIN VIDEOTAPE)", "(BEGIN VIDEO CLIP)\n")
        full_pdf = full_pdf.replace("(BEGIN VIDEO CLIP,", "(BEGIN VIDEO CLIP)\n")

        parts = full_pdf.split(self.commercial_sentinel)
        cleaned_parts = list()
        for part in parts:
            cleaned_part = self.clean_part(part)
            cleaned_parts.append(cleaned_part)

        return cleaned_parts



    def clean_part(self, part):

        lines = part.split('\n')
        start_idxs = list()
        end_idxs = list()
        prev = None
        for idx, line in enumerate(lines):
            if(self.begin_clip in line.strip()):
                start_idx = idx
            elif(self.end_clip in line.strip()):
                end_idxs.append(idx)
                start_idxs.append(start_idx)

        dialogues = list()
        in_video = False
        curr_dialogue = ""
        curr_speaker = None
        for idx, line in enumerate(lines):
            if(line.startswith('Content and Prog ramming Copyrig ht 2019')):
                curr_speaker = None
                continue
            if(line.strip() == ""):
                continue
            if(idx in start_idxs):
                if(curr_dialogue.strip() != ""):
                    dialogues.append((curr_speaker, curr_dialogue, str(in_video)))
                curr_speaker = None
                curr_dialogue = ""
                in_video = True
            elif(idx in end_idxs):
                if(curr_dialogue.strip() != ""):
                    dialogues.append((curr_speaker, curr_dialogue, str(in_video)))
                curr_speaker = None
                curr_dialogue = ""
                in_video = False
            else:
                speaker, dialogue = self.get_speaker_and_dialogue(line)
                if(speaker is not None):
                    if(curr_dialogue.strip() != ""):
                        dialogues.append((curr_speaker, curr_dialogue, str(in_video)))
                    curr_speaker = speaker
                    curr_dialogue = dialogue
                elif(speaker is None and curr_speaker is None):
                    continue
                else:
                    curr_dialogue += " " + dialogue.replace('--', ' ').strip()

        if(curr_dialogue.strip() != ""):
            dialogues.append((curr_speaker, curr_dialogue, str(in_video)))

        return dialogues




    def get_speaker_and_dialogue(self, line):
        names = re.findall("[A-Z\s\(\)\-\.,'\"\n]+:", line.strip())
        if(len(names) > 0):
            name = names[0]
            dialogue = line.split(name)[1]
        else:
            name = None
            dialogue = line.strip()
        
        return name, dialogue


    def extract_all_shows(self, host):

        shows = list()
        for file in os.listdir(os.path.join('../data/pdfs', host)):
            current_show = self.extract_single_show(os.path.join('../data/pdfs', host, file))
            shows.append((file, current_show))

        return shows

    def get_all_sentences(self, shows):

        sentences = list()
        for show in tqdm(shows):
            for part in show[1]:
                for dialogue in part:
                    for sentence in self.nlp(dialogue[1]).sents:
                        sentences.append(sentence.text)

        return sentences

    def get_all_dialogues(self, shows):

        dialogues = list()
        for show in shows:
            for part in show[1]:
                for dialogue in part:
                    dialogues.append(dialogue[1])

        return dialogues

    def load_names(self):
        self.hannity_names = list()
        self.maddow_names = list()
        self.pbs_names = list()
        with open('../data/names/hannity.txt') as f:
            for line in f:
                for token in self.nlp_light(line.strip()):
                    self.hannity_names.append(token.text)
        with open('../data/names/maddow.txt') as f:
            for line in f:
                for token in self.nlp_light(line.strip()):
                    self.maddow_names.append(token.text)
        with open('../data/names/pbs.txt') as f:
            for line in f:
                for token in self.nlp_light(line.strip()):
                    self.pbs_names.append(token.text)


class DebateParser:


    def parse_file(self, filepath):

        dialogues = list()
        with open(filepath) as f:

            for line in f:
                if(line.strip() == ''):
                    continue
                dialogue = dict()
                splits = line.split(':')
                name = splits[0]
                if(self.hhmmss(splits)):
                    hour = int(splits[1].strip())
                    minute = int(splits[2].strip())
                    second = int(splits[3][:2])
                    text = ' '.join([splits[3][2:]] + [split for split in splits[4:]]).strip()
                else:
                    hour = 0
                    minute = int(splits[1].strip())
                    second = int(splits[2][:2])
                    text = ' '.join([splits[2][2:]] + [split for split in splits[4:]]).strip()
                dialogue['text'] = text
                dialogue['speaker'] = name
                dialogue['time'] = hour*3600 + minute*60 + second
                dialogues.append(dialogue)

        return dialogues

    def isint(self, text):

        try:
            int(text)
            return True
        except:
            return False

    def hhmmss(self, splits):

        if(len(splits) < 4):
            return False
        elif(len(splits[2].strip()) == 2 and self.isint(splits[3][:2])):
            return True
        else:
            return False


