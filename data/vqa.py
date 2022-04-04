"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

VQA dataset
"""
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip

from .data import TxtVideoAudioDataset, TxtMapper
import json
import os
import string 
punctuation = string.punctuation
from pytorch_pretrained_bert import BertTokenizer
import random
import copy


class TxtMapperForOpenEndedVQA(TxtMapper):
    def __init__(self, txt_dir, answer_candidate_path, max_txt_len=30):
        super().__init__(txt_dir, max_txt_len)
        self.answer_candidate = json.load(open(answer_candidate_path))
    
    def __getitem__(self, id_, return_all=False):
        qa_pairs = self.json_dict[id_]
        if not return_all: ####training
            if isinstance(qa_pairs,list):
                while True:
                    sample = random.choice(qa_pairs)
                    if sample['answer'] in self.answer_candidate:
                        break
            else:
                assert NotImplementedError
            question_tokens , answer = self.process_sample(sample)
            return question_tokens, answer
        else:           ###testing
            question_tokens = []
            answer = []
            for sample in qa_pairs:
                q, a = self.process_sample(sample,answer_mapping=False)
                question_tokens.append(q)
                answer.append(a)
            return question_tokens, answer

    def process_sample(self, sample, answer_mapping=True):
        question = sample['question']
        answer = sample['answer']
        """process_question"""
        question = self.clean(question) 
        question_tokens = self.tokenize(question)
        question_tokens = question_tokens[:self.max_txt_len]
        #question_tokens = torch.tensor(question_tokens, dtype=torch.long)
        question_tokens = self.get_padded_tokens(question_tokens)
        """process_answer"""
        
        if answer_mapping:
            answer = self.answer_candidate[answer]
        return question_tokens, answer







class OpenEndedVQADataset(TxtVideoAudioDataset):
    def __init__(self, ids_path, txt_mapper, video_mapper, audio_mapper, training=True):
        super().__init__(ids_path, txt_mapper, video_mapper,audio_mapper)
        self.training = training
    def __getitem__(self, i):
        id_ = self.ids[i]
        video_pixels = self.video_mapper[id_]
        audio_spectrograms = self.audio_mapper[id_]
        if audio_spectrograms is None: ### current id doesn't have audio stream
            pass
        if self.training:
            question_tokens, answer = self.txt_mapper.__getitem__(id_,return_all=False)
        else:
            question_tokens, answer = self.txt_mapper.__getitem__(id_,return_all=True)
            num_samples = len(answer)
            video_pixels = [video_pixels for i in range(num_samples)]
            audio_spectrograms = [audio_spectrograms for i in range(num_samples)]
            id_ = [id_ for i in range(num_samples)]
        return id_, question_tokens, answer, video_pixels, audio_spectrograms
    

def concat_list(input_list):
    return [k  for i in input_list for k in i]

def openendedvqa_collate(inputs):
    (ids, question_tokens, answers, video_pixels, audio_spectrograms) = map(list, unzip(inputs))
    if isinstance(question_tokens[0],list): ###testing
        ids =  concat_list(ids)
        question_tokens = concat_list(question_tokens)
        answers = concat_list(answers)
        video_pixels = concat_list(video_pixels)
        audio_spectrograms = concat_list(audio_spectrograms)


    ids_2m = []
    question_tokens_2m = []
    video_pixels_2m = []
    answers_2m = []

    ids_3m = []
    question_tokens_3m = []
    video_pixels_3m = []
    audio_spectrograms_3m = []
    answers_3m = []
    for i in range(len(audio_spectrograms)):
        if audio_spectrograms[i] is not None:
            ids_3m.append(ids[i])
            question_tokens_3m.append(question_tokens[i])
            video_pixels_3m.append(video_pixels[i])
            audio_spectrograms_3m.append(audio_spectrograms[i])
            answers_3m.append(answers[i])
        else:
            ids_2m.append(ids[i])
            question_tokens_2m.append(question_tokens[i])
            video_pixels_2m.append(video_pixels[i])
            answers_2m.append(answers[i])

    if ids_3m != []:
        question_tokens_3m = torch.stack(question_tokens_3m, dim=0)
        video_pixels_3m = torch.stack(video_pixels_3m, dim=0)
        audio_spectrograms_3m = torch.stack(audio_spectrograms_3m, dim=0)
        if isinstance(answers_3m[0],int):  
            answers_3m = torch.tensor(answers_3m , dtype=torch.long)
    if ids_2m != []:
        question_tokens_2m = torch.stack(question_tokens_2m, dim=0)
        video_pixels_2m = torch.stack(video_pixels_2m, dim=0)
        if isinstance(answers_2m[0],int):  
            answers_2m = torch.tensor(answers_2m , dtype=torch.long)
    
    batch_2m =   {'ids': ids_2m,
             'txt_tokens': question_tokens_2m,
             'video_pixels': video_pixels_2m,
             'answers':answers_2m}
    
    
    batch_3m =   {'ids': ids_3m,
             'txt_tokens': question_tokens_3m,
             'video_pixels': video_pixels_3m,
             'audio_spectrograms': audio_spectrograms_3m,
            'answers':answers_3m}

    batch={'batch_2m':batch_2m,
            'batch_3m':batch_3m}
    

    return batch
    







class TxtMapperForMultipleChoiceVQA(TxtMapper):
    def __init__(self, txt_dir, max_txt_len=30):
        super().__init__(txt_dir, max_txt_len)
    
    def __getitem__(self, id_):
        sample = self.json_dict[id_]
        candidate_tokens, answer = self.process_sample(sample)
        
        return candidate_tokens, answer

    def process_sample(self, sample):
        candidate = sample['candidate']
        answer = sample['answer']
        """process_question"""
        candidate_tokens = []
        for  c in candidate:
            c = self.clean(c) 
            tokens = self.tokenize(c)
            tokens = self.get_padded_tokens(tokens)
            candidate_tokens.append(tokens)
        """process_answer"""
        
        return candidate_tokens, answer



# class MultipleChoiceVQADataset(TxtVideoDataset):   ### only used in testing
#     def __init__(self, ids_path, txt_mapper, video_mapper):
#         super().__init__(ids_path, txt_mapper, video_mapper)

#     def __getitem__(self, i):
#         id_ = self.ids[i]
#         video_pixels = self.video_mapper[id_]
#         candidate_tokens, answer = self.txt_mapper[id_]
#         video_pixels = [copy.deepcopy(video_pixels) for i in range(len(candidate_tokens))]
#         return candidate_tokens, answer, video_pixels

# def multiplechoicevqa_collate(inputs):
#     (candidate_tokens, answers, video_pixels) = map(list, unzip(inputs))
#     candidate_tokens = concat_list(candidate_tokens)
#     video_pixels = concat_list(video_pixels)

#     txt_tokens = torch.stack(candidate_tokens,dim=0)
#     video_pixels = torch.stack(video_pixels,dim=0)

#     #attn_masks = torch.cat((attn_masks_txt,attn_masks_video), dim = 1)

#     batch = {'txt_tokens': txt_tokens,
#              'video_pixels': video_pixels,                
#              'answers': answers}
#     return batch