#!/usr/bin/env python
# Copyright 2022, Educational Testing Service

from transformers import BertTokenizer, BertForMaskedLM
import torch


class WordProbabilityInContext(object):

    tokenizer = None
    model = None
    sm = None

    def __init__(self):
        # init model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.model.eval()

        # init softmax to get probabilities later on
        self.sm = torch.nn.Softmax(dim=0)
        torch.set_grad_enabled(False)

    def make_context_string(self, prefix, suffix):
        return f'{prefix} {self.tokenizer.mask_token} {suffix}'

    def probabilityInContext(self, word, context):
        print(context)
        token_ids = self.tokenizer.encode(context, return_tensors='pt')
        masked_position = (token_ids.squeeze() ==
                           self.tokenizer.mask_token_id).nonzero().item()

        # forward
        output = self.model(token_ids)
        last_hidden_state = output[0].squeeze(0)

        # only get output for masked token
        # output is the size of the vocabulary
        mask_hidden_state = last_hidden_state[masked_position]

        # convert to probabilities (softmax)
        # giving a probability for each item in the vocabulary
        probs = self.sm(mask_hidden_state)
        test_id = self.tokenizer.convert_tokens_to_ids(word)
        return probs[test_id].item()


wpic = WordProbabilityInContext()

