#!/usr/bin/env python3.10
# Copyright 2022, Educational Testing Service

import math
import numpy as np
import os
import re
import srsly
import wordfreq
import imp
import statistics
import awe_lexica

from spacy.tokens import Doc, Span, Token
from spacy.language import Language
from spacy.vocab import Vocab

from scipy.spatial.distance import cosine
# Standard cosine distance metric

from nltk.corpus import wordnet
# English dictionary. Contains information on senses associated with words
# (a lot more, but that's what we're currently using it for)

import wordfreq
# https://github.com/rspeer/wordfreq
# Large word frequency database. Provides Zipf frequencies
# (log scale of frequency) for most English words, based on a
# variety of corpora.

from .utility_functions import *
from ..errors import *
from importlib import resources


@Language.factory("lexicalfeatures")
def LexicalFeatures(nlp, name):
    ldf = LexicalFeatureDef()
    ldf.set_nlp(nlp)
    ldf.load_lexicons()
    ldf.add_extensions()
    return ldf


class LexicalFeatureDef(object):

    SYLLABLES_PATH = \
        resources.path('awe_lexica.json_data',
                           'syllables.json')

    ROOTS_PATH = \
        resources.path('awe_lexica.json_data',
                       'roots.json')

    FAMILY_SIZES_PATH = \
        resources.path('awe_lexica.json_data',
                       'family_sizes.json')

    FAMILY_MAX_FREQS_PATH = \
        resources.path('awe_lexica.json_data',
                       'family_max_freqs.json')

    FAMILY_IDX_PATH = \
        resources.path('awe_lexica.json_data',
                       'family_idxs.json')

    FAMILY_LISTS_PATH = \
        resources.path('awe_lexica.json_data',
                       'family_lists.json')

    CONCRETES_PATH = \
        resources.path('awe_lexica.json_data',
                       'concretes.json')

    MORPHOLEX_PATH = \
        resources.path('awe_lexica.json_data',
                       'morpholex.json')

    LATINATE_PATH = \
        resources.path('awe_lexica.json_data',
                       'latinate.json')

    ACADEMIC_PATH = \
        resources.path('awe_lexica.json_data',
                       'academic.json')

    NMORPH_STATUS_PATH = \
        resources.path('awe_lexica.json_data',
                       'nMorph_status.json')

    ACADEMIC_PATH = \
        resources.path('awe_lexica.json_data',
                       'academic.json')

    SENTIMENT_PATH = \
        resources.path('awe_lexica.json_data',
                       'sentiment.json')

    nlp = None

    syllables = {}
    roots = {}
    family_sizes = {}
    family_max_freqs = {}
    family_ids = {}
    family_lists = {}
    concretes = {}
    morpholex = {}
    latinate = {}
    nMorph_status = {}
    sentiment = {}
    academic = []
    animateNouns = {}
    abstractTraitNouns = {}

    content_tags = ['NN',
                    'NNS',
                    'NNP',
                    'NNPS',
                    'VB',
                    'VBD',
                    'VBG',
                    'VBN',
                    'VBP',
                    'VBZ',
                    'JJ',
                    'JJR',
                    'JJS',
                    'RB',
                    'RBR',
                    'RBS',
                    'RP',
                    'GW',
                    'NOUN',
                    'PROPN',
                    'VERB',
                    'ADJ',
                    'ADV',
                    'CD']

    def set_nlp(self, nlpIn):
        self.nlp = nlpIn

    def package_check(self, lang):

        if not os.path.exists(self.SYLLABLES_PATH):
            raise LexiconMissingError(
                "Trying to load AWE Workbench Lexicon Module \
                without Syllables datafile".format(lang)
            )
        if not os.path.exists(self.ROOTS_PATH):
            raise LexiconMissingError(
                "Trying to load AWE Workbench Lexicon Module \
                without Roots datafile".format(lang)
            )
        if not os.path.exists(self.FAMILY_SIZES_PATH):
            raise LexiconMissingError(
                "Trying to load AWE Workbench Lexicon Module \
                without Word Family Size datafile".format(lang)
            )
        if not os.path.exists(self.FAMILY_MAX_FREQS_PATH):
            raise LexiconMissingError(
                "Trying to load AWE Workbench Lexicon Module \
                without Word Family Max Size datafile".format(lang)
            )
        if not os.path.exists(self.FAMILY_IDX_PATH):
            raise LexiconMissingError(
                "Trying to load AWE Workbench Lexicon Module \
                without Word Family Max Size datafile".format(lang)
            )
        if not os.path.exists(self.FAMILY_LISTS_PATH):
            raise LexiconMissingError(
                "Trying to load AWE Workbench Lexicon Module \
                without Word Family Max Size datafile".format(lang)
            )
        if not os.path.exists(self.CONCRETES_PATH):
            raise LexiconMissingError(
                "Trying to load AWE Workbench Lexicon Module \
                without Concretes datafile".format(lang)
            )
        if not os.path.exists(self.MORPHOLEX_PATH):
            raise LexiconMissingError(
                "Trying to load AWE Workbench Lexicon Module \
                without Morpholex datafile".format(lang)
            )
        if not os.path.exists(self.LATINATE_PATH):
            raise LexiconMissingError(
                "Trying to load AWE Workbench Lexicon Module \
                without Latinates datafile".format(lang)
            )
        if not os.path.exists(self.NMORPH_STATUS_PATH):
            raise LexiconMissingError(
                "Trying to load AWE Workbench Lexicon Module \
                without NMorph_Status datafile".format(lang)
            )
        if not os.path.exists(self.SENTIMENT_PATH):
            raise LexiconMissingError(
                "Trying to load AWE Workbench Lexicon Module \
                without Sentiment datafile".format(lang)
            )
        if not os.path.exists(self.ACADEMIC_PATH):
            raise LexiconMissingError(
                "Trying to load AWE Workbench Lexicon Module \
                without Academic Language datafile".format(lang)
            )

    def load_lexicons(self):

        # To save memory, use the spacy string hash as key,
        # not the actual text string
        temp = srsly.read_json(self.SYLLABLES_PATH)
        for word in temp:
            if word in self.nlp.vocab.strings:
                key = self.nlp.vocab.strings[word]
            else:
                key = self.nlp.vocab.strings.add(word)
            self.syllables[key] = temp[word]

        temp = srsly.read_json(self.ROOTS_PATH)
        for word in temp:
            if word in self.nlp.vocab.strings:
                key = self.nlp.vocab.strings[word]
            else:
                key = self.nlp.vocab.strings.add(word)
            self.roots[key] = temp[word]

        temp = srsly.read_json(self.FAMILY_SIZES_PATH)
        for word in temp:
            if word in self.nlp.vocab.strings:
                key = self.nlp.vocab.strings[word]
            else:
                key = self.nlp.vocab.strings.add(word)
            self.family_sizes[key] = temp[word]

        temp = srsly.read_json(self.FAMILY_MAX_FREQS_PATH)
        for word in temp:
            if word in self.nlp.vocab.strings:
                key = self.nlp.vocab.strings[word]
            else:
                key = self.nlp.vocab.strings.add(word)
            self.family_max_freqs[key] = temp[word]

        temp = srsly.read_json(self.FAMILY_IDX_PATH)
        for word in temp:
            if word in self.nlp.vocab.strings:
                key = self.nlp.vocab.strings[word]
            else:
                key = self.nlp.vocab.strings.add(word)
            self.family_ids[key] = temp[word]

        temp = srsly.read_json(self.FAMILY_LISTS_PATH)
        for family in temp:
            self.family_lists[family] = temp[family]

        temp = srsly.read_json(self.CONCRETES_PATH)
        for word in temp:
            if word in self.nlp.vocab.strings:
                key = self.nlp.vocab.strings[word]
            else:
                key = self.nlp.vocab.strings[word]
            if key not in self.concretes:
                self.concretes[key] = {}
            for POS in temp[word]:
                self.concretes[key][POS] = temp[word][POS]

        temp = srsly.read_json(self.MORPHOLEX_PATH)

        for word in temp:
            if word in self.nlp.vocab.strings:
                key = self.nlp.vocab.strings[word]
            else:
                key = self.nlp.vocab.strings.add(word)
            self.morpholex[key] = temp[word]

        temp = srsly.read_json(self.LATINATE_PATH)
        for word in temp:
            if word in self.nlp.vocab.strings:
                key = self.nlp.vocab.strings[word]
            else:
                key = self.nlp.vocab.strings.add(word)
            self.latinate[key] = temp[word]

        temp = srsly.read_json(self.NMORPH_STATUS_PATH)
        for word in temp:
            if word in self.nlp.vocab.strings:
                key = self.nlp.vocab.strings[word]
            else:
                key = self.nlp.vocab.strings.add(word)
            self.nMorph_status[key] = temp[word]

        temp = srsly.read_json(self.SENTIMENT_PATH)
        for word in temp:
            if word in self.nlp.vocab.strings:
                key = self.nlp.vocab.strings[word]
            else:
                key = self.nlp.vocab.strings.add(word)
            self.sentiment[key] = temp[word]
        for word in temp:
            if word in self.nlp.vocab.strings:
                key = self.nlp.vocab.strings[word]
            else:
                key = self.nlp.vocab.strings.add(word)
            sentlist = []

            # modify the sentiment estimate using word families, but only if
            # no negative prefix or suffixes are involved in the word we are
            # taking the sentiment rating from, and it's not in the very high
            # frequency band.
            if key in self.family_ids \
               and str(self.family_ids[key]) in self.family_lists:
                for item in self.family_lists[str(self.family_ids[key])]:
                    if item in self.nlp.vocab.strings:
                        itemkey = self.nlp.vocab.strings[item]
                    else:
                        itemkey = self.nlp.vocab.strings.add(item)
                    if itemkey in self.sentiment \
                        and wordfreq.zipf_frequency(item, "en") < 5 \
                        and wordfreq.zipf_frequency(word, "en") < 5 \
                        and len(item) > 4 and len(word) > 4 \
                        and (len(item) < len(word)
                             or (not item.endswith('less')
                                 and not item.endswith('lessness')
                                 and not item.endswith('lessly')
                                 and not item.startswith('un')
                                 and not item.startswith('in')
                                 and not item.startswith('im')
                                 and not item.startswith('dis')
                                 and not item.startswith('mis')
                                 and not item.startswith('anti'))):
                        sentlist.append(self.sentiment[itemkey])

            if key not in self.sentiment or abs(self.sentiment[key]) <= .2:
                if len(sentlist) > 1 and statistics.mean(sentlist) > 0:
                    self.sentiment[key] = max(sentlist)
                elif len(sentlist) > 1 and statistics.mean(sentlist) < 0:
                    self.sentiment[key] = min(sentlist)
                elif len(sentlist) == 1:
                    self.sentiment[key] = sentlist[0]

            if abs(self.sentiment[key]) <= .2 \
               and key in self.roots \
               and wordfreq.zipf_frequency(word, "en") < 5:

                if self.roots[key] in self.nlp.vocab.strings:
                    rootkey = self.nlp.vocab.strings[self.roots[key]]
                else:
                    rootkey = self.nlp.vocab.strings.add(self.roots[key])
                if key != rootkey and rootkey in self.roots \
                   and rootkey in self.sentiment \
                   and abs(self.sentiment[rootkey]) > .2:
                    self.sentiment[key] = self.sentiment[rootkey]

        temp = srsly.read_json(self.ACADEMIC_PATH)
        for word in temp:
            if word in self.nlp.vocab.strings:
                key = self.nlp.vocab.strings[word]
            else:
                key = self.nlp.vocab.strings.add(word)
            self.academic.append(key)

    def __call__(self, doc):
        # We're using this component as a wrapper to add access
        # to the lexical features. There is no actual parsing of the
        # sentences except for handling of a few special cases for
        # specific features.

        return doc

    def __init__(self, lang="en"):
        super().__init__()
        self.package_check(lang)

    ###############################################
    # Block where we define getter functions used #
    # by spacy attribute definitions.             #
    ###############################################

    def lems(self, tokens):
        return [t.lemma_
                if self.alphanum_word(t.text)
                else None
                for t in tokens]

    def typs(self, tokens):
        return sorted(list(set([t.orth_ for t in tokens
                      if self.alphanum_word(t.text)])))

    def rt(self, token):
        if (token.text.lower() in self.nlp.vocab.strings
            and self.alphanum_word(token.text)
            and self.nlp.vocab.strings[token.text.lower()]
                in self.roots):
            return self.roots[
                self.nlp.vocab.strings[
                    token.text.lower()]]
        else:
            return None

    def mrts(self, tokens):
        return [self.roots[self.nlp.vocab.strings[token.text.lower()]]
                if (token.text.lower() in self.nlp.vocab.strings
                    and self.alphanum_word(token.text)
                    and self.nlp.vocab.strings[token.text.lower()]
                    in self.roots)
                else token.lemma_
                if self.alphanum_word(token.text)
                else None
                for token in tokens]

    def typ(self, tokens):
        return len(np.unique([t._.root for t in tokens
                   if not t.is_stop
                   and t._.root is not None
                   and t.pos_ in self.content_tags
                   and self.alphanum_word(t.text)]))

    def lemc(self, tokens):
        return len(np.unique([t.lemma_ for t in tokens
                   if not t.is_stop
                   and t.lemma_ is not None
                   and self.alphanum_word(t.text)
                   and t.pos_ in self.content_tags]))

    def typc(self, tokens):
        return len(np.unique([t.text.lower() for t in tokens
                   if not t.is_stop
                   and t.text is not None
                   and t.pos_ in self.content_tags
                   and self.alphanum_word(t.text)]))

    def tokc(self, tokens):
        return len([t.text.lower() for t in tokens
                   if not t.is_stop
                   and t.text is not None
                   and t.pos_ in self.content_tags
                   and self.alphanum_word(t.text)])

    def ns(self, token):
        if (token.text.lower() in self.nlp.vocab.strings
            and self.nlp.vocab.strings[token.text.lower()]
                in self.syllables):
            return self.syllables[
                self.nlp.vocab.strings[token.text.lower()]]
        else:
            return self.sylco(token.text.lower())

    def sylls(self, tokens):
        return [self.ns(token) for token in tokens]

    def mns(self, tokens):
        return summarize(lexFeat(tokens, 'nSyll'),
                         summaryType=FType.MEAN)

    def mdns(self, tokens):
        return summarize(lexFeat(tokens, 'nSyll'),
                         summaryType=FType.MEDIAN)

    def mxns(self, tokens):
        return summarize(lexFeat(tokens, 'nSyll'),
                         summaryType=FType.MAX)

    def minns(self, tokens):
        return summarize(lexFeat(tokens, 'nSyll'),
                         summaryType=FType.MIN)

    def stdns(self, tokens):
        return summarize(lexFeat(tokens, 'nSyll'),
                         summaryType=FType.STDEV)

    def nc(self, token):
        return math.sqrt(len(token.text))

    def chars(self, tokens):
        return [token._.sqrtNChars for token in tokens]

    def mnc(self, tokens):
        return summarize(lexFeat(tokens, 'sqrtNChars'),
                         summaryType=FType.MEAN)

    def mdnc(self, tokens):
        return summarize(lexFeat(tokens, 'sqrtNChars'),
                         summaryType=FType.MEDIAN)

    def mxnc(self, tokens):
        return summarize(lexFeat(tokens, 'sqrtNChars'),
                         summaryType=FType.MAX)

    def minnc(self, tokens):
        return summarize(lexFeat(tokens, 'sqrtNChars'),
                         summaryType=FType.MIN)

    def stdnc(self, tokens):
        return summarize(lexFeat(tokens, 'sqrtNChars'),
                         summaryType=FType.STDEV)

    def lats(self, tokens):
        return [token._.is_latinate for token in tokens]

    def mnlat(self, tokens):
        return summarize(lexFeat(tokens, 'is_latinate'),
                         summaryType=FType.MEAN)

    def acads(self, tokens):
        return [token._.is_academic for token in tokens]

    def mnacad(self, tokens):
        return summarize(lexFeat(tokens, 'is_academic'),
                         summaryType=FType.MEAN)

    def fmf(self, tokens):
        return [token._.max_freq for token in tokens]

    def fms(self, token):
        if self.nlp.vocab.strings[token.text.lower()] in self.family_sizes \
           and self.alphanum_word(token.text):
            return self.family_sizes[self.nlp.vocab.strings[token.text.lower()]]
        else:
            return None

    def fmss(self, tokens):
        return [token._.family_size for token in tokens]

    def mnfms(self, tokens):
        return summarize(lexFeat(tokens, 'family_size'),
                         summaryType=FType.MEAN)

    def mdfms(self, tokens):
        return summarize(lexFeat(tokens, 'family_size'),
                         summaryType=FType.MEDIAN)

    def mxfms(self, tokens):
        return summarize(lexFeat(tokens, 'family_size'),
                         summaryType=FType.MAX)

    def minfms(self, tokens):
        return summarize(lexFeat(tokens, 'family_size'),
                         summaryType=FType.MIN)

    def stdfms(self, tokens):
        return summarize(lexFeat(tokens, 'family_size'),
                         summaryType=FType.STDEV)

    def nsem(self, token):
        if self.alphanum_word(token.text) \
           and len(wordnet.synsets(token.lemma_)) > 0:
            return len(wordnet.synsets(token.lemma_))
        else:
            return None

    def lognsem(self, token):
        if self.alphanum_word(token.text) \
           and len(wordnet.synsets(token.lemma_)) > 0:
            return math.log(len(wordnet.synsets(token.lemma_)))
        else:
            return None

    def senseno(self, tokens):
        return [token._.nSenses for token in tokens]

    def logsenseno(self, tokens):
        return [token._.logNSenses for token in tokens]

    def mnsense(self, tokens):
        return summarize(lexFeat(tokens, 'nSenses'),
                         summaryType=FType.MEAN)

    def mdsense(self, tokens):
        return summarize(lexFeat(tokens, 'nSenses'),
                         summaryType=FType.MEDIAN)

    def mxsense(self, tokens):
        return summarize(lexFeat(tokens, 'nSenses'),
                         summaryType=FType.MAX)

    def minsense(self, tokens):
        return summarize(lexFeat(tokens, 'nSenses'),
                         summaryType=FType.MIN)

    def stdsense(self, tokens):
        return summarize(lexFeat(tokens, 'nSenses'),
                         summaryType=FType.STDEV)

    def mnlognsense(self, tokens):
        return summarize(lexFeat(tokens, 'logNSenses'),
                         summaryType=FType.MEAN)

    def mdlognsense(self, tokens):
        return summarize(lexFeat(tokens, 'logNSenses'),
                         summaryType=FType.MEDIAN)

    def mxlognsense(self, tokens):
        return summarize(lexFeat(tokens, 'logNSenses'),
                         summaryType=FType.MAX)

    def minlognsense(self, tokens):
        return summarize(lexFeat(tokens, 'logNSenses'),
                         summaryType=FType.MIN)

    def stdlognsense(self, tokens):
        return summarize(lexFeat(tokens, 'logNSenses'),
                         summaryType=FType.STDEV)

    def morpho(self, tokens):
        return [self.morpholex[self.nlp.vocab.strings[token.lemma_]]
                if token.lemma_ is not None
                and (token.lemma_ in self.nlp.vocab.strings
                and self.nlp.vocab.strings[token.lemma_]
                in self.morpholex)
                else None for token in tokens]

    def morpholexsegm(self, token):
        if (token.text is not None
            and self.nlp.vocab.strings[token.text.lower()]
                in self.morpholex):
            return self.morpholex[
                self.nlp.vocab.strings[
                    token.text.lower()]]['MorphoLexSegm']
        else:
            return None

    def morpholexsegms(self, tokens):
        return [token._.morpholexsegm for token in tokens]

    def nm(self, token):
        if token.text is not None \
           and token.text.lower() in self.nlp.vocab.strings \
           and self.alphanum_word(token.text) \
           and self.nlp.vocab.strings[token.text.lower()] \
           in self.nMorph_status:
            return self.nMorph_status[
                self.nlp.vocab.strings[token.text.lower()]]
        else:
            return None

    def morphn(self, tokens):
        return [token._.nMorph for token in tokens]

    def mnmorph(self, tokens):
        return summarize(lexFeat(tokens, 'nMorph'),
                         summaryType=FType.MEAN)

    def mdmorph(self, tokens):
        return summarize(lexFeat(tokens, 'nMorph'),
                         summaryType=FType.MEDIAN)

    def mxmorph(self, tokens):
        return summarize(lexFeat(tokens, 'nMorph'),
                         summaryType=FType.MAX)

    def minmorph(self, tokens):
        return summarize(lexFeat(tokens, 'nMorph'),
                         summaryType=FType.MIN)

    def stdmorph(self, tokens):
        return summarize(lexFeat(tokens, 'nMorph'),
                         summaryType=FType.STDEV)

    def rfqh(self, token):
        if (token.lemma_ is not None
            and token.lemma_ in self.nlp.vocab.strings
            and self.alphanum_word(token.lemma_)
            and self.nlp.vocab.strings[token.lemma_]
                in self.morpholex):
            return self.morpholex[
               self.nlp.vocab.strings[
                   token.lemma_]]['ROOT1_Freq_HAL']
        else:
            return None

    def rfqh2(self, token):
        if token.lemma_ is not None \
           and token.lemma_ in self.nlp.vocab.strings \
           and self.alphanum_word(token.lemma_) \
           and self.nlp.vocab.strings[token.lemma_] \
                in self.morpholex \
           and 'ROOT2_Freq_HAL' in \
               self.morpholex[
                   self.nlp.vocab.strings[token.lemma_]]:
                return self.morpholex[
                    self.nlp.vocab.strings[
                        token.lemma_]]['ROOT2_Freq_HAL']
        else:
            return None

    def rfqh3(self, token):
        if (token.lemma_ is not None
            and token.lemma_ in self.nlp.vocab.strings
            and self.alphanum_word(token.lemma_)
            and self.nlp.vocab.strings[token.lemma_]
                in self.morpholex) \
            and 'ROOT3_Freq_HAL' \
                in self.morpholex[self.nlp.vocab.strings[
                                  token.lemma_]]:
            return self.morpholex[
                self.nlp.vocab.strings[
                    token.lemma_]]['ROOT3_Freq_HAL']
        else:
            return None

    def rfsh(self, tokens):
        retlist = []
        for token in tokens:
            if self.min_root_freq(token) is not None:
                retlist.append(self.min_root_freq(token))
            else:
                retlist.append(None)
        return retlist

    def mnfrh(self, tokens):
        return summarize(tokens._.root_freqs_HAL,
                         summaryType=FType.MEAN)

    def mdfrh(self, tokens):
        return summarize(tokens._.root_freqs_HAL,
                         summaryType=FType.MEDIAN)

    def mxfrh(self, tokens):
        return summarize(tokens._.root_freqs_HAL,
                         summaryType=FType.MAX)

    def minfrh(self, tokens):
        return summarize(tokens._.root_freqs_HAL,
                         summaryType=FType.MIN)

    def stdfrh(self, tokens):
        return summarize(tokens._.root_freqs_HAL,
                         summaryType=FType.STDEV)

    def rfshlg(self, tokens):
        retlist = []
        for token in tokens:
            if self.min_root_freq(token) is not None \
               and self.min_root_freq(token) > 0:
                retlist.append(math.log(self.min_root_freq(token)))
            else:
                retlist.append(None)
        return retlist

    def mnlgfrh(self, tokens):
        return summarize(tokens._.log_root_freqs_HAL,
                         summaryType=FType.MEAN)

    def mdlgfrh(self, tokens):
        return summarize(tokens._.log_root_freqs_HAL,
                         summaryType=FType.MEDIAN)

    def mxlgfrh(self, tokens):
        return summarize(tokens._.log_root_freqs_HAL,
                         summaryType=FType.MAX)

    def minlgfrh(self, tokens):
        return summarize(tokens._.log_root_freqs_HAL,
                         summaryType=FType.MIN)

    def stdlgfrh(self, tokens):
        return summarize(tokens._.log_root_freqs_HAL,
                         summaryType=FType.STDEV)

    def rfs(self, token):
        if (token.lemma_ is not None
            and token.lemma_ in self.nlp.vocab.strings
            and self.alphanum_word(token.lemma_)
            and self.nlp.vocab.strings[token.lemma_]
                in self.morpholex):
            return self.morpholex[
                self.nlp.vocab.strings[
                    token.lemma_]]['ROOT1_FamSize']
        else:
            return None

    def rfsz(self, tokens):
        return [token._.root_famSize for token in tokens]

    def mnrfsz(self, tokens):
        return summarize(lexFeat(tokens, 'root_famSize'),
                         summaryType=FType.MEAN)

    def mdrfsz(self, tokens):
        return summarize(lexFeat(tokens, 'root_famSize'),
                         summaryType=FType.MEDIAN)

    def mxrfsz(self, tokens):
        return summarize(lexFeat(tokens, 'root_famSize'),
                         summaryType=FType.MAX)

    def minrfsz(self, tokens):
        return summarize(lexFeat(tokens, 'root_famSize'),
                         summaryType=FType.MIN)

    def stdrfsz(self, tokens):
        return summarize(lexFeat(tokens, 'root_famSize'),
                         summaryType=FType.STDEV)

    def rpfmf(self, token):
        if token.lemma_ is not None \
           and token.lemma_ in self.nlp.vocab.strings \
           and self.alphanum_word(token.lemma_) \
           and self.nlp.vocab.strings[token.lemma_] \
                in self.morpholex:
            return self.morpholex[
                self.nlp.vocab.strings[
                    token.lemma_]]['ROOT1_PFMF']
        else:
            return None

    def rfszrt(tokens):
        return [token._.root_pfmf for token in tokens]

    def rfszrt(self, tokens):
        return [token._.root_pfmf for token in tokens]

    def mnrpfmf(self, tokens):
        return summarize(lexFeat(tokens, 'root_pfmf'),
                         summaryType=FType.MEAN)

    def mdrpfmf(self, tokens):
        return summarize(lexFeat(tokens, 'root_pfmf'),
                         summaryType=FType.MEDIAN)

    def mxrpfmf(self, tokens):
        return summarize(lexFeat(tokens, 'root_pfmf'),
                         summaryType=FType.MAX)

    def minrpfmf(self, tokens):
        return summarize(lexFeat(tokens, 'root_pfmf'),
                         summaryType=FType.MIN)

    def stdrpfmf(self, tokens):
        return summarize(lexFeat(tokens, 'root_pfmf'),
                         summaryType=FType.STDEV)

    def tf(self, token):
        if self.alphanum_word(token.text):
            return wordfreq.zipf_frequency(token.text.lower(), "en")
        else:
            return None

    def lf(self, token):
        if self.alphanum_word(token.lemma_):
            return wordfreq.zipf_frequency(token.lemma_, "en")
        else:
            return None

    def zrf(self, token):
        if token._.root is not None \
           and self.alphanum_word(token._.root):
            return wordfreq.zipf_frequency(token._.root, "en")
        else:
            return None

    def mff(self, token):
        if self.nlp.vocab.strings[token.text.lower()] in self.roots \
           and self.roots[self.nlp.vocab.strings[token.text.lower()]] \
           in self.family_max_freqs:
            return self.family_max_freqs[
                self.roots[self.nlp.vocab.strings[
                    token.text.lower()]]]
        else:
            return wordfreq.zipf_frequency(token.lemma_, "en")

    def tkfrq(self, tokens):
        return [token._.token_freq for token in tokens]

    def lmfrqs(self, tokens):
        return [token._.lemma_freq for token in tokens]

    def rtfrqs(self, tokens):
        return [token._.root_freq for token in tokens]

    def fmf(self, tokens):
        return [token._.max_freq for token in tokens]

    def mnfrq(self, tokens):
        return summarize(lexFeat(tokens, 'token_freq'),
                         summaryType=FType.MEAN)

    def mdfrq(self, tokens):
        return summarize(lexFeat(tokens, 'token_freq'),
                         summaryType=FType.MEDIAN)

    def mxfrq(self, tokens):
        return summarize(lexFeat(tokens, 'token_freq'),
                         summaryType=FType.MAX)

    def minfrq(self, tokens):
        return summarize(lexFeat(tokens, 'token_freq'),
                         summaryType=FType.MIN)

    def stdfrq(self, tokens):
        return summarize(lexFeat(tokens, 'token_freq'),
                         summaryType=FType.STDEV)

    def mnlmfrq(self, tokens):
        return summarize(lexFeat(tokens, 'lemma_freq'),
                         summaryType=FType.MEAN)

    def mdlmfrq(self, tokens):
        return summarize(lexFeat(tokens, 'lemma_freq'),
                         summaryType=FType.MEDIAN)

    def mxlmfrq(self, tokens):
        return summarize(lexFeat(tokens, 'lemma_freq'),
                         summaryType=FType.MAX)

    def minlmfrq(self, tokens):
        return summarize(lexFeat(tokens, 'lemma_freq'),
                         summaryType=FType.MIN)

    def stdlmfrq(self, tokens):
        return summarize(lexFeat(tokens, 'lemma_freq'),
                         summaryType=FType.STDEV)

    def mnrtfrq(self, tokens):
        return summarize(lexFeat(tokens, 'max_freq'),
                         summaryType=FType.MEAN)

    def mdrtfrq(self, tokens):
        return summarize(lexFeat(tokens, 'max_freq'),
                         summaryType=FType.MEDIAN)

    def mxrtfrq(self, tokens):
        return summarize(lexFeat(tokens, 'max_freq'),
                         summaryType=FType.MAX)

    def minrtfrq(self, tokens):
        return summarize(lexFeat(tokens, 'max_freq'),
                         summaryType=FType.MIN)

    def stdrtfrq(self, tokens):
        return summarize(lexFeat(tokens, 'max_freq'),
                         summaryType=FType.STDEV)

    def concrs(self, tokens):
        return [self.concreteness(token) for token in tokens]

    def mncr(self, tokens):
        return summarize(lexFeat(tokens, 'concreteness'),
                         summaryType=FType.MEAN)

    def mdcr(self, tokens):
        return summarize(lexFeat(tokens, 'concreteness'),
                         summaryType=FType.MEDIAN)

    def mxcr(self, tokens):
        return summarize(lexFeat(tokens, 'concreteness'),
                         summaryType=FType.MAX)

    def mincr(self, tokens):
        return summarize(lexFeat(tokens, 'concreteness'),
                         summaryType=FType.MIN)

    def stdcr(self, tokens):
        return summarize(lexFeat(tokens, 'concreteness'),
                         summaryType=FType.STDEV)

    def sent(self, token):
        if (token.text.lower() in self.nlp.vocab.strings
            and self.nlp.vocab.strings[token.text.lower()]
                in self.sentiment):
            return self.sentiment[
                self.nlp.vocab.strings[token.text.lower()]]
        else:
            return 0

    def atr(self, token):
        if self.alphanum_word(token.text):
            return self.abstract_trait(token)
        else:
            return None

    def propn_abstract_traits(self, tokens):
        return sum(self.abstract_traits(tokens)) / len(tokens)

    def isanim(self, token):
        if self.alphanum_word(token.text):
            return self.is_animate(token)
        else:
            return None

    def propn_anims(self, tokens):
        return sum(self.animates(tokens)) / len(tokens)

    def isloc(self, token):
        if self.alphanum_word(token.text):
            return self.is_location(token)
        else:
            return None

    def locs(self, tokens):
        return [token._.location for token in tokens]

    def propn_locs(self, tokens):
        return sum([loc for loc in tokens._.locations
                    if loc is not None]) / len(tokens)

    def propn_deictics(self, tokens):
        return sum(self.deictics(tokens))/len(tokens)

    def dtv(self, document):
        return [[token.i, token.vector]
                for token in document
                if token.has_vector
                and not token.is_stop
                and token.tag_ in self.content_tags]

    #####################
    # Define extensions  #
    #####################

    def add_extensions(self):

        """
         Funcion to add extensions that allow us to access the various
         lexicons this module is designed to support.
        """

        #################################
        # Lemmas, word types, and roots #
        #################################

        # Get the lemmas in the document
        if not Doc.has_extension("lemmas") \
           or not Span.has_extension("lemmas"):
            Span.set_extension("lemmas", getter=self.lems, force=True)
            Doc.set_extension("lemmas", getter=self.lems, force=True)

        # Get the unique word types in the document
        if not Doc.has_extension("word_types") \
           or not Span.has_extension("word_types"):
            Span.set_extension("word_types", getter=self.typs, force=True)
            Doc.set_extension("word_types", getter=self.typs, force=True)

        # Access the roots dictionary from the token instance
        if not Token.has_extension('root'):
            Token.set_extension("root", getter=self.rt)

        # Access the roots dictionary from the Doc instance
        if not Doc.has_extension("morphroot") \
           or not Span.has_extension("morphroot"):
            Span.set_extension("morphroot", getter=self.mrts, force=True)
            Doc.set_extension("morphroot", getter=self.mrts, force=True)

        # Document level measure: unique word family type count
        # number of distinct word families in the text
        if not Doc.has_extension("wf_type_count") \
           or not Span.has_extension("wf_type_count"):
            Span.set_extension("wf_type_count", getter=self.typ, force=True)
            Doc.set_extension("wf_type_count", getter=self.typ, force=True)

        # Document level measure: unique lemma count
        if not Doc.has_extension("lemma_type_count") \
           or not Span.has_extension("lemma_type_count"):
            Span.set_extension("lemma_type_count",
                               getter=self.lemc,
                               force=True)
            Doc.set_extension("lemma_type_count",
                              getter=self.lemc,
                              force=True)

        # Document level measure: unique word type count
        if not Doc.has_extension("type_count") \
           or not Span.has_extension("type_count"):
            Span.set_extension("type_count",
                               getter=self.typc,
                               force=True)
            Doc.set_extension("type_count",
                              getter=self.typc,
                              force=True)

        # Document level measure: unique word token count
        if not Doc.has_extension("token_count") \
           or not Span.has_extension("token_count"):
            Span.set_extension("token_count",
                               getter=self.tokc,
                               force=True)
            Doc.set_extension("token_count",
                              getter=self.tokc,
                              force=True)

        ###########################
        # Syllable count features #
        ###########################

        # Number of syllables has been validated as a measure
        # of vocabulary difficulty

        # Get the number of syllables for a Token
        if not Token.has_extension('nSyll'):
            Token.set_extension("nSyll", getter=self.ns)

        # Access the syllables dictionary from the Doc instance
        if not Doc.has_extension("nSyllables") \
           or not Span.has_extension("nSyllables"):
            Span.set_extension("nSyllables", getter=self.sylls, force=True)
            Doc.set_extension("nSyllables", getter=self.sylls, force=True)

        # Document level measure: mean number of syllables in a content token
        if not Doc.has_extension("mean_nSyll") \
           or not Span.has_extension("mean_nSyll"):
            Span.set_extension("mean_nSyll", getter=self.mns, force=True)
            Doc.set_extension("mean_nSyll", getter=self.mns, force=True)

        # Document level measure: median number of syllables in a content token
        if not Doc.has_extension("med_nSyll") \
           or not Span.has_extension("med_nSyll"):
            Span.set_extension("med_nSyll", getter=self.mdns, force=True)
            Doc.set_extension("med_nSyll", getter=self.mdns, force=True)

        # Document level measure: max number of syllables
        # in a content token
        if not Doc.has_extension("max_nSyll") \
           or not Span.has_extension("max_nSyll"):
            Span.set_extension("max_nSyll", getter=self.mxns, force=True)
            Doc.set_extension("max_nSyll", getter=self.mxns, force=True)

        # Document level measure: min number of syllables in a content token
        if not Doc.has_extension("min_nSyll") \
           or not Span.has_extension("min_nSyll"):
            Span.set_extension("min_nSyll", getter=self.minns, force=True)
            Doc.set_extension("min_nSyll", getter=self.minns, force=True)

        # Document level measure: std. dev. of number
        # of syllables in a content token
        if not Doc.has_extension("std_nSyll") \
           or not Span.has_extension("std_nSyll"):
            Span.set_extension("std_nSyll", getter=self.stdns, force=True)
            Doc.set_extension("std_nSyll", getter=self.stdns, force=True)

        #####################################
        # Word length features (sqrt n chars) #
        #####################################

        # Word length in characters has been validated as
        # a measure of vocabulary difficulty

        # Get the number of characters for a Token
        if not Token.has_extension('sqrtNChars'):
            Token.set_extension("sqrtNChars", getter=self.nc)

        # Access the list of sqr nchars from the Doc instance
        if not Doc.has_extension("sqrtNChars") \
           or not Span.has_extension("sqrtNChars"):
            Span.set_extension("sqrtNChars", getter=self.chars, force=True)
            Doc.set_extension("sqrtNChars", getter=self.chars, force=True)

        # Document level measure: mean sqrt of n chars in a content token
        if not Doc.has_extension("mean_sqnChars") \
           or not Span.has_extension("mean_sqnChars"):
            Span.set_extension("mean_sqnChars", getter=self.mnc, force=True)
            Doc.set_extension("mean_sqnChars", getter=self.mnc, force=True)

        # Document level measure: median  sqrt of n chars in a content token
        if not Doc.has_extension("med_sqnChars") \
           or not Span.has_extension("med_sqnChars"):
            Span.set_extension("med_sqnChars", getter=self.mdnc, force=True)
            Doc.set_extension("med_sqnChars", getter=self.mdnc, force=True)

        # Document level measure: max  sqrt of n chars in a content token
        if not Doc.has_extension("max_sqnChars") \
           or not Span.has_extension("max_sqnChars"):
            Span.set_extension("max_sqnChars", getter=self.mxnc, force=True)
            Doc.set_extension("max_sqnChars", getter=self.mxnc, force=True)

        # Document level measure: min  sqrt of n chars in a content token
        if not Doc.has_extension("min_sqnChars") \
           or not Span.has_extension("min_sqnChars"):
            Span.set_extension("min_sqnChars", getter=self.minnc, force=True)
            Doc.set_extension("min_sqnChars", getter=self.minnc, force=True)

        # Document level measure: std. dev. of  sqrt of n chars
        # in a content token
        if not Doc.has_extension("std_sqnChars") \
           or not Span.has_extension("std_sqnChars"):
            Span.set_extension("std_sqnChars", getter=self.stdnc, force=True)
            Doc.set_extension("std_sqnChars", getter=self.stdnc, force=True)

        ############################
        # Latinate vocabulary flag #
        ############################

        # The latinate flag identifies words that appear likely to be
        # more academic as they are formed using latin or greek prefixes
        # and suffixes/ Latinate words are less likely to be known, all
        # other things being equal

        # Get flag 1 or 0 indicating whether a word has latinate
        # prefixes or suffixes
        if not Token.has_extension('is_latinate'):
            Token.set_extension("is_latinate",
                                getter=self.is_latinate,
                                force=True)

        # Access the latinates list from the Doc instance
        if not Doc.has_extension("latinates") \
           or not Span.has_extension("latinates"):
            Span.set_extension("latinates", getter=self.lats, force=True)
            Doc.set_extension("latinates", getter=self.lats, force=True)

        # Document level measure: proportion of latinate content words
        if not Doc.has_extension("propn_latinate") \
           or not Span.has_extension("propn_latinate"):
            Span.set_extension("propn_latinate", getter=self.mnlat, force=True)
            Doc.set_extension("propn_latinate", getter=self.mnlat, force=True)

        ################################
        # Academic vocabulary measures #
        ################################

        # Academic status is a measure of vocabulary difficulty
        if not Token.has_extension('is_academic'):
            Token.set_extension("is_academic", getter=self.is_academic)

        # Access the academic status list from the Doc instance
        if not Doc.has_extension("academics") \
           or not Span.has_extension("academics"):
            Span.set_extension("academics", getter=self.acads, force=True)
            Doc.set_extension("academics", getter=self.acads, force=True)

        # Document level measure: proportion of content words
        # on targeted academic language/tier II vocabulary lists
        if not Doc.has_extension("propn_academic") \
           or not Span.has_extension("propn_academic"):
            Span.set_extension("propn_academic",
                               getter=self.mnacad,
                               force=True)
            Doc.set_extension("propn_academic",
                              getter=self.mnacad,
                              force=True)

        #################################################################
        # Word Family Sizes as measured by slightly modified version of #
        # Paul Nation's word family list.                               #
        #################################################################

        # The family size flag identifies the number of morphologically
        # related words in this word's word family. Words with larger
        # word families have been shown to be, on average, easier
        # vocabulary.

        if not Token.has_extension('family_size'):
            Token.set_extension("family_size", getter=self.fms, force=True)

        # Access the family size dictionary from the Doc instance
        if not Doc.has_extension("family_sizes") \
           or not Span.has_extension("family_sizes"):
            Span.set_extension("family_sizes", getter=self.fmss, force=True)
            Doc.set_extension("family_sizes", getter=self.fmss, force=True)

        # Document level measure: mean word family size for content words
        if not Doc.has_extension("mean_family_size") \
           or not Span.has_extension("mean_family_size"):
            Span.set_extension("mean_family_size",
                               getter=self.mnfms,
                               force=True)
            Doc.set_extension("mean_family_size",
                              getter=self.mnfms,
                              force=True)

        # Document level measure: median word family size for content words
        if not Doc.has_extension("med_family_size") \
           or not Span.has_extension("med_family_size"):
            Span.set_extension("med_family_size",
                               getter=self.mdfms, force=True)
            Doc.set_extension("med_family_size", getter=self.mdfms, force=True)

        # Document level measure: max word family size for content words
        if not Doc.has_extension("max_family_size") \
           or not Span.has_extension("max_family_size"):
            Span.set_extension("max_family_size",
                               getter=self.mxfms,
                               force=True)
            Doc.set_extension("max_family_size",
                              getter=self.mxfms,
                              force=True)

        # Document level measure: min word family size for content words
        if not Doc.has_extension("min_family_size") \
           or not Span.has_extension("min_family_size"):
            Span.set_extension("min_family_size",
                               getter=self.minfms,
                               force=True)
            Doc.set_extension("min_family_size",
                              getter=self.minfms,
                              force=True)

        # Document level measure: st dev of word family size for content words
        if not Doc.has_extension("std_family_size") \
           or not Span.has_extension("std_family_size"):
            Span.set_extension("std_family_size",
                               getter=self.stdfms,
                               force=True)
            Doc.set_extension("std_family_size",
                              getter=self.stdfms,
                              force=True)

        ########################################
        # Sense count measures (using WordNet) #
        ########################################

        # The number of senses associated with a word is a measure
        # of vocabulary difficulty
        if not Token.has_extension('nSenses'):
            Token.set_extension("nSenses", getter=self.nsem)
            Token.set_extension("logNSenses", getter=self.lognsem)

        # Document level measure: list of number of word senses
        # for each token in doc
        if not Doc.has_extension("sensenums") \
           or not Span.has_extension("sensenums"):
            Span.set_extension("sensnums", getter=self.senseno, force=True)
            Doc.set_extension("sensenums", getter=self.senseno, force=True)

        # Document level measure: list of number of word senses
        # for each token in doc
        if not Doc.has_extension("logsensenums") \
           or not Span.has_extension("logsensenums"):
            Span.set_extension("logsensenums",
                               getter=self.logsenseno,
                               force=True)
            Doc.set_extension("logsensenums",
                              getter=self.logsenseno,
                              force=True)

        # Document level measure: mean number of word senses
        if not Doc.has_extension("mean_nSenses") \
           or not Span.has_extension("mean_nSenses"):
            Span.set_extension("self.mean_nSenses",
                               getter=self.mnsense,
                               force=True)
            Doc.set_extension("mean_nSenses",
                              getter=self.mnsense,
                              force=True)

        # Document level measure: median number of word senses
        if not Doc.has_extension("med_nSenses") \
           or not Span.has_extension("med_nSenses"):
            Span.set_extension("med_nSenses",
                               getter=self.mdsense,
                               force=True)
            Doc.set_extension("med_nSenses",
                              getter=self.mdsense,
                              force=True)

        # Document level measure: max number of word senses
        if not Doc.has_extension("max_nSenses") \
           or not Span.has_extension("max_nSenses"):
            Span.set_extension("max_nSenses",
                               getter=self.mxsense,
                               force=True)
            Doc.set_extension("max_nSenses",
                              getter=self.mxsense,
                              force=True)

        # Document level measure: min number of word senses
        if not Doc.has_extension("min_nSenses") \
           or not Span.has_extension("min_nSenses"):
            Span.set_extension("min_nSenses",
                               getter=self.minsense,
                               force=True)
            Doc.set_extension("min_nSenses",
                              getter=self.minsense,
                              force=True)

        # Document level measure: standard deviation of number of word senses
        if not Doc.has_extension("std_nSenses") \
           or not Span.has_extension("std_nSenses"):
            Span.set_extension("std_nSenses",
                               getter=self.stdsense,
                               force=True)
            Doc.set_extension("std_nSenses",
                              getter=self.stdsense,
                              force=True)

        # Document level measure: mean log number of word senses
        if not Doc.has_extension("mean_logNSenses") \
           or not Span.has_extension("mean_logNSenses"):
            Span.set_extension("mean_logNSenses",
                               getter=self.mnlognsense,
                               force=True)
            Doc.set_extension("mean_logNSenses",
                              getter=self.mnlognsense,
                              force=True)

        # Document level measure: median log number of word senses
        if not Doc.has_extension("med_logNSenses") \
           or not Span.has_extension("med_logNSenses"):
            Span.set_extension("med_logNSenses",
                               getter=self.mdlognsense,
                               force=True)
            Doc.set_extension("med_logNSenses",
                              getter=self.mdlognsense,
                              force=True)

        # Document level measure: max of log number of word senses
        if not Doc.has_extension("max_logNSenses") \
           or not Span.has_extension("max_logNSenses"):
            Span.set_extension("max_logNSenses",
                               getter=self.mxlognsense,
                               force=True)
            Doc.set_extension("max_logNSenses",
                              getter=self.mxlognsense,
                              force=True)

        # Document level measure: min number of log word senses
        if not Doc.has_extension("min_logNSenses") \
           or not Span.has_extension("min_logNSenses"):
            Span.set_extension("min_logNSenses",
                               getter=self.minlognsense,
                               force=True)
            Doc.set_extension("min_logNSenses",
                              getter=self.minlognsense,
                              force=True)

        # Document level measure: standard deviation of log number
        # of word senses
        if not Doc.has_extension("std_logNSenses") \
           or not Span.has_extension("std_logNSenses"):
            Span.set_extension("std_logNSenses",
                               getter=self.stdlognsense,
                               force=True)
            Doc.set_extension("std_logNSenses",
                              getter=self.stdlognsense,
                              force=True)

        #############################
        # Morphology based measures #
        #############################

        # Morpholex includes information about prefixes and suffixes --
        # such as the size of families and frequency of specific prefixes
        # and suffixes. We're not currently using this information
        # but these are also known to be predictors of vocabulary difficulty

        # Access the morpholex dictionary from the Doc instance
        if not Doc.has_extension("morpholex") \
           or not Span.has_extension("morpholex"):
            Span.set_extension("morpholex", getter=self.morpho, force=True)
            Doc.set_extension("morpholex", getter=self.morpho, force=True)

        # Access a string that identifies roots, prefixes, and
        # suffixes in the word. Not currently used but can be
        # processed to identify the specific morphemes
        # in a word according to MorphoLex calculations
        if not Token.has_extension('morpholexsegm'):
            Token.set_extension("morpholexsegm", getter=self.morpholexsegm)

        if not Doc.has_extension("morpholexSegm") \
           or not Span.has_extension("morpholexSegm"):
            Span.set_extension("morpholexSegm",
                               getter=self.morpholexsegms,
                               force=True)
            Doc.set_extension("morpholexSegm",
                              getter=self.morpholexsegms,
                              force=True)

        # The number of morphemes in a word is a measure
        # of vocabulary difficulty
        if not Token.has_extension('nMorph'):
            Token.set_extension("nMorph", getter=self.nm)

        # Document level measure: list of number of morphemes
        # for each token in document
        if not Doc.has_extension("morphnums") \
           or not Span.has_extension("morphnums"):
            Span.set_extension("morphnums", getter=self.morphn, force=True)
            Doc.set_extension("morphnums", getter=self.morphn, force=True)

        # Document level measure: mean number of morphemes
        if not Doc.has_extension("mean_nMorph") \
           or not Span.has_extension("mean_nMorph"):
            Span.set_extension("mean_nMorph", getter=self.mnmorph, force=True)
            Doc.set_extension("mean_nMorph", getter=self.mnmorph, force=True)

        # Document level measure: median number of morphemes
        if not Doc.has_extension("med_nMorph") \
           or not Span.has_extension("med_nMorph"):
            Span.set_extension("med_nMorph", getter=self.mdmorph, force=True)
            Doc.set_extension("med_nMorph", getter=self.mdmorph, force=True)

        # Document level measure: max number of morphemes
        if not Doc.has_extension("max_nMorph") \
           or not Span.has_extension("max_nMorph"):
            Span.set_extension("max_nMorph", getter=self.mxmorph, force=True)
            Doc.set_extension("max_nMorph", getter=self.mxmorph, force=True)

        # Document level measure: min number of morphemes
        if not Doc.has_extension("min_nMorph") \
           or not Span.has_extension("max_nMorph"):
            Span.set_extension("min_nMorph", getter=self.minmorph, force=True)
            Doc.set_extension("min_nMorph", getter=self.minmorph, force=True)

        # Document level measure: standard deviation of number of morphemes
        if not Doc.has_extension("std_nMorph") \
           or not Span.has_extension("std_nMorph"):
            Span.set_extension("std_nMorph", getter=self.stdmorph, force=True)
            Doc.set_extension("std_nMorph", getter=self.stdmorph, force=True)

        # The frequency of the root is a measure of vocabulary difficulty
        if not Token.has_extension('root1_freq_HAL'):
            Token.set_extension("root1_freq_HAL", getter=self.rfqh)

        # The frequency of the root is a measure of vocabulary difficulty
        if not Token.has_extension('root2_freq_HAL'):
            Token.set_extension("root2_freq_HAL", getter=self.rfqh2)

        # The frequency of the root is a measure of vocabulary difficulty
        if not Token.has_extension('root3_freq_HAL'):
            Token.set_extension("root3_freq_HAL", getter=self.rfqh3)

        # Document level measure: list of HAL frequencies
        # for the first root for for each token in document
        if not Doc.has_extension("root_freqs_HAL") \
           or not Span.has_extension("root_freqs_HAL"):
            Span.set_extension("root_freqs_HAL", getter=self.rfsh, force=True)
            Doc.set_extension("root_freqs_HAL", getter=self.rfsh, force=True)

        # Document level measure: mean HAL root frequency
        if not Doc.has_extension("mean_freq_HAL") \
           or not Span.has_extension("mean_freq_HAL"):
            Span.set_extension("mean_freq_HAL", getter=self.mnfrh, force=True)
            Doc.set_extension("mean_freq_HAL", getter=self.mnfrh, force=True)

        # Document level measure: median HAL root frequency
        if not Doc.has_extension("med_freq_HAL") \
           or not Span.has_extension("med_freq_HAL"):
            Span.set_extension("med_freq_HAL", getter=self.mdfrh, force=True)
            Doc.set_extension("med_freq_HAL", getter=self.mdfrh, force=True)

        # Document level measure: max HAL root frequency
        if not Doc.has_extension("max_freq_HAL") \
           or not Span.has_extension("max_freq_HAL"):
            Span.set_extension("max_freq_HAL", getter=self.mxfrh, force=True)
            Doc.set_extension("max_freq_HAL", getter=self.mxfrh, force=True)

        # Document level measure: min HAL root frequency
        if not Doc.has_extension("min_freq_HAL") \
           or not Span.has_extension("max_freq_HAL"):
            Span.set_extension("min_freq_HAL", getter=self.minfrh, force=True)
            Doc.set_extension("min_freq_HAL", getter=self.minfrh, force=True)

        # Document level measure: standard deviation of HAL root frequency
        if not Doc.has_extension("std_freq_HAL") \
           or not Span.has_extension("std_freq_HAL"):
            Span.set_extension("std_freq_HAL", getter=self.stdfrh, force=True)
            Doc.set_extension("std_freq_HAL", getter=self.stdfrh, force=True)

        # Document level measure: list of HAL frequencies for the first root
        # for each token in document
        if not Doc.has_extension("log_root_freqs_HAL") \
           or not Span.has_extension("log_root_freqs_HAL"):
            Span.set_extension("log_root_freqs_HAL",
                               getter=self.rfshlg,
                               force=True)
            Doc.set_extension("log_root_freqs_HAL",
                              getter=self.rfshlg,
                              force=True)

        # Document level measure: mean HAL root frequency
        if not Doc.has_extension("mean_logfreq_HAL") \
           or not Span.has_extension("mean_logfreq_HAL"):
            Span.set_extension("mean_logfreq_HAL",
                               getter=self.mnlgfrh,
                               force=True)
            Doc.set_extension("mean_logfreq_HAL",
                              getter=self.mnlgfrh,
                              force=True)

        # Document level measure: median HAL root frequency
        if not Doc.has_extension("med_logfreq_HAL") \
           or not Span.has_extension("med_logfreq_HAL"):
            Span.set_extension("med_logfreq_HAL",
                               getter=self.mdlgfrh,
                               force=True)
            Doc.set_extension("med_logfreq_HAL",
                              getter=self.mdlgfrh,
                              force=True)

        # Document level measure: max HAL root frequency
        if not Doc.has_extension("max_logfreq_HAL") \
           or not Span.has_extension("max_logfreq_HAL"):
            Span.set_extension("max_logfreq_HAL",
                               getter=self.mxlgfrh,
                               force=True)
            Doc.set_extension("max_logfreq_HAL",
                              getter=self.mxlgfrh,
                              force=True)

        # Document level measure: min HAL root frequency
        if not Doc.has_extension("min_logfreq_HAL") \
           or not Span.has_extension("max_logfreq_HAL"):
            Span.set_extension("min_logfreq_HAL",
                               getter=self.minlgfrh,
                               force=True)
            Doc.set_extension("min_logfreq_HAL",
                              getter=self.minlgfrh,
                              force=True)

        # Document level measure: standard deviation of HAL root frequency
        if not Doc.has_extension("std_logfreq_HAL") \
           or not Span.has_extension("std_logfreq_HAL"):
            Span.set_extension("std_logfreq_HAL",
                               getter=self.stdlgfrh,
                               force=True)
            Doc.set_extension("std_logfreq_HAL",
                              getter=self.stdlgfrh,
                              force=True)

        # The family size of the root is a measure of vocabulary difficulty
        if not Token.has_extension('root_famSize'):
            Token.set_extension("root_famSize", getter=self.rfs)

        # Document level measure: list of family sizes for the first root for
        # each token in document
        if not Doc.has_extension("root_fam_sizes") \
           or not Span.has_extension("root_fam_sizes"):
            Span.set_extension("root_fam_sizes", getter=self.rfsz, force=True)
            Doc.set_extension("root_fam_sizes", getter=self.rfsz, force=True)

        # Document level measure: mean root family size
        if not Doc.has_extension("mean_root_fam_size") \
           or not Span.has_extension("mean_root_fam_size"):
            Span.set_extension("mean_root_fam_size",
                               getter=self.mnrfsz,
                               force=True)
            Doc.set_extension("mean_root_fam_size",
                              getter=self.mnrfsz,
                              force=True)

        # Document level measure: median root family size
        if not Doc.has_extension("med_root_fam_size") \
           or not Span.has_extension("med_root_fam_size"):
            Span.set_extension("med_root_fam_size",
                               getter=self.mdrfsz,
                               force=True)
            Doc.set_extension("med_root_fam_size",
                              getter=self.mdrfsz,
                              force=True)

        # Document level measure: max root family size
        if not Doc.has_extension("max_root_fam_size") \
           or not Span.has_extension("max_root_fam_size"):
            Span.set_extension("max_root_fam_size",
                               getter=self.mxrfsz,
                               force=True)
            Doc.set_extension("max_root_fam_size",
                              getter=self.mxrfsz,
                              force=True)

        # Document level measure: min root family size
        if not Doc.has_extension("min_root_fam_size") \
           or not Span.has_extension("max__root_fam_size"):
            Span.set_extension("min_root_fam_size",
                               getter=self.minrfsz,
                               force=True)
            Doc.set_extension("min_root_fam_size",
                              getter=self.minrfsz,
                              force=True)

        # Document level measure: standard deviation of HAL family size
        if not Doc.has_extension("std_root_fam_size") \
           or not Span.has_extension("std_root_fam_size"):
            Span.set_extension("std_root_fam_size",
                               getter=self.stdrfsz,
                               force=True)
            Doc.set_extension("std_root_fam_size",
                              getter=self.stdrfsz,
                              force=True)

        # The percentage of words more frequent in the family size
        # is a measure of vocabulary difficulty
        if not Token.has_extension('root_pfmf'):
            Token.set_extension("root_pfmf", getter=self.rpfmf)

        # Document level measure: list of pfmfs for the first root for
        # each token in document
        if not Doc.has_extension("root_pfmfs") \
           or not Span.has_extension("root_pfmfs"):
            Span.set_extension("root_pfmfs",
                               getter=self.rfszrt,
                               force=True)
            Doc.set_extension("root_pfmfs",
                              getter=self.rfszrt,
                              force=True)

        # Document level measure: mean root pfmf
        if not Doc.has_extension("mean_root_pfmf") \
           or not Span.has_extension("mean_root_pfmf"):
            Span.set_extension("mean_root_pfmf",
                               getter=self.mnrpfmf,
                               force=True)
            Doc.set_extension("mean_root_pfmf",
                              getter=self.mnrpfmf,
                              force=True)

        # Document level measure:

        if not Doc.has_extension("med_root_pfmf") \
           or not Span.has_extension("med_root_pfmf"):
            Span.set_extension("med_root_pfmf",
                               getter=self.mdrpfmf,
                               force=True)
            Doc.set_extension("med_root_pfmf",
                              getter=self.mdrpfmf,
                              force=True)

        # Document level measure: max root family size
        if not Doc.has_extension("max_root_pfmf") \
           or not Span.has_extension("max_root_pfmf"):
            Span.set_extension("max_root_pfmf",
                               getter=self.mxrpfmf,
                               force=True)
            Doc.set_extension("max_root_pfmf",
                              getter=self.mxrpfmf,
                              force=True)

        # Document level measure: min root family size
        if not Doc.has_extension("min_root_pfmf") \
           or not Span.has_extension("max__root_pfmf"):
            Span.set_extension("min_root_pfmf",
                               getter=self.minrpfmf,
                               force=True)
            Doc.set_extension("min_root_pfmf",
                              getter=self.minrpfmf,
                              force=True)

        # Document level measure: standard deviation of HAL family size
        if not Doc.has_extension("std_root_pfmf") \
           or not Span.has_extension("std_root_pfmf"):
            Span.set_extension("std_root_pfmf",
                               getter=self.stdrpfmf,
                               force=True)
            Doc.set_extension("std_root_pfmf",
                              getter=self.stdrpfmf,
                              force=True)

        ###########################
        # Word Frequency Measures #
        ###########################

        # Word frequency is a measure of vocabulary difficulty.

        # We can calculate word frequency for the specific word form
        if not Token.has_extension('token_freq'):
            Token.set_extension("token_freq", getter=self.tf)

        # Or we can calculate the frequency for the lemma (base form)
        # of the word
        if not Token.has_extension('lemma_freq'):
            Token.set_extension("lemma_freq", getter=self.lf)

        # Or we can calculate the frequency for the root form of the word
        if not Token.has_extension('root_freq'):
            Token.set_extension("root_freq", getter=self.zrf)

        # Or we can calculate the frequency for the root for a
        # whole word family
        if not Token.has_extension('max_freq'):
            Token.set_extension("max_freq", getter=self.mff)

        # Document level measure: list of token frequencies
        if not Doc.has_extension("token_freqs") \
           or not Span.has_extension("token_freqs"):
            Span.set_extension("token_freqs",
                               getter=self.tkfrq,
                               force=True)
            Doc.set_extension("token_freqs",
                              getter=self.tkfrq,
                              force=True)

        # Document level measure: list of lemma frequencies for tokens
        # in the document
        if not Doc.has_extension("lemma_freqs") \
           or not Span.has_extension("lemma_freqs"):
            Span.set_extension("lemma_freqs",
                               getter=self.lmfrqs,
                               force=True)
            Doc.set_extension("lemma_freqs",
                              getter=self.lmfrqs,
                              force=True)

        # Document level measure: list of root frequencies for tokens in
        # the document (counting root frequency)
        if not Doc.has_extension("root_freqs") \
           or not Span.has_extension("root_freqs"):
            Span.set_extension("root_freqs",
                               getter=self.rtfrqs,
                               force=True)
            Doc.set_extension("root_freqs",
                              getter=self.rtfrqs,
                              force=True)

        # Document level measure: list of max freqs for word, lemma or
        # root in document
        if not Doc.has_extension("max_freqs") \
           or not Span.has_extension("max_freqs"):
            Span.set_extension("max_freqs",
                               getter=self.fmf,
                               force=True)
            Doc.set_extension("max_freqs",
                              getter=self.fmf,
                              force=True)

        # Document level measure: mean token frequency for content words
        if not Doc.has_extension("mean_token_frequency") \
           or not Span.has_extension("mean_token_frequency"):
            Span.set_extension("mean_token_frequency",
                               getter=self.mnfrq,
                               force=True)
            Doc.set_extension("mean_token_frequency",
                              getter=self.mnfrq,
                              force=True)

        # Document level measure: median token frequency for content words
        if not Doc.has_extension("median_token_frequency") \
           or not Span.has_extension("median_token_frequency"):
            Span.set_extension("median_token_frequency",
                               getter=self.mdfrq,
                               force=True)
            Doc.set_extension("median_token_frequency",
                              getter=self.mdfrq,
                              force=True)

        # Document level measure: max token frequency for content words
        if not Doc.has_extension("max_token_frequency") \
           or not Span.has_extension("max_token_frequency"):
            Span.set_extension("max_token_frequency",
                               getter=self.mxfrq,
                               force=True)
            Doc.set_extension("max_token_frequency",
                              getter=self.mxfrq,
                              force=True)

        # Document level measure: min token frequency for content words
        if not Doc.has_extension("min_token_frequency") \
           or not Span.has_extension("min_token_frequency"):
            Span.set_extension("min_token_frequency",
                               getter=self.minfrq,
                               force=True)
            Doc.set_extension("min_token_frequency",
                              getter=self.minfrq,
                              force=True)

        # Document level measure: standard deviation of token frequency
        # for content words
        if not Doc.has_extension("std_token_frequency") \
           or not Span.has_extension("std_token_frequency"):
            Span.set_extension("std_token_frequency",
                               getter=self.stdfrq,
                               force=True)
            Doc.set_extension("std_token_frequency",
                              getter=self.stdfrq,
                              force=True)

        # Document level measure: mean lemma frequency for content words
        # (counting lemma base frequency)
        if not Doc.has_extension("mean_lemma_frequency") \
           or not Span.has_extension("mean_lemma_frequency"):
            Span.set_extension("mean_lemma_frequency",
                               getter=self.mnlmfrq,
                               force=True)
            Doc.set_extension("mean_lemma_frequency",
                              getter=self.mnlmfrq,
                              force=True)

        # Document level measure: median lemma frequency for content words
        # (counting lemma base frequency)
        if not Doc.has_extension("median_lemma_frequency") \
           or not Span.has_extension("median_lemma_frequency"):
            Span.set_extension("median_lemma_frequency",
                               getter=self.mdlmfrq,
                               force=True)
            Doc.set_extension("median_lemma_frequency",
                              getter=self.mdlmfrq,
                              force=True)

        # Document level measure: max lemma frequency for content words
        # (counting lemma base frequency)
        if not Doc.has_extension("max_lemma_frequency") \
           or not Span.has_extension("max_lemma_frequency"):
            Span.set_extension("max_lemma_frequency",
                               getter=self.mxlmfrq,
                               force=True)
            Doc.set_extension("max_lemma_frequency",
                              getter=self.mxlmfrq,
                              force=True)

        # Document level measure: min lemma frequency for content words
        # (counting lemma base frequency)
        if not Doc.has_extension("min_lemma_frequency") \
           or not Span.has_extension("min_lemma_frequency"):
            Span.set_extension("min_lemma_frequency",
                               getter=self.minlmfrq,
                               force=True)
            Doc.set_extension("min_lemma_frequency",
                              getter=self.minlmfrq,
                              force=True)

        # Document level measure: standard deviation of lemma
        # frequency for content words (counting lemma base frequency)
        if not Doc.has_extension("std_lemma_frequency") \
           or not Span.has_extension("std_lemma_frequency"):
            Span.set_extension("std_lemma_frequency",
                               getter=self.stdlmfrq,
                               force=True)
            Doc.set_extension("std_lemma_frequency",
                              getter=self.stdlmfrq,
                              force=True)

        # Document level measure: mean word family frequency for content words
        # (counting root frequency)
        if not Doc.has_extension("mean_max_frequency") \
           or not Span.has_extension("mean_max_frequency"):
            Span.set_extension("mean_max_frequency",
                               getter=self.mnrtfrq,
                               force=True)
            Doc.set_extension("mean_max_frequency",
                              getter=self.mnrtfrq,
                              force=True)

        # Document level measure: median word family frequency for
        # content words (counting root frequency)
        if not Doc.has_extension("median_max_frequency") \
           or not Span.has_extension("median_max_frequency"):
            Span.set_extension("median_max_frequency",
                               getter=self.mdrtfrq,
                               force=True)
            Doc.set_extension("median_max_frequency",
                              getter=self.mdrtfrq,
                              force=True)

        # Document level measure: max word family frequency for content words
        # (counting root frequency)
        if not Doc.has_extension("max_max_frequency") \
           or not Span.has_extension("max_max_frequency"):
            Span.set_extension("max_max_frequency",
                               getter=self.mxrtfrq,
                               force=True)
            Doc.set_extension("max_max_frequency",
                              getter=self.mxrtfrq,
                              force=True)

        # Document level measure: min word family frequency for content words
        # (counting root frequency)
        if not Doc.has_extension("min_max_frequency") \
           or not Span.has_extension("min_max_frequency"):
            Span.set_extension("min_max_frequency",
                               getter=self.minrtfrq,
                               force=True)
            Doc.set_extension("min_max_frequency",
                              getter=self.minrtfrq,
                              force=True)

        # Document level measure: standard deviation of root frequency
        # for content words (counting root frequency)
        if not Doc.has_extension("std_max_frequency") \
           or not Span.has_extension("std_max_frequency"):
            Span.set_extension("std_max_frequency",
                               getter=self.stdrtfrq,
                               force=True)
            Doc.set_extension("std_max_frequency",
                              getter=self.stdrtfrq,
                              force=True)

        #################################################
        # Measures of lexical concreteness/abstractness #
        #################################################

        # Concreteness is another measure of vocabulary difficulty
        if not Token.has_extension('concreteness'):
            Token.set_extension("concreteness", getter=self.concreteness)

        # Access the concreteness status dictionary from the Doc instance
        if not Doc.has_extension("concretes") \
           or not Span.has_extension("concretes"):
            Span.set_extension("concretes", getter=self.concrs, force=True)
            Doc.set_extension("concretes", getter=self.concrs, force=True)

        # Document level measure: mean concreteness of content words
        if not Doc.has_extension("mean_concreteness") \
           or not Span.has_extension("mean_concreteness"):
            Span.set_extension("mean_concreteness",
                               getter=self.mncr,
                               force=True)
            Doc.set_extension("mean_concreteness",
                              getter=self.mncr,
                              force=True)

        # Document level measure: median concreteness of content words
        if not Doc.has_extension("med_concreteness") \
           or not Span.has_extension("med_concreteness"):
            Span.set_extension("med_concreteness",
                               getter=self.mdcr,
                               force=True)
            Doc.set_extension("med_concreteness",
                              getter=self.mdcr,
                              force=True)

        # Document level measure: max concreteness of content words
        if not Doc.has_extension("max_concreteness") \
           or not Span.has_extension("max_concreteness"):
            Span.set_extension("max_concreteness",
                               getter=self.mxcr,
                               force=True)
            Doc.set_extension("max_concreteness",
                              getter=self.mxcr,
                              force=True)

        # Document level measure: min concreteness of content words
        if not Doc.has_extension("min_concreteness") \
           or not Span.has_extension("min_concreteness"):
            Span.set_extension("min_concreteness",
                               getter=self.mincr,
                               force=True)
            Doc.set_extension("min_concreteness",
                              getter=self.mincr,
                              force=True)

        # Document level measure: standard deviation of concreteness of
        # content words
        if not Doc.has_extension("std_concreteness") \
           or not Span.has_extension("std_concreteness"):
            Span.set_extension("std_concreteness",
                               getter=self.stdcr,
                               force=True)
            Doc.set_extension("std_concreteness",
                              getter=self.stdcr,
                              force=True)

        #######################################
        # Sentiment and subjectivity measures #
        #######################################

        # To get SpacyTextBlob polarity, use extension ._.polarity
        # to get SpacyTextBlob subjectivity, use extension ._.subjectivity

        # to get list of assertion terms recognized by SpacyTextBlob,
        # use extension ._.assessments

        # Positive or negative polarity of words as measured by the
        #  SentiWord database. We also have SpacyTextBlob sentiment,
        # which includes two extensions: Token._.polarity
        # (for positive/negative sentiment) and Token._.subjectivity,
        # which evaluates the subjectivity (stance-taking) valence of
        # a word.
        if not Token.has_extension('sentiword'):
            Token.set_extension("sentiword", getter=self.sent)

        # Most sentiment/subjectivity measures are defined in
        # viewpointFeatures.py

        #############################################
        # Ontological categories (based on WordNet) #
        #############################################

        # For various purposes we need to know whether a noun
        # denotes an abstract trait
        if not Token.has_extension('abstract_trait'):
            Token.set_extension("abstract_trait", getter=self.atr)

        # List of abstract trait tokens in document
        if not Doc.has_extension("abstract_traits") \
           or not Span.has_extension("abstract_traits"):
            Span.set_extension("abstract_traits",
                               getter=self.abstract_traits,
                               force=True)
            Doc.set_extension("abstract_traits",
                              getter=self.abstract_traits,
                              force=True)

        # Proportion of tokens classified as abstract traits
        if not Doc.has_extension("propn_abstract_traits") \
           or not Span.has_extension("propn_abstract_traits"):
            Span.set_extension("propn_abstract_traits",
                               getter=self.propn_abstract_traits,
                               force=True)
            Doc.set_extension("propn_abstract_traits",
                              getter=self.propn_abstract_traits,
                              force=True)

        # For various purposes we need to know whether a noun is animate
        if not Token.has_extension('animate'):
            Token.set_extension("animate",
                                getter=self.isanim,
                                force=True)

        # List of animate tokens in document
        if not Doc.has_extension("animates") \
           or not Span.has_extension("animates"):
            Span.set_extension("animates",
                               getter=self.animates,
                               force=True)
            Doc.set_extension("animates",
                              getter=self.animates,
                              force=True)

        # Proportion of tokens classified as animate
        if not Doc.has_extension("propn_animates") \
           or not Span.has_extension("propn_animates"):
            Span.set_extension("propn_animates",
                               getter=self.propn_anims,
                               force=True)
            Doc.set_extension("propn_animates",
                              getter=self.propn_anims,
                              force=True)

        # List of deictic tokens in document
        if not Doc.has_extension("deictics") \
           or not Span.has_extension("deictics"):
            Span.set_extension("deictics", getter=self.deictics, force=True)
            Doc.set_extension("deictics", getter=self.deictics, force=True)

        # For various purposes we need to know whether a noun is locative
        if not Token.has_extension('location'):
            Token.set_extension("location", getter=self.isloc)

        # List of location tokens in document
        if not Doc.has_extension("locations") \
           or not Span.has_extension("locations"):
            Span.set_extension("locations", getter=self.locs, force=True)
            Doc.set_extension("locations", getter=self.locs, force=True)

        # Proportion of tokens classified as locations
        if not Doc.has_extension("propn_locations") \
           or not Span.has_extension("propn_locations"):
            Span.set_extension("propn_locations",
                               getter=self.propn_locs,
                               force=True)
            Doc.set_extension("propn_locations",
                              getter=self.propn_locs,
                              force=True)

        # Proportion of tokens classified as deictic
        if not Doc.has_extension("propn_deictics") \
           or not Span.has_extension("propn_deictics"):
            Span.set_extension("propn_deictics",
                               getter=self.propn_deictics,
                               force=True)
            Doc.set_extension("propn_deictics",
                              getter=self.propn_deictics,
                              force=True)

        ########################
        # Word Vector Measures #
        ########################

        # Extensions to allow us to get vectors for tokens in a spacy
        # doc or span
        if not Doc.has_extension('token_vectors') \
           or not Span.has_extension('token_vectors'):
            Span.set_extension("token_vectors",
                               getter=self.dtv,
                               force=True)
            Doc.set_extension("token_vectors",
                              getter=self.dtv,
                              force=True)

        # Word vector based measures are mostly contained in
        # syntaxDiscourseFeats.py

    attribute = wordnet.synsets('attribute')[1]
    attribute2 = wordnet.synsets('attribute')[2]
    quantity = wordnet.synsets('quantity')[0]
    part = wordnet.synsets('part')[0]
    part1 = wordnet.synsets('part')[1]
    possession = wordnet.synsets('possession')[1]
    group = wordnet.synsets('grouping')[0]
    vegetation = wordnet.synsets('vegetation')[0]
    gathering = wordnet.synsets('assemblage')[0]
    magnitude = wordnet.synsets('magnitude')[0]

    def abstract_trait(self, token):

        if token.text.lower() in self.abstractTraitNouns:
            return self.abstractTraitNouns[token.text.lower()]

        # Note that we're defining abstract trait this way in order to support
        # identifying semantically empty heads of noun phrases. The elements
        # listed here all can appear as the head word of a noun phrase, either
        # anaphorically (for numbers, determines, and adjectives) or as
        # semantically empty head nouns.
        if token.pos_ in ['NUM', 'DET', 'ADJ'] \
           and (token.dep_ == 'nsubj'
                or token.dep_ == 'nsubjpass'
                or token.dep_ == 'dobj'
                or token.dep_ == 'pobj'):
            return True
        if token.pos_ == 'NOUN':
            try:
                ttext = token.lemma_
                synsets = wordnet.synsets(ttext)
                if len(synsets) > 0:
                    hypernyms = set([i for i
                                     in synsets[0].closure(lambda s:
                                                           s.hypernyms())])
                    if len(hypernyms)>0 and \
                       (self.attribute in hypernyms \
                        or self.quantity in hypernyms \
                        or self.part in hypernyms \
                        or self.possession in hypernyms \
                        or (self.group[0] in hypernyms
                            and self.vegetation not in hypernyms) \
                        or self.gathering in hypernyms \
                        or self.magnitude in hypernyms \
                        or synsets[0] == self.attribute \
                        or synsets[0] == self.quantity \
                        or synsets[0] == self.part \
                        or synsets[0] == self.possession \
                        or synsets[0] == self.group[0] \
                        or synsets[0] == self.gathering \
                        or synsets[0] == self.magnitude):
                        self.abstractTraitNouns[token.text.lower()] = True
                        return True
            except Exception as e:
                print('Wordnet error a while checking synsets for ', token, e)

        self.abstractTraitNouns[token.text.lower()] = False
        return False

    # preparation -- grab relevant wordnet synsets
    # to support the animacy function
    organism = wordnet.synsets('organism')
    social_group = wordnet.synsets('social_group')
    people = wordnet.synsets("people")
    human_beings = wordnet.synsets("human_beings")
    ethnos = wordnet.synsets("ethnos")
    race = wordnet.synsets("race")
    population = wordnet.synsets("population")
    hoi_polloi = wordnet.synsets("hoi_polloi")

    def is_animate(self, token):
        """
         It's useful to measure which NPs in a text are animate.
         A text with a high degree of references to animates may,
         for instance, be more likely to be a narrative. In the
         linguistic literature, texts are often organized to
         prefer nominals high in an concreteness/animacy/referentiality
         hierarchy in rheme position, and nominals low in such a
         hierarchy in theme position
        """

        if token.text.lower() in self.animateNouns:
            return self.animateNouns[token.text.lower()]

        if token.ent_type_ == 'PERSON' \
           or token.ent_type_ == 'GPE' \
           or token.ent_type_ == 'NORP':
            self.animateNouns[token.text.lower()] = True
            return True

        # assume NER-unlabeled proper nouns are probably animate.
        # May be able to improve later with a lookup of human names
        if token.ent_type is None or token.ent_type_ == '' \
           and token.tag_ == 'NNP':
            self.animateNouns[token.text.lower()] = True
            return True

        if token.pos_ == 'PRONOUN' \
           or token.tag_ in ['PRP',
                             'PRP$',
                             'WDT',
                             'WP',
                             'WP$',
                             'WRB',
                             'DT'] \
           and token.doc._.coref_chains is not None:
            try:
                antecedents = token.doc._.coref_chains.resolve(token)

                if antecedents is not None:
                    for antecedent in antecedents:
                        return self.is_animate(antecedent)
            except Exception as e:
                print('animacy exception', e)
                if token.text.lower() in ['i',
                                          'me',
                                          'my',
                                          'mine',
                                          'we',
                                          'us',
                                          'our',
                                          'ours',
                                          'you',
                                          'your',
                                          'yours',
                                          'he',
                                          'him',
                                          'they',
                                          'them',
                                          'their',
                                          'theirs',
                                          'his',
                                          'she',
                                          'her',
                                          'hers',
                                          'everyone',
                                          'anyone',
                                          'everybody',
                                          'anybody',
                                          'nobody',
                                          'someone',
                                          'somebody',
                                          'myself',
                                          'ourselves',
                                          'yourself',
                                          'yourselves',
                                          'himself',
                                          'herself',
                                          'themselves',
                                          'one',
                                          'oneself',
                                          'oneselves']:
                    self.animateNouns[token.text.lower()] = True
                    return True
                self.animateNouns[token.text.lower()] = False
                return False
        if token.text.lower() in ['i',
                                  'me',
                                  'my',
                                  'mine',
                                  'we',
                                  'us',
                                  'our',
                                  'ours',
                                  'you',
                                  'your',
                                  'yours',
                                  'he',
                                  'him',
                                  'they',
                                  'them',
                                  'their',
                                  'theirs',
                                  'his',
                                  'she',
                                  'her',
                                  'hers',
                                  'everyone',
                                  'anyone',
                                  'everybody',
                                  'anybody',
                                  'nobody',
                                  'someone',
                                  'somebody',
                                  'myself',
                                  'ourselves',
                                  'yourself',
                                  'yourselves',
                                  'himself',
                                  'herself',
                                  'themselves',
                                  'one',
                                  'oneself',
                                  'oneselves']:
            self.animateNouns[token.text.lower()] = True
            return True

        person = token.doc.vocab.get_vector("person")
        company = token.doc.vocab.get_vector("company")
        try:
            # We occasionally get invalid vectors if the token is not
            # a normal content word. It's hard to detect in advance.
            # TBD: put in a better check to eliminate this case.
            if not all_zeros(token.vector) \
               and token.pos_ in ['NOUN', 'PRONOUN']:
                if 1 - cosine(person, token.vector) > 0.8:
                    self.animateNouns[token.text.lower()] = True
                    return True
                if 1 - cosine(company, token.vector) > 0.8:
                    self.animateNouns[token.text.lower()] = True
                    return True
        except Exception as e:
            print('Token vector invalid for ', token, e)
        if token.pos_ in ['NOUN', 'PRONOUN']:
            synsets = wordnet.synsets(token.lemma_)
            if len(synsets) > 0 \
               and token.pos_ in ['NOUN', 'PRONOUN']:
                try:
                    hypernyms = set([i for i
                                     in synsets[0].closure(lambda s:
                                                           s.hypernyms())])
                    if self.organism[0] in hypernyms \
                       or self.organism[0] == synsets[0]:
                        self.animateNouns[token.text.lower()] = True
                        return True
                    if self.social_group[0] in hypernyms \
                       or self.social_group[0] == synsets[0]:
                        self.animateNouns[token.text.lower()] = True
                        return True
                    if self.people[0] in hypernyms \
                       or self.people[0] == synsets[0]:
                        self.animateNouns[token.text.lower()] = True
                        return True
                    if self.human_beings[0] in hypernyms \
                       or self.human_beings[0] == synsets[0]:
                        self.animateNouns[token.text.lower()] = True
                        return True
                    if self.ethnos[0] in hypernyms \
                       or self.ethnos[0] == synsets[0]:
                        self.animateNouns[token.text.lower()] = True
                        return True
                    if self.race[2] in hypernyms \
                       or self.race[0] == synsets[0]:
                        self.animateNouns[token.text.lower()] = True
                        return True
                    if self.population[0] in hypernyms \
                       or self.population[0] == synsets[0]:
                        self.animateNouns[token.text.lower()] = True
                        return True
                    if self.hoi_polloi[0] in hypernyms \
                       or self.hoi_polloi[0] == synsets[0]:
                        self.animateNouns[token.text.lower()] = True
                        return True
                except Exception as e:
                    print('Wordnet error b while \
                           checking synsets for ', token, e)

        self.animateNouns[token.text.lower()] = False
        return False

    location = wordnet.synsets('location')
    structure = wordnet.synsets('structure')
    pobject = wordnet.synsets('object')
    group = wordnet.synsets('group')
    loc_sverbs = ['contain',
                  'cover',
                  'include',
                  'occupy']
    loc_overbs = ['abandon',
                  'approach',
                  'clear',
                  'depart',
                  'inhabit',
                  'occupy',
                  'empty',
                  'enter',
                  'escape',
                  'exit',
                  'fill',
                  'leave',
                  'near']
    loc_preps = ['above',
                 'across',
                 'against',
                 'along',
                 'amid',
                 'amidst',
                 'among',
                 'amongst',
                 'around',
                 'at',
                 'athwart',
                 'atop',
                 'before',
                 'below',
                 'beneath',
                 'beside',
                 'between',
                 'betwixt',
                 'beyond',
                 'down',
                 'from',
                 'in',
                 'inside',
                 'into',
                 'near',
                 'off',
                 'on',
                 'opposite'
                 'out',
                 'outside',
                 'over',
                 'through',
                 'throughout',
                 'to',
                 'toward',
                 'under',
                 'up',
                 'within',
                 'without',
                 'yon',
                 'yonder']

    travelV = wordnet.synsets('travel', pos=wordnet.VERB)
    travelN = wordnet.synsets('travel', pos=wordnet.NOUN)
    eventN = wordnet.synsets('event', pos=wordnet.NOUN)

    def is_location(self, token):
        """
         It's useful to measure which NPs in a text are
         location references.
        """

        if token.is_stop:
            return False

        if is_temporal(token):
            return False

        if self.concreteness(token) is not None \
           and self.concreteness(token)<3.5:
            return False

        if self.is_animate(token):
            return False

        if self.abstract_trait(token):
            return False

        if self.is_academic(token):
            return False

        for child in token.children:
            if is_temporal(child):
                return False

        if token.ent_type_ in ['FAC', 'GPE', 'LOC']:
            return True
        elif token.ent_type_ in ['PERSON', 'ORG', 'WORK_OF_ART']:
            return False

        if token.orth_ in ['here',
                           'there',
                           'where',
                           'somewhere',
                           'anywhere']:
            if token.pos_ not in ['ADV', 'PRON'] \
               or token.tag_ == 'EX':
                return False
            if token.i+1 < len(token.doc) and token.nbor(1) is not None \
               and token.nbor(1).orth_ in ['is', 'was', 'are', 'were'] \
               and token.orth_ in ['here', 'there']:
                return False
            return True

        # If a word is object of a locative preposition associated with a
        # motion verb, it's a location
        if token.dep_ == 'pobj' \
           and token.head.lemma_ in ['to',
                                     'from',
                                     'in',
                                     'on',
                                     'at',
                                     'upon',
                                     'over',
                                     'under',
                                     'beneath',
                                     'beyond',
                                     'along',
                                     'against',
                                     'through',
                                     'throughout',
                                     'by',
                                     'near',
                                     'into',
                                     'onto',
                                     'off',
                                     'out'] \
           and token.head.head.pos_ in ['VERB']:
            wrdsyns = wordnet.synsets(token.head.head.lemma_,
                                      pos=wordnet.VERB)
            if len(wrdsyns) > 0:
                wrdhyp = set([i for i
                              in wrdsyns[0].closure(lambda s:
                                                    s.hypernyms())])
                if (len(self.eventN) > 0
                    and len(wrdsyns) > 0
                    and (self.eventN[0] in wrdhyp
                         or self.eventN[0] == wrdsyns[0])):
                    return False

                if (len(self.travelV) > 0
                    and len(wrdsyns) > 0
                    and (self.travelV[0] in wrdhyp
                         or self.travelV[0] == wrdsyns[0])):
                    return True

        # If a word is object of a locative preposition associated with a
        # motion noun, it's a location
        elif (token.dep_ == 'pobj'
              and token.head.lemma_ in ['to',
                                        'from',
                                        'in',
                                        'on',
                                        'at',
                                        'upon',
                                        'over',
                                        'under',
                                        'beneath',
                                        'beyond',
                                        'along',
                                        'against',
                                        'through',
                                        'throughout',
                                        'by',
                                        'near',
                                        'into',
                                        'onto',
                                        'off',
                                        'out']
              and token.head.head.pos_ in ['NOUN']):

            wrdsyns = wordnet.synsets(token.head.head.lemma_, pos=wordnet.NOUN)
            if len(wrdsyns) > 0:
                wrdhyp = set([i for i
                              in wrdsyns[0].closure(lambda s:
                                                    s.hypernyms())])
                if (len(self.eventN) > 0
                    and len(wrdsyns) > 0
                    and (self.eventN[0] in wrdhyp
                         or self.eventN[0] == wrdsyns[0])):
                    return False

                if len(self.travelV) > 0 \
                   and len(wrdsyns) > 0 \
                   and (self.travelV[0] in wrdhyp or
                        self.travelV[0] == wrdsyns[0]):
                    return True

        # If a word is under the wordnet location, structure nodes, or
        # subject or object of location verbs like leave and arrive, it's
        # a location
        if not token.is_stop and token.pos_ in ['NOUN']:
            synsets = wordnet.synsets(token.lemma_)
            if len(synsets) > 0:
                try:
                    hypernyms = set([i for i
                                     in synsets[0].closure(lambda s:
                                                           s.hypernyms())])

                    wrdsyns = wordnet.synsets(token.head.head.lemma_,
                                              pos=wordnet.VERB)
                
                    if (len(self.eventN) > 0
                        and len(wrdsyns) > 0
                        and (self.eventN[0] in hypernyms
                             or self.eventN[0] == wrdsyns[0])):
                        return False

                    if len(self.location) > 0 \
                       and (self.location[0] in hypernyms
                            or self.location[0] == synsets[0]):
                        return True
                    if len(self.structure) > 0 \
                       and (self.structure[0] in hypernyms
                            or self.structure[0] == synsets[0]):
                        return True
                    if not token._.animate \
                       and (token.dep_ == 'pobj'
                            and token.head.lemma_ in self.loc_preps
                            or (token.dep_ == 'pobj'
                                and token.head.lemma_ == 'of'
                                and token.head.head.lemma_ in self.loc_preps)
                            or (token.dep_ == 'pobj'
                                and token.head.lemma_ == 'of'
                                and token.head.head._.location)
                            or (token.dep_ == 'subj'
                                and token.lemma_ in self.loc_sverbs)
                            or (token.dep_ == 'dobj'
                                and token.lemma_ in self.loc_overbs)
                            or (token.dep_ == 'nsubjpass'
                                and token.lemma_ in self.loc_overbs)):
                        if len(self.pobject) > 0 \
                           and (self.pobject[0] in hypernyms
                           or self.pobject[0] == synsets[0]):
                            return True
                        if len(self.group) > 0 \
                           and (self.group[0] in hypernyms
                                or self.group[0] == synsets[0]):
                            return True
                except Exception as e:
                    print('Wordnet error c while \
                           checking synsets for ', token, e)

        return False

    def abstract_traits(self, tokens):
        """
        Get a list of the offets of all the abstract trait nominals in the text
        """
        abstracts = []
        for token in tokens:
            if token._.abstract_trait:
                abstracts.append(1)
            else:
                abstracts.append(0)
        return abstracts

    def animates(self, tokens):
        """
        Get a list of the offets of all the animate nominals in the text
        """
        animates = []
        for token in tokens:
            if token._.animate:
                animates.append(1)
            else:
                animates.append(0)
        return animates

    def deictic(self, token):
        """
         In a concreteness/animacy/referentiality hierarchy, deictic elements
         come highest. They are prototypical rheme elements.
        """
        if token.text.lower() in \
            ['i',
             'me',
             'my',
             'mine',
             'myself',
             'we',
             'us',
             'our',
             'ours',
             'ourselves',
             'you',
             'your',
             'yours',
             'yourself',
             'yourselves',
             'here',
             'there',
             'hither',
             'thither',
             'yonder',
             'yon',
             'now',
             'then',
             'anon',
             'today',
             'tomorrow',
             'yesterday',
             'this',
             'that',
             'these',
             'those'
             ]:
            return True
        return False

    def deictics(self, tokens):
        """
         Get a list of the offset of all deictic elements in the text
        """
        deictics = []
        for token in tokens:
            if self.deictic(token):
                deictics.append(1)
            else:
                deictics.append(0)
        return deictics

    def sylco(self, word):
        """
        from discussion posted to
        https://stackoverflow.com/questions/46759492/syllable-count-in-python

        Fallback to calculate number of syllables for words that aren't in the
        moby hyphenator lexicon.
        """
        word = word.lower()

        if not self.alphanum_word(word):
            return None

        # exception_add are words that need extra syllables
        # exception_del are words that need less syllables

        exception_add = ['serious', 'crucial']
        exception_del = ['fortunately', 'unfortunately']

        co_one = ['cool',
                  'coach',
                  'coat',
                  'coal',
                  'count',
                  'coin',
                  'coarse',
                  'coup',
                  'coif',
                  'cook',
                  'coign',
                  'coiffe',
                  'coof',
                  'court']
        co_two = ['coapt', 'coed', 'coinci']

        pre_one = ['preach']

        syls = 0  # added syllable number
        disc = 0  # discarded syllable number

        # 1) if letters < 3 : return 1
        if len(word) <= 3:
            syls = 1
            return syls

        # 2) if doesn't end with "ted" or "tes" or "ses" or "ied" or "ies",
        # discard "es" and "ed" at the end. If it has only 1 vowel or 1 set
        # of consecutive vowels, discard. (like "speed", "fled" etc.)

        if word[-2:] == "es" or word[-2:] == "ed":
            doubleAndtripple_1 = len(re.findall(r'[eaoui][eaoui]', word))
            if doubleAndtripple_1 > 1 \
               or len(re.findall(r'[eaoui][^eaoui]', word)) > 1:
                if word[-3:] == "ted" \
                   or word[-3:] == "tes" \
                   or word[-3:] == "ses" \
                   or word[-3:] == "ied" \
                   or word[-3:] == "ies":
                    pass
            else:
                disc += 1

        # 3) discard trailing "e", except where ending is "le"

        le_except = ['whole',
                     'mobile',
                     'pole',
                     'male',
                     'female',
                     'hale',
                     'pale',
                     'tale',
                     'sale',
                     'aisle',
                     'whale',
                     'while']

        if word[-1:] == "e":
            if word[-2:] == "le" and word not in le_except:
                pass

            else:
                disc += 1

        # 4) check if consecutive vowels exists, triplets or pairs,
        #    count them as one.

        doubleAndtripple = len(re.findall(r'[eaoui][eaoui]', word))
        tripple = len(re.findall(r'[eaoui][eaoui][eaoui]', word))
        disc += doubleAndtripple + tripple

        # 5) count remaining vowels in word.
        numVowels = len(re.findall(r'[eaoui]', word))

        # 6) add one if starts with "mc"
        if word[:2] == "mc":
            syls += 1

        # 7) add one if ends with "y" but is not surrouned by vowel
        if word[-1:] == "y" and word[-2] not in "aeoui":
            syls += 1

        # 8) add one if "y" is surrounded by non-vowels and is
        #    not in the last word.

        for i, j in enumerate(word):
            if j == "y":
                if (i != 0) and (i != len(word) - 1):
                    if word[i-1] not in "aeoui" and word[i+1] not in "aeoui":
                        syls += 1

        # 9) if starts with "tri-" or "bi-" and is followed by a vowel,
        #    add one.

        if word[:3] == "tri" and word[3] in "aeoui":
            syls += 1

        if word[:2] == "bi" and word[2] in "aeoui":
            syls += 1

        # 10) if ends with "-ian", should be counted as two syllables,
        #  except for "-tian" and "-cian"

        if word[-3:] == "ian" and (word[-4:] != "cian" or word[-4:] != "tian"):
            if word[-4:] == "cian" or word[-4:] == "tian":
                pass
            else:
                syls += 1

        # 11) if starts with "co-" and is followed by a vowel, check if exists
        # in the double syllable dictionary, if not, check if in single
        # dictionary and act accordingly.

        if word[:2] == "co" and word[2] in 'eaoui':

            if word[:4] in co_two or word[:5] in co_two or word[:6] in co_two:
                syls += 1
            elif (word[:4] in co_one
                  or word[:5] in co_one
                  or word[:6] in co_one):
                pass
            else:
                syls += 1

        # 12) if starts with "pre-" and is followed by a vowel, check if
        # exists in the double syllable dictionary, if not, check if in
        # single dictionary and act accordingly.

        if word[:3] == "pre" and word[3] in 'eaoui':
            if word[:6] in pre_one:
                pass
            else:
                syls += 1

        # 13) check for "-n't" and cross match with dictionary to add syllable.

        negative = ["doesn't", "isn't", "shouldn't", "couldn't", "wouldn't"]

        if word[-3:] == "n't":
            if word in negative:
                syls += 1
            else:
                pass

        # 14) Handling the exceptional words.

        if word in exception_del:
            disc += 1

        if word in exception_add:
            syls += 1

        sylcount = numVowels - disc + syls

        if sylcount == 0:
            sylcount = 1
        # calculate the output
        return sylcount

    def is_latinate(self, token: Token):
        if not self.alphanum_word(token.text):
            return None
        if token.text.lower() is not None:
            key1 = self.nlp.vocab.strings[token.text.lower()]
        else:
            key1 is None
        if token.lemma_ is not None:
            key2 = self.nlp.vocab.strings[token.lemma_]
        else:
            key2 is None
        if token._.root is not None:
            key3 = self.nlp.vocab.strings[token._.root]
        else:
            key3 = None
        if key1 is not None and key1 in self.latinate:
            return self.latinate[key1]
        if key2 is not None and key2 in self.latinate:
            return self.latinate[key2]
        if key3 is not None and key3 in self.latinate:
            return self.latinate[key3]
        return None

    def is_academic(self, token: Token):
        if not self.alphanum_word(token.text) or len(token.text)<3:
            return None
        if token.text.lower() is not None:
            key1 = self.nlp.vocab.strings[token.text.lower()]
        else:
            key1 = None
        if token.lemma_ is not None:
            key2 = self.nlp.vocab.strings[token.lemma_]
        else:
            key2 = None
        if token._.root is not None:
            key3 = self.nlp.vocab.strings[token._.root]
        else:
            key3 = None
            
        if key1 is not None and key1 in self.academic \
           or key2 is not None and key2 in self.academic \
           or key3 is not None and key3 in self.academic:
            return 1
        else:
            return 0

    def alphanum_word(self, word: str):
        if not re.match('[-A-Za-z0-9\'.]', word) or re.match('[-\'.]+', word):
            return False
        else:
            return True

    def concreteness(self, token: Token):
        key = self.nlp.vocab.strings[token.orth_]
        POS = token.pos_
        if POS == 'PROPN':
            POS = 'NOUN'
        if POS not in ['NOUN', 'VERB', 'ADJ', 'ADV']:
            return None
        else:
            if key in self.concretes:
                if POS in self.concretes[key]:
                    return self.concretes[key][POS]
            if token.lemma_ is not None:
                key2 = self.nlp.vocab.strings[token.lemma_]
                if key2 in self.concretes:
                    if POS in self.concretes[key2]:
                        return self.concretes[key2][POS]
                if POS == 'ADV':
                    key3 = self.nlp.vocab.strings[token.lemma_]
                    if key3 in self.concretes:
                        if 'ADV' in self.concretes[key3]:
                            return self.concretes[key3]['ADV']
            if token._.root is not None and POS == 'ADJ':
                key3 = self.nlp.vocab.strings[token._.root]
                if key3 in self.concretes:
                    if 'ADJ' in self.concretes[key3]:
                        return self.concretes[key3]['ADJ']
            if token._.root is not None and POS == 'NOUN':
                key3 = self.nlp.vocab.strings[token._.root]
                if key3 in self.concretes:
                    if 'NOUN' in self.concretes[key3]:
                        return self.concretes[key3]['NOUN']
            if token._.root is not None and POS == 'VERB':
                key3 = self.nlp.vocab.strings[token._.root]
                if key3 in self.concretes:
                    if 'VERB' in self.concretes[key3]:
                        return self.concretes[key3]['VERB']
            return None

    def min_root_freq(self, token: Token):
        t1 = token._.root1_freq_HAL
        t2 = token._.root2_freq_HAL
        t3 = token._.root3_freq_HAL
        if t1 is None and t2 is None and t3 is None:
            return None
        if t1 is not None and t2 is None and t3 is None:
            return int(t1)
        if t1 is None and t2 is not None and t3 is None:
            return int(t2)
        if t1 is None and t2 is None and t3 is not None:
            return int(t3)
        elif t1 is not None and t2 is not None and t3 is None:
            return min(int(t1), int(t2))
        elif t1 is not None and t2 is None and t3 is not None:
            return min(int(t1), int(t3))
        elif t1 is None and t2 is not None and t3 is not None:
            return min(int(t2), int(t3))
        elif t1 is not None and t2 is not None and t3 is not None:
            return min(int(t1), int(t2), int(t3))
        else:
            return None
