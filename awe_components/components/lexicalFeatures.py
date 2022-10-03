#!/usr/bin/env python3
# Copyright 2022, Educational Testing Service

import math
import numpy as np
import os
import re
import srsly
import wordfreq
import statistics
import awe_lexica

from varname import nameof
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

    with resources.path('awe_lexica.json_data',
                        'syllables.json') as file:
        SYLLABLES_PATH = file

    with resources.path('awe_lexica.json_data',
                        'roots.json') as file:
        ROOTS_PATH = file

    with resources.path('awe_lexica.json_data',
                        'family_sizes.json') as file:
        FAMILY_SIZES_PATH = file

    with resources.path('awe_lexica.json_data',
                        'family_max_freqs.json') as file:
        FAMILY_MAX_FREQS_PATH = file

    with resources.path('awe_lexica.json_data',
                        'family_idxs.json') as file:
        FAMILY_IDX_PATH = file

    with resources.path('awe_lexica.json_data',
                        'family_lists.json') as file:
        FAMILY_LISTS_PATH = file

    with resources.path('awe_lexica.json_data',
                        'concretes.json') as file:
        CONCRETES_PATH = file

    with resources.path('awe_lexica.json_data',
                        'morpholex.json') as file:
        MORPHOLEX_PATH = file

    with resources.path('awe_lexica.json_data',
                        'latinate.json') as file:
        LATINATE_PATH = file

    with resources.path('awe_lexica.json_data',
                        'academic.json') as file:
        ACADEMIC_PATH = file

    with resources.path('awe_lexica.json_data',
                        'nMorph_status.json') as file:
        NMORPH_STATUS_PATH = file

    with resources.path('awe_lexica.json_data',
                        'sentiment.json') as file:
        SENTIMENT_PATH = file

    datapaths = [{'pathname': nameof(SYLLABLES_PATH),
                  'value': SYLLABLES_PATH},
                 {'pathname': nameof(ROOTS_PATH),
                  'value': ROOTS_PATH},
                 {'pathname': nameof(FAMILY_SIZES_PATH),
                  'value': FAMILY_SIZES_PATH},
                 {'pathname': nameof(FAMILY_MAX_FREQS_PATH),
                  'value': FAMILY_MAX_FREQS_PATH},
                 {'pathname': nameof(FAMILY_IDX_PATH),
                  'value': FAMILY_IDX_PATH},
                 {'pathname': nameof(FAMILY_LISTS_PATH),
                  'value': FAMILY_LISTS_PATH},
                 {'pathname': nameof(CONCRETES_PATH),
                  'value': CONCRETES_PATH},
                 {'pathname': nameof(MORPHOLEX_PATH),
                  'value': MORPHOLEX_PATH},
                 {'pathname': nameof(LATINATE_PATH),
                  'value': LATINATE_PATH},
                 {'pathname': nameof(ACADEMIC_PATH),
                  'value': ACADEMIC_PATH},
                 {'pathname': nameof(NMORPH_STATUS_PATH),
                  'value': NMORPH_STATUS_PATH},
                 {'pathname': nameof(SENTIMENT_PATH),
                  'value': SENTIMENT_PATH}
                 ]
    nlp = None

    syllables = {}
    roots = {}
    family_sizes = {}
    family_max_freqs = {}
    family_idx = {}
    family_lists = {}
    concretes = {}
    morpholex = {}
    latinate = {}
    nmorph_status = {}
    sentiment = {}
    academic = []
    animateNouns = {}
    abstractTraitNouns = {}

    def set_nlp(self, nlpIn):
        self.nlp = nlpIn

    def package_check(self, lang):
        for path in self.datapaths:
            if not os.path.exists(path['value']):
                raise LexiconMissingError(
                    "Trying to load AWE Workbench Lexicon Module \
                    without {name} datafile".format(name=path['pathname'])
                )

    def load_lexicons(self):
        for path in self.datapaths:
            lexicon_name = \
                path['pathname'].replace('_PATH', '').lower()
            lexicon = getattr(self, lexicon_name)

            # To save memory, use the spacy string hash as key,
            # not the actual text string
            temp = srsly.read_json(path['value'])
            for word in temp:
                if word in self.nlp.vocab.strings:
                    key = self.nlp.vocab.strings[word]
                else:
                    key = self.nlp.vocab.strings.add(word)

                if lexicon_name == 'family_lists':
                    lexicon[word] = temp[word]
                else:
                    if type(lexicon) == list:
                        lexicon.append(key)
                    else:
                        lexicon[key] = temp[word]

                # Note: this code assumes that we already
                # loaded the family_idx and family_list lexicons
                if lexicon_name == 'sentiment':
                    self.add_morphological_relatives(word, key)

    def __call__(self, doc):
        # We're using this component as a wrapper to add access
        # to the lexical features. There is no actual processing of the
        # sentences.
        
        return doc

    def __init__(self, lang="en"):
        super().__init__()
        self.package_check(lang)

    #####################
    # Define extensions  #
    #####################

    def add_extensions(self):

        """
         Funcion to add extensions that allow us to access the various
         lexicons this module is designed to support.
        """
        extensions = [
            {"name": "AWE_Info",
             "method": self.AWE_Info,
             "type": "docspan"},
            {"name": "token_vectors",
             "getter": self.token_vectors,
             "type": "docspan"},
            {"name": "root",
             "getter": self.root,
             "type": "token"},
            {"name": "nSyll",
             "getter": self.nSyll,
             "type": "token"},
            {"name": "sqrtNChars",
             "getter": self.sqrtNChars,
             "type": "token"},
            {"name": "is_latinate",
             "getter": self.is_latinate,
             "type": "token"},
            {"name": "is_academic",
             "getter": self.is_academic,
             "type": "token"},
            {"name": "family_size",
             "getter": self.family_size,
             "type": "token"},
            {"name": "nSenses",
             "getter": self.nSenses,
             "type": "token"},
            {"name": "logNSenses",
             "getter": self.logNSenses,
             "type": "token"},
            {"name": "morphology",
             "getter": self.morphology,
             "type": "token"},
            {"name": "morpholexsegm",
             "getter": self.morpholexsegm,
             "type": "token"},
            {"name": "nMorph",
             "getter": self.nMorph,
             "type": "token"},
            {"name": "root1_freq_HAL",
             "getter": self.root1_freq_HAL,
             "type": "token"},
            {"name": "root2_freq_HAL",
             "getter": self.root2_freq_HAL,
             "type": "token"},
            {"name": "root3_freq_HAL",
             "getter": self.root3_freq_HAL,
             "type": "token"},
            {"name": "root_famSize",
             "getter": self.root_famSize,
             "type": "token"},
            {"name": "root_pfmf",
             "getter": self.root_pfmf,
             "type": "token"},
            {"name": "token_freq",
             "getter": self.token_freq,
             "type": "token"},
            {"name": "lemma_freq",
             "getter": self.lemma_freq,
             "type": "token"},
            {"name": "root_freq",
             "getter": self.root_freq,
             "type": "token"},
            {"name": "min_root_freq",
             "getter": self.min_root_freq,
             "type": "token"},
            {"name": "max_freq",
             "getter": self.max_freq,
             "type": "token"},
            {"name": "concreteness",
             "getter": self.concreteness,
             "type": "token"},
            {"name": "sentiword",
             "getter": self.sentiword,
             "type": "token"},
            {"name": "abstract_trait",
             "getter": self.abstract_trait,
             "type": "token"},
            {"name": "deictic",
             "getter": self.deictic,
             "type": "token"},
            {"name": "animate",
             "getter": self.is_animate,
             "type": "token"},
            {"name": "location",
             "getter": self.is_location,
             "type": "token"}
        ]

        for extension in extensions:
            if extension['type'] == 'docspan':
                if not Doc.has_extension(extension['name']):
                    if 'getter' in extension:
                        Doc.set_extension(extension['name'],
                                          getter=extension['getter'])

                    elif 'method' in extension:
                        Doc.set_extension(extension['name'],
                                          method=extension['method'])
                    else:
                        raise AWE_Workbench_Error('Invalid extension specification ')

                if not Span.has_extension(extension['name']):
                    if 'getter' in extension:
                        Span.set_extension(extension['name'],
                                           getter=extension['getter'])

                    elif 'method' in extension:
                        Span.set_extension(extension['name'],
                                           method=extension['method'])

                    else:
                        raise AWE_Workbench_Error('Invalid extension specification ')

            elif extension['type'] == 'token':

                if 'getter' in extension:
                    if not Token.has_extension(extension['name']):
                        Token.set_extension(extension['name'],
                                            getter=extension['getter'])

                elif 'method' in extension:
                    if not Token.has_extension(extension['name']):
                        Token.set_extension(extension['name'],
                                            method=extension['method'])
                else:
                    raise AWE_Workbench_Error('Invalid extension specification ')
            else:
                raise AWE_Workbench_Error('Invalid extension specification ')

    ###############################################
    # Block where we define getter functions used #
    # by spacy attribute definitions.             #
    ###############################################

    def root(self, token):
        ''' Access the roots dictionary from the token instance
        '''
        if (token.text.lower() in self.nlp.vocab.strings
            and alphanum_word(token.text)
            and self.nlp.vocab.strings[token.text.lower()]
                in self.roots):
            return self.roots[
                self.nlp.vocab.strings[
                    token.text.lower()]]
        elif alphanum_word(token.text):
            return token.lemma_
        else:
            return None

    def nSyll(self, token):
        ''' Get the number of syllables for a Token
            Number of syllables has been validated as a measure
            of vocabulary difficulty
        '''
        if (token.text.lower() in self.nlp.vocab.strings
            and self.nlp.vocab.strings[token.text.lower()]
                in self.syllables):
            return self.syllables[
                self.nlp.vocab.strings[token.text.lower()]]
        else:
            return sylco(token.text.lower())

    def sqrtNChars(self, token):
        ''' Get the number of characters for a Token
        '''
        return math.sqrt(len(token.text))

    def family_size(self, token):
        ''' Word Family Sizes as measured by slightly modified version of #
            Paul Nation's word family list.                               #

          The family size flag identifies the number of morphologically
          related words in this word's word family. Words with larger
          word families have been shown to be, on average, easier
          vocabulary.
        '''
        if self.nlp.vocab.strings[token.text.lower()] in self.family_sizes \
           and alphanum_word(token.text):
            return self.family_sizes[
                self.nlp.vocab.strings[token.text.lower()]]
        else:
            return None

    def nSenses(self, token):
        ''' Sense count measures (using WordNet)

          The number of senses associated with a word is a measure
          of vocabulary difficulty
        '''
        if alphanum_word(token.text) \
           and len(wordnet.synsets(token.lemma_)) > 0:
            return len(wordnet.synsets(token.lemma_))
        else:
            return None

    def logNSenses(self, token):
        ''' The number of senses associated with a word is a measure
            of vocabulary difficulty
        '''
        if alphanum_word(token.text) \
           and len(wordnet.synsets(token.lemma_)) > 0:
            return math.log(len(wordnet.synsets(token.lemma_)))
        else:
            return None

    def morphology(self, token):
        ''' Access the Morpholex morphological dictionary
        '''
        if (token.text is not None
            and self.nlp.vocab.strings[token.text.lower()]
                in self.morpholex):
            return self.morpholex[
                self.nlp.vocab.strings[
                    token.text.lower()]]
        else:
            return None


    def morpholexsegm(self, token):
        ''' Access a string that identifies roots, prefixes, and
            suffixes in the word. Can be processed to identify
            the specific morphemes in a word according to MorphoLex
        '''
        if (token.text is not None
            and self.nlp.vocab.strings[token.text.lower()]
                in self.morpholex):
            return self.morpholex[
                self.nlp.vocab.strings[
                    token.text.lower()]]['MorphoLexSegm']
        else:
            return None

    def nMorph(self, token):
        ''' The number of morphemes in a word is a measure
            of vocabulary difficulty
        '''
        if token.text is not None \
           and token.text.lower() in self.nlp.vocab.strings \
           and alphanum_word(token.text) \
           and self.nlp.vocab.strings[token.text.lower()] \
           in self.nmorph_status:
            return int(self.nmorph_status[
                self.nlp.vocab.strings[token.text.lower()]])
        else:
            return None

    def root1_freq_HAL(self, token):
        ''' The frequency of the 1st root is a measure of
            vocabulary difficulty
        '''
        if (token.lemma_ is not None
            and token.lemma_ in self.nlp.vocab.strings
            and alphanum_word(token.lemma_)
            and self.nlp.vocab.strings[token.lemma_]
                in self.morpholex):
            return float(self.morpholex[
               self.nlp.vocab.strings[
                   token.lemma_]]['ROOT1_Freq_HAL'])
        else:
            return None

    def root2_freq_HAL(self, token):
        ''' The frequency of the 2nd root is a measure of
            vocabulary difficulty
        '''
        if token.lemma_ is not None \
           and token.lemma_ in self.nlp.vocab.strings \
           and alphanum_word(token.lemma_) \
           and self.nlp.vocab.strings[token.lemma_] \
                in self.morpholex \
           and 'ROOT2_Freq_HAL' in \
               self.morpholex[
                   self.nlp.vocab.strings[token.lemma_]]:
            return float(self.morpholex[self.nlp.vocab.strings[
                                  token.lemma_]
                                  ]['ROOT2_Freq_HAL'])
        else:
            return None

    def root3_freq_HAL(self, token):
        ''' The frequency of the 3rd root is a measure of
            vocabulary difficulty
        '''
        if (token.lemma_ is not None
            and token.lemma_ in self.nlp.vocab.strings
            and alphanum_word(token.lemma_)
            and self.nlp.vocab.strings[token.lemma_]
                in self.morpholex) \
            and 'ROOT3_Freq_HAL' \
                in self.morpholex[self.nlp.vocab.strings[
                                  token.lemma_]]:
            return float(self.morpholex[
                self.nlp.vocab.strings[
                    token.lemma_]]['ROOT3_Freq_HAL'])
        else:
            return None

    def root_famSize(self, token):
        ''' The family size of the root is a measure of vocabulary
            difficulty
        '''
        if (token.lemma_ is not None
            and token.lemma_ in self.nlp.vocab.strings
            and alphanum_word(token.lemma_)
            and self.nlp.vocab.strings[token.lemma_]
                in self.morpholex):
            return int(self.morpholex[
                self.nlp.vocab.strings[
                    token.lemma_]]['ROOT1_FamSize'])
        else:
            return None

    def root_pfmf(self, token):
        ''' The percentage of words more frequent in the family size
             is a measure of vocabulary difficulty
        '''
        if token.lemma_ is not None \
           and token.lemma_ in self.nlp.vocab.strings \
           and alphanum_word(token.lemma_) \
           and self.nlp.vocab.strings[token.lemma_] \
                in self.morpholex:
            return float(self.morpholex[
                self.nlp.vocab.strings[
                    token.lemma_]]['ROOT1_PFMF'])
        else:
            return None

    def token_freq(self, token):
        ''' Word frequency is a measure of vocabulary difficulty.
            We can calculate word frequency for the specific token
        '''
        if alphanum_word(token.text):
            return float(wordfreq.zipf_frequency(token.text.lower(), "en"))
        else:
            return None

    def lemma_freq(self, token):
        ''' Word frequency is a measure of vocabulary difficulty.
            We can calculate word frequency for the lemma
        '''
        if alphanum_word(token.lemma_):
            return float(wordfreq.zipf_frequency(token.lemma_, "en"))
        else:
            return None

    def root_freq(self, token):
        ''' Word frequency is a measure of vocabulary difficulty.
            We can calculate word frequency for the word root
        '''
        if token._.root is not None \
           and alphanum_word(token._.root):
            return float(wordfreq.zipf_frequency(token._.root, "en"))
        else:
            return None

    def max_freq(self, token):
        ''' Or we can calculate the frequency for the root for a
            whole word family
        '''
        if self.nlp.vocab.strings[token.text.lower()] in self.roots \
           and self.roots[self.nlp.vocab.strings[token.text.lower()]] \
           in self.family_max_freqs:
            return float(self.family_max_freqs[
                self.roots[self.nlp.vocab.strings[
                    token.text.lower()]]])
        else:
            return float(wordfreq.zipf_frequency(token.lemma_, "en"))

    def sentiword(self, token):
        '''
          Positive or negative polarity of words as measured by the
          SentiWord database. We also have SpacyTextBlob sentiment,
          which includes two extensions: Token._.polarity
          (for positive/negative sentiment) and Token._.subjectivity,
          which evaluates the subjectivity (stance-taking) valence of
          a word.

          To get SpacyTextBlob polarity, use extension ._.polarity
          to get SpacyTextBlob subjectivity, use extension ._.subjectivity

          to get list of assertion terms recognized by SpacyTextBlob,
          use extension ._.assessments
        '''
        if (token.text.lower() in self.nlp.vocab.strings
            and self.nlp.vocab.strings[token.text.lower()]
                in self.sentiment):
            return self.sentiment[
                self.nlp.vocab.strings[token.text.lower()]]
        else:
            return 0


    def token_vectors(self, document):
        ''' Extensions to allow us to get vectors for tokens in a spacy
            doc or span
        '''
        return [[token.i, token.vector]
                for token in document
                if token.has_vector
                and not token.is_stop
                and token.tag_ in content_tags]

    def AWE_Info(self, 
                 document: Doc,
                 infoType='Token',
                 indicator='pos_',
                 filters=[],
                 transformations=[],
                 summaryType=None):
        ''' This function provides a general-purpose API for
            obtaining information about indicators reported in
            the AWE Workbench Spacy parse tree. 
           
            This is a general-purpose utility. Cloning inside
            the class to simplify the add_extensions class
        '''
        return AWE_Info(document, infoType, indicator, filters,
                 transformations, summaryType)

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
    cognition = wordnet.synsets('cognition')[0]

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
                    if len(hypernyms) > 0 \
                       and (self.attribute in hypernyms
                            or self.quantity in hypernyms
                            or self.part in hypernyms
                            or self.possession in hypernyms
                            or (self.group[0] in hypernyms
                                and self.vegetation not in hypernyms)
                            or self.gathering in hypernyms
                            or self.magnitude in hypernyms
                            or self.cognition in hypernyms
                            or synsets[0] == self.attribute
                            or synsets[0] == self.quantity
                            or synsets[0] == self.part
                            or synsets[0] == self.possession
                            or synsets[0] == self.group[0]
                            or synsets[0] == self.gathering
                            or synsets[0] == self.magnitude
                            or synsets[0] == self.cognition):
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
    mind = wordnet.synsets("mind")
    thought = wordnet.synsets("thought")

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

        if token.pos_ == 'NOUN' \
           and token.text.lower() in self.animateNouns:
            return self.animateNouns[token.text.lower()]

        # exceptional cases do need to be listed out unfortunately.
        # The problem is that anaphoric elements like 'other'
        # aren't handled currently by coreferee
        if token.pos_ == 'NOUN' \
           and token.lemma_ in ['other']:
            return True

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
           or token.tag_ in possessive_or_determiner \
           and token.doc._.coref_chains is not None:
            try:
                antecedents = [token.doc[index]
                               for index
                               in ResolveReference(token,
                                                   token.doc)]

                if antecedents is not None:
                    for antecedent in antecedents:
                        if antecedent.i != token.i:
                            return self.is_animate(antecedent)
            except Exception as e:
                print('animacy exception', e)
                if token.text.lower() in personal_or_indefinite_pronoun:
                    return True
                return False

        if token.text.lower() in personal_or_indefinite_pronoun:
            return True

        person = token.doc.vocab.get_vector("person")
        company = token.doc.vocab.get_vector("company")
        try:
            # We occasionally get invalid vectors if the token is not
            # a normal content word. It's hard to detect in advance.
            # TBD: put in a better check to eliminate this case.
            if not all_zeros(token.vector) \
               and token.pos_ in ['NOUN']:
                if 1 - cosine(person, token.vector) > 0.8:
                    self.animateNouns[token.text.lower()] = True
                    return True
                if 1 - cosine(company, token.vector) > 0.8:
                    self.animateNouns[token.text.lower()] = True
                    return True
        except Exception as e:
            print('Token vector invalid for ', token, e)

        if token.pos_ in ['NOUN', 'PRON']:
            if token.pos_ == 'NOUN':
                synsets = wordnet.synsets(token.lemma_)
            else:
                antecedents = ResolveReference(token, token.doc)
                if token._.antecedents is not None \
                   and len(token._.antecedents) > 0:
                    synsets = wordnet.synsets(token.doc[antecedents[0]].lemma_)
                else:
                    return False
            if len(synsets) > 0 \
               and token.pos_ in ['NOUN', 'PRON']:
                try:
                    hypernyms = set([i for i
                                     in synsets[0].closure(lambda s:
                                                           s.hypernyms())])
                    if self.organism[0] in hypernyms \
                       or self.organism[0] == synsets[0]:
                        if token.pos_ == 'NOUN':
                            self.animateNouns[token.text.lower()] = True
                        return True
                    if self.social_group[0] in hypernyms \
                       or self.social_group[0] == synsets[0]:
                        if token.pos_ == 'NOUN':
                            self.animateNouns[token.text.lower()] = True
                        return True
                    if self.people[0] in hypernyms \
                       or self.people[0] == synsets[0]:
                        if token.pos_ == 'NOUN':
                            self.animateNouns[token.text.lower()] = True
                        return True
                    if self.human_beings[0] in hypernyms \
                       or self.human_beings[0] == synsets[0]:
                        if token.pos_ == 'NOUN':
                            self.animateNouns[token.text.lower()] = True
                        return True
                    if self.ethnos[0] in hypernyms \
                       or self.ethnos[0] == synsets[0]:
                        if token.pos_ == 'NOUN':
                            self.animateNouns[token.text.lower()] = True
                        return True
                    if self.race[2] in hypernyms \
                       or self.race[0] == synsets[0]:
                        if token.pos_ == 'NOUN':
                            self.animateNouns[token.text.lower()] = True
                        return True
                    if self.population[0] in hypernyms \
                       or self.population[0] == synsets[0]:
                        if token.pos_ == 'NOUN':
                            self.animateNouns[token.text.lower()] = True
                        return True
                    if self.hoi_polloi[0] in hypernyms \
                       or self.hoi_polloi[0] == synsets[0]:
                        if token.pos_ == 'NOUN':
                            self.animateNouns[token.text.lower()] = True
                        return True
                    if self.mind[0] in hypernyms \
                       or self.mind[0] == synsets[0]:
                        if token.pos_ == 'NOUN':
                            self.animateNouns[token.text.lower()] = True
                        return True
                    if self.thought[0] in hypernyms \
                       or self.thought[0] == synsets[0]:
                        if token.pos_ == 'NOUN':
                            self.animateNouns[token.text.lower()] = True
                        return True
                except Exception as e:
                    print('Wordnet error b while \
                           checking synsets for ', token, e)

        if token.pos_ in ['NOUN', 'PROPN']:
            self.animateNouns[token.text.lower()] = False
        return False

    location = wordnet.synsets('location')
    structure = wordnet.synsets('structure')
    pobject = wordnet.synsets('object')
    group = wordnet.synsets('group')

    travelV = wordnet.synsets('travel', pos=wordnet.VERB)
    travelN = wordnet.synsets('travel', pos=wordnet.NOUN)
    eventN = wordnet.synsets('event', pos=wordnet.NOUN)

    def is_event(self, token, eventN):
        wrdsyns = wordnet.synsets(token.head.head.lemma_,
                                  pos=wordnet.VERB)
        if len(wrdsyns) > 0:
            wrdhyp = set([i for i
                          in wrdsyns[0].closure(lambda s:
                                                s.hypernyms())])
        if len(self.eventN) > 0 \
           and len(wrdsyns) > 0 \
           and (self.eventN[0] in wrdhyp
                or self.eventN[0] == wrdsyns[0]):
            return True
        return False

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
           and self.concreteness(token) < 3.5:
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

        if token.orth_ in locative_adverbs:
            if token.pos_ not in ['ADV', 'PRON'] \
               or token.tag_ == existential_there:
                return False
            if token.i+1 < len(token.doc) and token.nbor(1) is not None \
               and token.nbor(1).orth_ in ['is', 'was', 'are', 'were'] \
               and token.orth_ in ['here', 'there']:
                return False
            return True

        # If a word is object of a locative preposition associated with a
        # motion verb, it's a location
        if token.dep_ == 'pobj' \
           and token.head.lemma_ in major_locative_prepositions \
           and token.head.head.pos_ in ['VERB']:

            if self.is_event(token.head, self.eventN):
                return False

            wrdsyns = wordnet.synsets(token.head.head.lemma_,
                                      pos=wordnet.VERB)
            if len(wrdsyns) > 0:
                wrdhyp = set([i for i
                              in wrdsyns[0].closure(lambda s:
                                                    s.hypernyms())])

                if (len(self.travelV) > 0
                    and len(wrdsyns) > 0
                    and (self.travelV[0] in wrdhyp
                         or self.travelV[0] == wrdsyns[0])):
                    return True

        # motion noun, it's a location
        elif (token.dep_ == 'pobj'
              and token.head.lemma_ in major_locative_prepositions
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
                            and token.head.lemma_ in all_locative_prepositions
                            or (token.dep_ == 'pobj'
                                and token.head.lemma_ == 'of'
                                and token.head.head.lemma_
                                in all_locative_prepositions)
                            or (token.dep_ == 'pobj'
                                and token.head.lemma_ == 'of'
                                and token.head.head._.location)
                            or (token.dep_ == 'subj'
                                and token.lemma_ in loc_sverbs)
                            or (token.dep_ == 'dobj'
                                and token.lemma_ in loc_overbs)
                            or (token.dep_ == 'nsubjpass'
                                and token.lemma_ in loc_overbs)):
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
        return False


    def deictic(self, token):
        """
         In a concreteness/animacy/referentiality hierarchy, deictic elements
         come highest. They are prototypical rheme elements.
        """
        if token.text.lower() in deictics:
            return True
        return False

    def deictics(self, tokens):
        """
         Get a list of the offset of all deictic elements in the text
        """
        deictics = []
        for token in tokens:
            if self.deictic(token):
                deictics.append(True)
            else:
                deictics.append(False)
        return deictics

    def is_latinate(self, token: Token):
        ''' The latinate flag identifies words that appear likely to be
            more academic as they are formed using latin or greek prefixes
            nd suffixes/ Latinate words are less likely to be known, all
            other things being equal

            Get flag 1 or 0 indicating whether a word has latinate
            prefixes or suffixes
        '''
        if not alphanum_word(token.text):
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
            return self.latinate[key1] == 1
        if key2 is not None and key2 in self.latinate:
            return self.latinate[key2] == 1
        if key3 is not None and key3 in self.latinate:
            return self.latinate[key3] == 1
        return None

    def is_academic(self, token: Token):
        if not alphanum_word(token.text) or len(token.text) < 3:
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
            return True
        else:
            return False

    # Concreteness is a measure of vocabulary difficulty
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

    def add_morphological_relatives(self, word, key):
        sentlist = []
        # modify the sentiment estimate using word families, but only if
        # no negative prefix or suffixes are involved in the word we are
        # taking the sentiment rating from, and it's not in the very high
        # frequency band.
        if key in self.family_idx \
           and str(self.family_idx[key]) in self.family_lists:
            for item in self.family_lists[str(self.family_idx[key])]:
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
