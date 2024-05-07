#!/usr/bin/env python3
'''
This file provides access to several open-source dictionaries of
lexical features from AWE_Lexica that provide useful features of
words. These include things like:

- Number of syllables
- Root word
- Frequency
- Concrete or abstract
- Academic
- Latinate (using Greek or Latin prefixes)
... and so on.

This is a SpaCy component, which means it is added to the SpaCy pipeline.

Copyright © 2022, Educational Testing Service
'''

import importlib.resources
import math
import numpy as np
import os
import re
from varname import nameof

# English dictionary. Contains information on senses associated with words
# (a lot more, but that's what we're currently using it for)
from nltk.corpus import wordnet
from scipy.spatial.distance import cosine  # Standard cosine distance metric
from spacy.language import Language
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab
import srsly
import statistics
# https://github.com/rspeer/wordfreq
# Large word frequency database. Provides Zipf frequencies
# (log scale of frequency) for most English words, based on a
# variety of corpora.
import wordfreq

import awe_lexica

from .utility_functions import *  # <-- Paul, import only what you need here
from ..errors import LexiconMissingError

def lexicon_path(lexicon):
    '''
    This should move into `AWE_Lexica`. It converts the name of a
    lexicon to its path on disk. E.g. 'academic' will give the
    location of the academic.json_data file from `AWE_Lexica`.
    '''
    # with importlib.resources.path('awe_lexica.json_data', f"{lexicon}.json") as file:
    with importlib.resources.as_file(
        importlib.resources.files('awe_lexica.json_data').joinpath(f"{lexicon}.json")
    ) as file:
        return file


@Language.factory("lexicalfeatures")
def LexicalFeatures(nlp, name):
    '''
    This registers us with SpaCy, and SpaCy will call this ones at pipeline
    definition. This loads several of the AWE_Lexica dictionaries, and so
    is expensive to run. However, in typical usage, SpaCy should not run this
    more than once.
    '''
    ldf = LexicalFeatureDef(nlp)
    return ldf


class LexicalFeatureDef(object):
    '''
    This is our main system.
    '''
    lexica_lexicons = [
        'academic', 'concretes', 'family_idxs', 'family_lists',
        'family_max_freqs', 'family_sizes', 'latinate', 'morpholex',
        'nMorph_status', 'roots', 'sentiment', 'syllables'
    ]

    # These will be populated with our lexicons
    abstractTraitNouns = {}
    academic = []
    animateNouns = {}
    concretes = {}
    family_idxs = {}
    family_lists = {}
    family_max_freqs = {}
    family_sizes = {}
    latinate = {}
    morpholex = {}
    nMorph_status = {}
    roots = {}
    sentiment = {}
    syllables = {}

    def package_check(self, lang):
        '''
        Check whether all the required files exist, and if not, which ones
        are missing.
        '''
        missing_files = []
        for lexicon_name in self.lexica_lexicons:
            path = lexicon_path(lexicon_name)
            if not os.path.exists(path):
                missing_files.append(path)
        if missing_files:
            raise LexiconMissingError(
                "Trying to load AWE Workbench Lexica, but missing:\n{paths}".format(
                    paths="\n".join(map(str, missing_files))
                )
            )

    def load_lexicons(self):
        '''
        Here, we load all of the lexicons. We might consider moving this
        into AWE_Lexica, so these are only loaded once, even if used by
        multiple modules.
        '''
        for lexicon_name in self.lexica_lexicons:
            lexica_data = srsly.read_json(lexicon_path(lexicon_name))

            # To save memory, use the spacy string hash as key,
            # not the actual text string
            lexicon = getattr(self, lexicon_name)

            for word in lexica_data:
                # Get the SpaCy word hash. Add the word if missing.
                key = self.nlp.vocab.strings.add(word)

                if lexicon_name == 'family_lists':
                    lexicon[word] = lexica_data[word]
                else:
                    if type(lexicon) == list:
                        lexicon.append(key)
                    else:
                        lexicon[key] = lexica_data[word]

                # Note: this code assumes that we already
                # loaded the family_idxs and family_list lexicons
                if lexicon_name == 'sentiment':
                    self.add_morphological_relatives(word, key)

        self.academic = set(self.academic)

    def __call__(self, doc):
        '''
        Do nothing; just return the doc.

        We're using this component as a wrapper to add access
        to the lexical features. There is no actual processing of the
        sentences.
        '''
        return doc

    def __init__(self, nlp, lang="en"):
        '''
        This is a fairly heavy call, but we expect it to only be called once.
        '''
        super().__init__()
        self.nlp = nlp
        self.package_check(lang)
        self.load_lexicons()
        self.add_extensions()

    ######################
    # Define extensions  #
    ######################

    def add_extensions(self):
        '''
        Funcion to add extensions that allow us to access the various
        lexicons this module is designed to support.
        '''
        method_extensions = [self.AWE_Info]
        docspan_extensions = [self.token_vectors]
        token_extensions = [self.root, self.nSyll, self.sqrtNChars,
                            self.is_latinate, self.is_academic,
                            self.family_size, self.nSenses,
                            self.logNSenses, self.morphology,
                            self.morpholexsegm, self.nMorph,
                            self.root1_freq_HAL, self.root2_freq_HAL,
                            self.root3_freq_HAL, self.root_famSize,
                            self.root_pfmf, self.token_freq,
                            self.lemma_freq, self.root_freq,
                            self.min_root_freq, self.max_freq,
                            self.concreteness, self.sentiword,
                            self.abstract_trait, self.deictic,
                            self.animate, self.location,
                            self.antecedents, self.usage]

        setExtensionFunctions(method_extensions,
                              docspan_extensions,
                              token_extensions)

        if not Token.has_extension('antecedents_'):
            Token.set_extension('antecedents_', default=None, force=True)

        if not Token.has_extension('usage_'):
            Token.set_extension('usage_', default=None, force=True)

    ###############################################
    # Block where we define getter functions used #
    # by spacy attribute definitions.             #
    ###############################################

    def root(self, token):
        '''
        Access the roots dictionary from the token instance
        '''
        if (token.lower_ in self.nlp.vocab.strings
            and alphanum_word(token.text)
            and self.nlp.vocab.strings[token.lower_]
                in self.roots):
            return self.roots[
                self.nlp.vocab.strings[
                    token.lower_]]
        elif alphanum_word(token.text):
            return token.lemma_
        else:
            return None

    def nSyll(self, token):
        '''
        Get the number of syllables for a Token
        Number of syllables has been validated as a measure
        of vocabulary difficulty
        '''
        if (token.lower_ in self.nlp.vocab.strings
            and self.nlp.vocab.strings[token.lower_]
                in self.syllables):
            return self.syllables[
                self.nlp.vocab.strings[token.lower_]]
        else:
            return sylco(token.lower_)

    def sqrtNChars(self, token):
        '''
        Get the number of characters for a Token
        '''
        return math.sqrt(len(token.text))

    def family_size(self, token):
        '''
        Word Family Sizes as measured by slightly modified version oF
        Paul Nation's word family list.

        The family size flag identifies the number of morphologically
        related words in this word's word family. Words with larger
        word families have been shown to be, on average, easier
        vocabulary.
        '''
        if self.nlp.vocab.strings[token.lower_] in self.family_sizes \
           and alphanum_word(token.text):
            return self.family_sizes[
                self.nlp.vocab.strings[token.lower_]]
        else:
            return None

    def nSenses(self, token):
        '''
        Sense count measures (using WordNet)

        The number of senses associated with a word is a measure
        of vocabulary difficulty
        '''
        if alphanum_word(token.text) \
           and len(wordnet.synsets(token.lemma_)) > 0:
            return len(wordnet.synsets(token.lemma_))
        else:
            return None

    def logNSenses(self, token):
        '''
        The number of senses associated with a word is a measure
        of vocabulary difficulty
        '''
        if alphanum_word(token.text) \
           and len(wordnet.synsets(token.lemma_)) > 0:
            return math.log(len(wordnet.synsets(token.lemma_)))
        else:
            return None

    def morphology(self, token):
        '''
        Access the Morpholex morphological dictionary
        '''
        if (token.text is not None
            and self.nlp.vocab.strings[token.lower_]
                in self.morpholex):
            return self.morpholex[
                self.nlp.vocab.strings[token.lower_]
            ]
        else:
            return None

    def morpholexsegm(self, token):
        '''
        Access a string that identifies roots, prefixes, and
        suffixes in the word. Can be processed to identify
        the specific morphemes in a word according to MorphoLex
        '''
        if (token.text is not None
            and self.nlp.vocab.strings[token.lower_]
                in self.morpholex):
            return self.morpholex[
                self.nlp.vocab.strings[
                    token.lower_]]['MorphoLexSegm']
        else:
            return None

    def nMorph(self, token):
        ''' The number of morphemes in a word is a measure
            of vocabulary difficulty
        '''
        if token.text is not None \
           and token.lower_ in self.nlp.vocab.strings \
           and alphanum_word(token.text) \
           and self.nlp.vocab.strings[token.lower_] \
           in self.nMorph_status:
            return int(self.nMorph_status[
                self.nlp.vocab.strings[token.lower_]])
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
            return float(wordfreq.zipf_frequency(token.lower_, "en"))
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
        if self.nlp.vocab.strings[token.lower_] in self.roots \
           and self.roots[self.nlp.vocab.strings[token.lower_]] \
           in self.family_max_freqs:
            return float(self.family_max_freqs[
                self.roots[self.nlp.vocab.strings[
                    token.lower_]]])
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
        if (token.lower_ in self.nlp.vocab.strings
            and self.nlp.vocab.strings[token.lower_]
                in self.sentiment):
            return self.sentiment[
                self.nlp.vocab.strings[token.lower_]]
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

    def antecedents(self, token):
        ''' Extensions to allow us to get a list of antecedents for
            a pronoun. If wordseqProbabilityServer is running,
            this function will modify coreferee information for
            third person pronoun antecedents to take the selection
            restrictions of the matrix predicate for the pronoun
            into account
        '''
        if token.pos_ == 'PRON' \
           and token._.antecedents_ is None:
            for token in token.doc:
                # Add corrected antecedents to the parse tree
                if token.pos_ == 'PRON':
                    antecedents = ResolveReference(token, token.doc)
                    token._.antecedents_ = antecedents
        else:
            return token._.antecedents_

    def usage(self, token):
        ''' Extension that uses WordNet information about slang
            status to indicate colloquialisms, slang, etc.
        '''
        if token._.usage_ is None:
            self.markSlang(token)
        return token._.usage_

    def markSlang(self, inputtoken):
        for token in inputtoken.doc:
            token._.usage_ = False
            # Use of slang or colloquial expressions
            # If the first sense of the word in WordNet
            # is classified as slang, colloquialism,
            # vulgarism, ethnic slur, or disparagement,
            # classify the word as spoken/interactive
            try:
                s = wordnet.synsets(token.orth_)
                for synset in s:
                    domains = synset.usage_domains()
                    for dom in domains:
                        if dom.lemma_names()[0] == 'slang' \
                           or dom.lemma_names()[0] == 'colloquialism' \
                           or dom.lemma_names()[0] == 'vulgarism' \
                           or dom.lemma_names()[0] == 'ethnic_slur' \
                           or dom.lemma_names()[0] == 'disparagement':
                            # special cases where the colloquial label doesn't
                            # apply to the dominant sense of the word
                            if token.lower_ not in ['think']:
                                token._.usage_ = True
                        break
                    break
            except Exception as e:
                print('No Wordnet synset found for ', token, e)

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

    ###############################################
    # Block of ontological features using WordNet #
    # to identify abstract traits, animacy, loca- #
    # tions, etc.                                 #
    ###############################################

    def abstract_trait(self, token):

        if token.lower_ in self.abstractTraitNouns:
            return self.abstractTraitNouns[token.lower_]

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
                        self.abstractTraitNouns[token.lower_] = True
                        return True
            except Exception as e:
                print('Wordnet error a while checking synsets for ', token, e)

        self.abstractTraitNouns[token.lower_] = False
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

    def animate(self, token):
        '''
         It's useful to measure which NPs in a text are animate.
         A text with a high degree of references to animates may,
         for instance, be more likely to be a narrative. In the
         linguistic literature, texts are often organized to
         prefer nominals high in an concreteness/animacy/referentiality
         hierarchy in rheme position, and nominals low in such a
         hierarchy in theme position
        '''

        # If we already calculated a noun's animacy,
        # we stored it off and can just use the previously
        # calculated value
        if token.pos_ == 'NOUN' \
           and token.lower_ in self.animateNouns:
            return self.animateNouns[token.lower_]

        # exceptional cases do need to be listed out unfortunately.
        # The problem is that anaphoric elements like 'other'
        # aren't handled currently by coreferee
        if token.pos_ == 'NOUN' \
           and token.lemma_ in ['other']:
            return True

        # Named entities classified as person, geopolitial
        # entity, or nationalities or political/religous groups
        # count as animate
        if token.ent_type_ == 'PERSON' \
           or token.ent_type_ == 'GPE' \
           or token.ent_type_ == 'NORP':
            self.animateNouns[token.lower_] = True
            return True

        # assume NER-unlabeled proper nouns are probably animate.
        # May be able to improve later with a lookup of human names
        if token.ent_type is None or token.ent_type_ == '' \
           and token.tag_ == 'NNP':
            self.animateNouns[token.lower_] = True
            return True

        # Use the antecedents of pronouns to calculate their animacy
        if token.pos_ == 'PRONOUN' \
           or token.tag_ in possessive_or_determiner \
           and token.doc._.coref_chains is not None:
            try:
                antecedents = [token.doc[index]
                               for index
                               in ResolveReference(token,
                                                   token.doc)]

                if antecedents is not None:
                    lastAntecedent = None
                for antecedent in antecedents:
                    if antecedent != lastAntecedent \
                       and antecedent.i != token.i:
                        return self.animate(antecedent)
                    lastAntecedent = antecedent
            except Exception as e:
                print('animacy exception', e)
                if token.lower_ in personal_or_indefinite_pronoun:
                    return True
                return False

        # Personal and indefinite pronouns on this list
        # can be assumed to be animate
        if token.lower_ in personal_or_indefinite_pronoun:
            return True

        # similarity to the vector for person or company
        # is another fallback we can use
        person = token.doc.vocab.get_vector("person")
        company = token.doc.vocab.get_vector("company")
        try:
            # We occasionally get invalid vectors if the token is not
            # a normal content word. It's hard to detect in advance.
            # TBD: put in a better check to eliminate this case.
            if not all_zeros(token.vector) \
               and token.pos_ in ['NOUN']:
                if 1 - cosine(person, token.vector) > 0.8:
                    self.animateNouns[token.lower_] = True
                    return True
                if 1 - cosine(company, token.vector) > 0.8:
                    self.animateNouns[token.lower_] = True
                    return True
        except Exception as e:
            print('Token vector invalid for ', token, e)

        # If all other methods fail, we search the wordnet hierarchy
        # for a list of superordinate categories we count as animate
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
                            self.animateNouns[token.lower_] = True
                        return True
                    if self.social_group[0] in hypernyms \
                       or self.social_group[0] == synsets[0]:
                        if token.pos_ == 'NOUN':
                            self.animateNouns[token.lower_] = True
                        return True
                    if self.people[0] in hypernyms \
                       or self.people[0] == synsets[0]:
                        if token.pos_ == 'NOUN':
                            self.animateNouns[token.lower_] = True
                        return True
                    if self.human_beings[0] in hypernyms \
                       or self.human_beings[0] == synsets[0]:
                        if token.pos_ == 'NOUN':
                            self.animateNouns[token.lower_] = True
                        return True
                    if self.ethnos[0] in hypernyms \
                       or self.ethnos[0] == synsets[0]:
                        if token.pos_ == 'NOUN':
                            self.animateNouns[token.lower_] = True
                        return True
                    if self.race[2] in hypernyms \
                       or self.race[0] == synsets[0]:
                        if token.pos_ == 'NOUN':
                            self.animateNouns[token.lower_] = True
                        return True
                    if self.population[0] in hypernyms \
                       or self.population[0] == synsets[0]:
                        if token.pos_ == 'NOUN':
                            self.animateNouns[token.lower_] = True
                        return True
                    if self.hoi_polloi[0] in hypernyms \
                       or self.hoi_polloi[0] == synsets[0]:
                        if token.pos_ == 'NOUN':
                            self.animateNouns[token.lower_] = True
                        return True
                    if self.mind[0] in hypernyms \
                       or self.mind[0] == synsets[0]:
                        if token.pos_ == 'NOUN':
                            self.animateNouns[token.lower_] = True
                        return True
                    if self.thought[0] in hypernyms \
                       or self.thought[0] == synsets[0]:
                        if token.pos_ == 'NOUN':
                            self.animateNouns[token.lower_] = True
                        return True
                except Exception as e:
                    print('Wordnet error b while \
                           checking synsets for ', token, e)

        if token.pos_ in ['NOUN', 'PROPN']:
            self.animateNouns[token.lower_] = False
        return False

    locationSyns = wordnet.synsets('location')
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
        if len(eventN) > 0 \
           and len(wrdsyns) > 0 \
           and (eventN[0] in wrdhyp
                or eventN[0] == wrdsyns[0]):
            return True
        return False

    def location(self, token):
        '''
         It's useful to measure which NPs in a text are
         location references.
        '''

        if token.is_stop:
            return False

        if is_temporal(token):
            return False

        if self.concreteness(token) is not None \
           and self.concreteness(token) < 3.5:
            return False

        if self.animate(token):
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

                    if len(self.locationSyns) > 0 \
                       and (self.locationSyns[0] in hypernyms
                            or self.locationSyns[0] == synsets[0]):
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
        '''
        In a concreteness/animacy/referentiality hierarchy, deictic elements
        come highest. They are prototypical rheme elements.
        '''
        if token.lower_ in deictics:
            return True
        return False

    def deictics(self, tokens):
        '''
        Get a list of the offset of all deictic elements in the text
        '''
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
        if token.lower_ is not None:
            key1 = self.nlp.vocab.strings[token.lower_]
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
        if token.lower_ is not None:
            key1 = self.nlp.vocab.strings[token.lower_]
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
        '''
        This function is part of setup for the sentiment lexicon.
        We modify the sentiment estimate using word families, but only if
        no negative prefix or suffixes are involved in the word we are
        taking the sentiment rating from, and it's not in the very high
        frequency band.
        '''
        sentiment_list = []
        if key in self.family_idxs \
           and str(self.family_idxs[key]) in self.family_lists:
            for item in self.family_lists[str(self.family_idxs[key])]:
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
                    sentiment_list.append(self.sentiment[itemkey])

        if key not in self.sentiment or abs(self.sentiment[key]) <= .2:
            if len(sentiment_list) > 1 and statistics.mean(sentiment_list) > 0:
                self.sentiment[key] = max(sentiment_list)
            elif len(sentiment_list) > 1 and statistics.mean(sentiment_list) < 0:
                self.sentiment[key] = min(sentiment_list)
            elif len(sentiment_list) == 1:
                self.sentiment[key] = sentiment_list[0]

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
