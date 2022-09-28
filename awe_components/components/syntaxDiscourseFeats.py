#!/usr/bin/env python3
# Copyright 2022, Educational Testing Service

import math
import os
import srsly
from varname import nameof

from enum import Enum
from spacy.tokens import Doc, Span, Token
from spacy.language import Language

from scipy.spatial.distance import cosine
# Standard cosine distance metric

from .utility_functions import *
from ..errors import *
from importlib import resources

from nltk.corpus import wordnet
# English dictionary. Contains information on senses associated with words
# (a lot more, but that's what we're currently using it for)


@Language.factory("syntaxdiscoursefeatures")
def SyntaxAndDiscourseFeatures(nlp, name):
    return SyntaxAndDiscourseFeatDef()


class SyntaxAndDiscourseFeatDef(object):

    with resources.path('awe_lexica.json_data',
                        'transition_terms.json') as filepath:
        TRANSITION_TERMS_PATH = filepath

    with resources.path('awe_lexica.json_data',
                        'transition_categories.json') as filepath:
        TRANSITION_CATEGORIES_PATH = filepath

    datapaths = [{'pathname': nameof(TRANSITION_TERMS_PATH),
                  'value': TRANSITION_TERMS_PATH},
                 {'pathname': nameof(TRANSITION_CATEGORIES_PATH),
                  'value': TRANSITION_CATEGORIES_PATH}]

    transition_terms = {}
    transition_categories = {}

    def package_check(self, lang):
        for path in self.datapaths:
            if not os.path.exists(path['value']):
                raise LexiconMissingError(
                    "Trying to load AWE Workbench Lexicon Module \
                    without {name} datafile".format(name=path['pathname'])
                )

    def load_lexicons(self, lang):
        self.transition_terms = \
            srsly.read_json(self.TRANSITION_TERMS_PATH)
        self.transition_categories = \
            srsly.read_json(self.TRANSITION_CATEGORIES_PATH)

    def __call__(self, doc):
        # We're using this component as a wrapper to add access
        # to the syntactic features. There is no actual parsing of the
        # sentences, except for a scan to label transition terms.

        self.quotedText(doc)
        doc._.transition_word_profile = self.transitionProfile(doc)

        return doc

    extensions = [{"name": "AWE_Info",
                   "getter": "AWE_Info",
                   "type": "docspan"},
                  {"name": "sentence_types",
                   "getter": "sentence_types",
                   "type": "docspan"},
                  {"name": "transitions",
                   "getter": "transitions",
                   "type": "docspan"},
                  {"name": "transition_word_profile",
                   "getter": "transitionProfile",
                   "type": "docspan"},
                  {"name": "transition_distances",
                   "getter": "transition_distances",
                   "type": "docspan"},
                  {"name": "intersentence_cohesions",
                   "getter": "interSentenceCohesions",
                   "type": "docspan"},
                  {"name": "sliding_window_cohesions",
                   "getter": "slidingWindowCohesions",
                   "type": "docspan"},
                  {"name": "corefChainInfo",
                   "getter": "corefChainInfo",
                   "type": "docspan"},
                  {"name": "sentenceThemes",
                   "getter": "sentenceThemes",
                   "type": "docspan"},
                  {"name": "syntacticDepthsOfRhemes",
                   "getter": "syntacticDepthsOfRhemes",
                   "type": "docspan"},
                  {"name": "syntacticDepthsOfThemes",
                   "getter": "syntacticDepthsOfThemes",
                   "type": "docspan"},
                  {"name": "syntacticProfile",
                   "getter": "syntacticProfile",
                   "type": "docspan"},
                  {"name": "syntacticProfileNormed",
                   "getter": "syntacticProfileNormed",
                   "type": "docspan"},
                  {"name": "syntacticVariety",
                   "getter": "syntacticVariety",
                   "type": "docspan"},
                  {"name": "in_past_tense_scope",
                   "getter": "in_past_tense_scope",
                   "type": "token"},
                  {"name": "subjectVerbInversion",
                   "getter": "sv_inversion",
                   "type": "token"},
                  {"name": "weightedSyntacticDepth",
                   "getter": "weightedSyntacticDepth",
                   "type": "token"},
                  {"name": "weightedSyntacticBreadth",
                   "getter": "weightedSyntacticBreadth",
                   "type": "token"},
                  {"name": "syntacticDepth",
                   "getter": "syntacticDepth",
                   "type": "token"}]

    def add_extensions(self):

        """
         Funcion to add extensions with getter functions that allow us
         to access the various lexicons this module is designed to support.
        """

        for extension in self.extensions:
            if extension['type'] == 'docspan':
                if not Doc.has_extension(extension['name']):
                    Doc.set_extension(extension['name'],
                                      getter=eval('self.'
                                                  + extension['getter']))
                if not Span.has_extension(extension['name']):
                    Span.set_extension(extension['name'],
                                       getter=eval('self.'
                                                   + extension['getter']))
            if extension['type'] == 'token':
                if not Token.has_extension(extension['name']):
                    Token.set_extension(extension['name'],
                                        getter=eval('self.'
                                                    + extension['getter']))
        # By default, we do not classify words as transition terms
        # We set the flag true when we identif them later
        if not Token.has_extension('transition'):
            Token.set_extension('transition', default=False)

        if not Token.has_extension('transition_category'):
            Token.set_extension('transition_category', default=None)

        Token.set_extension('vwp_quoted', default=False, force=True)


    def __init__(self, lang="en"):
        super().__init__()
        self.package_check(lang)
        self.load_lexicons(lang)
        self.add_extensions()

    ##########################################
    # Define getter functions for attributes #
    ##########################################

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
        AWE_Info(document, infoType, indicator, filters,
                 transformations, summaryType)

    def syntacticVariety(self, tokens):
        '''Syntactic variety (number of different dependency patterns
           detected in the text
        '''
        return len(self.syntacticProfile(tokens))

    def in_past_tense_scope(self, tok):
        ''' Past tense scopee (tokens in clauses with past tense verbs)
            Importing from utility functions -- needs to be in-class
            due to the way we add extensions via the add_extension function
        '''
        return in_past_tense_scope(tok)

    def sv_inversion(self, tok: Token):
        if (tok.lemma_ in ['be', 'have', 'do']
           or tok.tag_ == 'MD'):
            for child in tok.children:
                if child.dep_ in ['nsubj', 'nsubjpass', 'csubj', 'csubjpass'] \
                   and child.i > tok.i:
                    return True
        return False

    def quotedText(self, hdoc):
        """
         Mark tokens that occur between punctuation marks as quoted.
         However, do not extend quotations across paragraph boundaries
         (in case of a missing end-punctuation ...)

         This will work well for students who mark quotes properly.
         It will not, of course, help when/if students fail to mark
         quotes or forget end quotes in long paragraphs. Can't be helped.
         So we should interpret the quotedText results with some degree
         of caution.
        """

        inQuote = False
        for token in hdoc:
            if token.tag_ not in ['-LRB-', '-RRB-']:
                if 'Ini' in token.morph.get('PunctSide'):
                    inQuote = True
                elif not inQuote and 'Fin' in token.morph.get('PunctSide'):
                    inQuote = True
                elif 'Fin' in token.morph.get('PunctSide'):
                    inQuote = False
                elif token.text == '“':
                    inQuote = True
                elif token.text == '”':
                    inQuote = False
                elif '\n' in token.text:
                    inQuote = False
                if inQuote:
                    token._.vwp_quoted = True
                if token.text == '\'' \
                   and hdoc[token.head.left_edge.i - 1].text == '\'':
                    token.head._.vwp_quoted = True
                    for child in token.head.subtree:
                        if child.i < token.i:
                            child._.vwp_quoted = True

    def newTransitionEntry(self, startTok, toklen, category):
        entry = {}
        entry['name'] = 'transition'
        entry['startToken'] = startTok.i
        entry['endToken'] = startTok.i + toklen - 1
        entry['offset'] = startTok.idx
        entry['length'] = startTok.doc[entry['endToken']].idx + len(startTok.doc[entry['endToken']].text_with_ws) - startTok.idx
        entry['value'] = category
        entry['text'] = startTok.doc[startTok.i:startTok.i + toklen].text
        return entry

    def transitions(self, document: Doc):
        """
         Transition words and phrases like 'however', 'in any case', 'because'
         play a key role in understanding the structure of a text.
         So we need to detect them, both to display (for feedback purposes) and
         so we can calculate a number of summary statistics that measure the
         complexity of the text structure signals a text sends.

         Pattern match a list of transition terms against the word sequences
         in the document. Return a list of transition terms detected in the
         document, with offsets and the category to which that transition
         term belongs
        """

        transitionList = []

        i = 0
        while i < len(document):

            tok = document[i]
            leftEdge = None
            for item in getRoot(tok).subtree:
                leftEdge = item
                break
            if temporalPhrase(tok) is not None \
                and (tok.head.dep_ is None
                     or tok.head.dep_ is None
                     or tok.head.dep_ == 'ROOT'
                     or tok.i == leftEdge.i
                     or (tok.head.dep_ == 'advcl'
                         and (tok.head.head.dep_ is None
                              or tok.head.head.dep_ == 'ROOT'))):
                start, trans = temporalPhrase(tok)
                gram = ''
                if trans is not None and len(trans) > 0:
                    first = True
                    for loc in trans:
                        if first:
                            gram += document[loc].text
                            first = False
                        else:
                            gram += ' ' + document[loc].text
                        document[loc]._.transition = True
                        document[loc]._.transition_category = 'temporal'

                    if not tok._.vwp_quoted:
                        for loc in trans:
                            document[loc]._.transition = True
                        document[loc]._.transition_category = 'temporal'
                        newEntry = \
                            self.newTransitionEntry(tok,
                                                    len(trans),
                                                    'temporal')
                        transitionList.append(newEntry)

                    i = i + len(trans)
                    continue

            gram0 = None
            gram1 = None
            gram2 = None
            gram3 = None
            gram4 = None
            gram5 = None

            if i + 5 < len(document):
                gram5 = document[i].text.lower()
                for j in range(i + 1, i + 5):
                    gram5 += ' ' + document[j].text.lower()
            if i + 4 < len(document):
                gram4 = document[i].text.lower()
                for j in range(i + 1, i + 4):
                    gram4 += ' ' + document[j].text.lower()
            if i + 3 < len(document):
                gram3 = document[i].text.lower()
                for j in range(i + 1, i + 3):
                    gram3 += ' ' + document[j].text.lower()
            if i + 2 < len(document):
                gram2 = document[i].text.lower()
                for j in range(i + 1, i + 3):
                    gram2 += ' ' + document[j].text.lower()
            if i + 1 < len(document):
                gram1 = document[i].text.lower() \
                    + ' ' + document[i+1].text.lower()
            gram0 = document[i].text.lower()
            if gram5 in self.transition_terms:
                for loc in range(i, i + 6):
                    document[loc]._.transition = True
                    document[loc]._.transition_category = \
                        self.transition_categories[
                            self.transition_terms[gram5]]

                entry = self.newTransitionEntry(tok, 6,
                    self.transition_categories[
                        self.transition_terms[gram5]])
                transitionList.append(entry)

            elif gram4 in self.transition_terms:
                for loc in range(i, i + 5):
                    document[loc]._.transition = True
                    document[loc]._.transition_category = \
                        self.transition_categories[
                            self.transition_terms[gram4]]

                entry = self.newTransitionEntry(tok, 5,
                    self.transition_categories[
                        self.transition_terms[gram4]])
                transitionList.append(entry)

            elif gram3 in self.transition_terms:
                for loc in range(i, i + 4):
                    document[loc]._.transition = True
                    document[loc]._.transition_category = \
                        self.transition_categories[
                            self.transition_terms[gram3]]

                entry = self.newTransitionEntry(tok, 4,
                    self.transition_categories[
                        self.transition_terms[gram3]])
                transitionList.append(entry)

            elif gram2 in self.transition_terms:
                for loc in range(i, i + 3):
                    document[loc]._.transition = True
                    document[loc]._.transition_category = \
                        self.transition_categories[
                            self.transition_terms[gram2]]

                entry = self.newTransitionEntry(tok, 3,
                    self.transition_categories[
                        self.transition_terms[gram2]])
                transitionList.append(entry)

            elif gram1 in self.transition_terms:
                for loc in range(i, i + 2):
                    document[loc]._.transition = True
                    document[loc]._.transition_category = \
                        self.transition_categories[
                            self.transition_terms[gram1]]

                entry = self.newTransitionEntry(tok, 2,
                    self.transition_categories[
                        self.transition_terms[gram1]])
                transitionList.append(entry)


            elif (gram0 in self.transition_terms
                  and (document[i].tag_ not in adj_noun_or_verb
                       or document[i].tag_ == 'NNP')):
                # basically we require one-word transition terms
                # to be adverbs or function words, with the caveat
                # that the parser will sometimes falsely call capitalized
                # transition words proper nouns
                document[i]._.transition = True
                if gram0 in '?!':
                    
                    document[i]._.transition_category = \
                        self.transition_categories[
                            self.transition_terms[gram0]]

                    entry = self.newTransitionEntry(tok, 1,
                        self.transition_categories[
                            self.transition_terms[gram0]])
                    transitionList.append(entry)

                elif (document[i].dep_ == 'cc'
                      or document[i].dep_ == 'advmod'):
                    if document[i].head.dep_ is None \
                       or document[i].head.dep_ == 'ROOT' \
                       or document[i].head.head.dep_ is None \
                       or document[i].head.head.dep_ == 'ROOT':

                        document[i]._.transition_category = \
                            self.transition_categories[
                                self.transition_terms[gram0]]

                        entry = self.newTransitionEntry(tok, 1,
                            self.transition_categories[
                                self.transition_terms[gram0]])
                        transitionList.append(entry)


                elif (document[i].head.dep_ is None
                      or document[i].head.dep_ == 'ROOT'):

                    document[i]._.transition_category = \
                        self.transition_categories[
                            self.transition_terms[gram0]]

                    entry = self.newTransitionEntry(tok, 1,
                        self.transition_categories[
                            self.transition_terms[gram0]])
                    transitionList.append(entry)

                elif (document[i].head.dep_ in 'advcl'
                      and document[i].head.head.dep_ is None
                      or document[i].head.head.dep_ == 'ROOT'):
                    for item in document[i].head.subtree:

                        document[i]._.transition_category = \
                            self.transition_categories[
                                self.transition_terms[gram0]]
                        break
    
                    entry = self.newTransitionEntry(
                        document[i].head.left_edge,
                        1,
                        self.transition_categories[
                            self.transition_terms[gram0]])
                    transitionList.append(entry)

            if document[i].pos_ == 'SPACE' and '\n' in document[i].text:
                # we treat paragraph breaks as another type of transition cue


                document[i]._.transition_category = \
                    self.transition_categories[-1]

                entry = self.newTransitionEntry(tok, 1,
                    self.transition_categories[-1])
                transitionList.append(entry)

            i += 1
        return transitionList

    def transitionProfile(self, document: Doc):
        """
         Output a summary of the frequency of transition words overall,
         by category, and by individual expression (plus the base transition
         list that gives the offsets and categories for each detected
         transition word.)
        """
        transitionList = self.transitions(document)	
        total = 0
        catProfile = {}
        detProfile = {}
        newList = []
        for item in transitionList:
            transition = document[item['startToken']:item['endToken']+1].text
            category = item['value']
            if '\n' in transition:
                transition = 'NEWLINE'
                category = 'PARAGRAPH'
            total += 1
            if item['value'] not in catProfile:
                catProfile[item['value']] = 1
            else:
                catProfile[item['value']] += 1
            if transition not in detProfile:
                detProfile[transition] = 1
            else:
                detProfile[transition] += 1
            newList.append([transition,
                            document[item['startToken']].sent.start,
                            document[item['startToken']].i,
                            document[item['endToken']].i,
                            category])
        return [total, catProfile, detProfile, newList]

    def transition_distances(self, Document: Doc):
        """
         Compare the cosine distances between blocks of 10 words that
         appear BEFORE and AFTER a transition word. We summarize the
         resulting set of cosine distances to yield a score for how big
         a shift occurs at any transition word boundary. Well written text
         will typically include more large shifts in topic between major
         text boundaries, so higher means/medians should be a good thing.

         We use coreferee to resolve pronouns to their antecedents to make the
         transition distance score more accurate.
        """
        distances = []
        start = 0
        end = 0
        transitionList = Document._.transition_word_profile[3]
        for item in transitionList:
            entry = {}
            entry['name'] = 'transitionDistance'
            entry['offset'] = Document[int(item[2])].idx
            entry['length'] = 1
            entry['text'] = item
            entry['length'] = len(item)
            
            start = item[2]
            end = item[3]
            left = []
            right = []
            if start is None:
                continue
            for i in range(int(start) - 10, int(start) - 1):
                if i < 0:
                    continue
                for j in range(int(end) + 1, int(end) + 10):
                    if i >= len(Document) or Document[i] is None:
                        continue
                    if j >= len(Document) or Document[j] is None:
                        continue
                    if Document[i].has_vector \
                       and not Document[i].is_stop \
                       and Document[i].tag_ in content_tags:
                        left.append(Document[i].vector)
                    elif Document[i].tag_ in possessive_or_determiner:
                        Resolution = ResolveReference(Document[i], Document)
                        if Resolution is not None and len(Resolution) > 0:
                            left.append(sum([Document[item].vector
                                             for item in Resolution]))
                    if Document[j].has_vector \
                       and not Document[j].is_stop \
                       and Document[j].tag_ in content_tags:
                        right.append(Document[j].vector)
                    elif Document[j].tag_ in possessive_or_determiner:
                        Resolution = ResolveReference(Document[j], Document)
                        if Resolution is not None and len(Resolution) > 0:
                            right.append(sum([Document[item].vector
                                              for item in Resolution]))
            if len(left) > 0 and len(right) > 0:
                entry['value'] = cosine(sum(left), sum(right))
                distances.append(entry)
        return distances

    def sentence_types(self, Document: Doc):
        '''
            Classify sentences by the syntactic pattern they display --
            specifically, simple kernal sentences, simple sentences
            with complex predicates, simple sentences with compound
            predicates, simple sentences with compound/complex
            predicates, compound sentences, complex sentences, and
            compound/complex sentences
        '''
        stypes = []
        for sent in Document.sents:
            entry = {}
            entry['name'] = 'sentence_type'
            entry['offset'] = Document[sent.start].idx
            entry['length'] = \
                Document[sent.end-1].idx \
                    + len(Document[sent.end-1].text_with_ws) \
                    - Document[sent.start].idx
            entry['startToken'] = sent.start
            entry['endToken'] = sent.end - 1
            entry['text'] = Document[sent.start: sent.end].text

            compoundS = False
            complexS = False
            complexPred = False
            compoundPred = False
            for token in sent:
                if token.dep_ == 'conj' \
                   and (token.pos_ == 'VERB'
                        or token.tag_.startswith('VB')
                        or token.head.pos_ == 'VERB'
                        or token.head.tag_.startswith('VB')) \
                   and tensed_clause(token):
                    compoundS = True
                elif (token.tag_ == 'CC'
                      and len([child.dep_
                               for child in token.head.children
                               if child.dep_ == 'conj']) == 0):
                    compoundS = True
                elif ((token.pos_ == 'VERB'
                       or token.tag_.startswith('VB'))
                      and token.dep_ is not None
                      and token.dep_ not in ['ROOT', 'conj']
                      and tensed_clause(token)):
                    complexS = True
                elif (token.pos_ == 'VERB'
                      and token.dep_ in ['ccomp',
                                         'acomp',
                                         'pcomp',
                                         'xcomp']
                      and not tensed_clause(token)):
                    complexPred = True
                elif ((token.pos_ == 'VERB'
                       or token.tag_.startswith('VB'))
                      and token.dep_ == 'conj'
                      and not tensed_clause(token)):
                    compoundPred = True
            if not compoundS \
               and not complexS \
               and not complexPred \
               and not compoundPred:
                entry['value'] = 'Simple'
                stypes.append(entry)
            elif (not compoundS
                  and not complexS
                  and complexPred
                  and not compoundPred):
                entry['value'] = 'SimpleComplexPred'
                stypes.append(entry)
            elif (not compoundS
                  and not complexS
                  and not complexPred
                  and compoundPred):
                entry['value'] = 'SimpleCompoundPred'
                stypes.append(entry)
            elif (not compoundS
                  and not complexS
                  and complexPred
                  and compoundPred):
                entry['value'] = 'SimpleCompoundComplexPred'
                stypes.append(entry)
            elif compoundS and not complexS:
                entry['value'] = 'Compound'
                stypes.append(entry)
            elif complexS and not compoundS:
                entry['value'] = 'Complex'
                stypes.append(entry)
            elif compoundS and complexS:
                entry['value'] = 'CompoundComplex'
                stypes.append(entry)
            else:
                entry['value'] = 'Other'
                stypes.append(entry)
        return stypes

    def corefChainInfo(self, Document: Doc):
        """
         Calculate statistics for the length of chains of coreferential
         nouns/pronouns identified by coreferee. Longer chains implies
         more development of specific topics in the essay.
        """
        chainInfo = []
        for chain in Document._.coref_chains:
            references = []
            for i in range(0, len(chain)):
                for reference in chain[i]:
                    references.append(reference)
            entry = {}
            entry['name'] = 'coref_chain_lengths'
            entry['text'] = ' '.join([Document[item].text
                                      for item in references])

            entry['length'] = \
                Document[len(references)-1].idx \
                    + len(Document[len(references)-1].text_with_ws) \
                    - Document[references[0]].idx
            entry['startToken'] = references[0]
            entry['endToken'] = references[len(references)-1]
            entry['value'] = references
            chainInfo.append(entry)       
        return chainInfo

    def interSentenceCohesions(self, Document: Doc):
        """
         Calculate cohesion between adjacent sentences using vector similarity.
         High mean cosines means successive sentences tend to address the
         same content
        """
        lastSentence = None
        similarities = []
        for sentence in Document.sents:
            entry = {}
            entry['name'] = 'intersentence_cohesions'
            entry['text'] = Document[sentence.start:sentence.end].text
            entry['offset'] = Document[sentence.start].idx
            entry['length'] = \
                Document[sentence.end-1].idx \
                    + len(Document[sentence.end-1].text_with_ws) \
                    - Document[sentence.start].idx
            entry['startToken'] = sentence.start
            entry['endToken'] = sentence.end - 1
            entry['value'] = None
            if lastSentence is not None \
               and sentence.has_vector \
               and lastSentence.has_vector:
                entry['value'] = float(sentence.similarity(lastSentence))
                similarities.append(entry)
            lastSentence = sentence
        return similarities

    def slidingWindowCohesions(self, Document: Doc):
        """
         Compare the cosine similarity between adjacent blocks of 10 words each
         by sliding a 10-word window over the whole document. We summarize the
         resulting set of cosine similarities to yield a local cohesion score.
         High mean cosines means that drastic shifts in topic are rare within
         the twenty-word span examined by the window.

         We use coreferee to resolve pronouns to their antecedents to make the
         local cohesion score more accurate.
        """
        similarities = []
        for i in range(0, len(Document) - 20):
            entry = {}
            entry['name'] = 'intersentence_cohesions'
            entry['offset'] = Document[i].idx
            entry['length'] = 1
            entry['startToken'] = i
            entry['endToken'] = i
            entry['text'] = Document[i].text
            entry['value'] = None
            left = []
            right = []
            for j in range(0, 9):
                if Document[i + j] is None:
                    continue
                if Document[i + j + 10] is None:
                    continue
                if Document[i + j].has_vector \
                   and not Document[i + j].is_stop \
                   and Document[i + j].tag_ in content_tags:
                    left.append(Document[i + j].vector)
                elif Document[i + j].tag_ in possessive_or_determiner:
                    Resolution = Document[i + j]._.coref_chains.resolve(
                        Document[i + j])
                    if Resolution is not None and len(Resolution) > 0:
                        left.append(sum([item.vector for item in Resolution]))
                if Document[i + j + 10].has_vector \
                   and not Document[i + j + 10].is_stop \
                   and Document[i + j + 10].tag_ in content_tags:
                    right.append(Document[i + j + 10].vector)
                elif Document[i + j + 10].tag_ in possessive_or_determiner:
                    Resolution = Document._.coref_chains.resolve(
                        Document[i + j + 10])
                    if Resolution is not None and len(Resolution) > 0:
                        right.append(sum([item.vector for item in Resolution]))
            if len(left) > 0 and len(right) > 0:
                entry['value'] = 1 - cosine(sum(left), sum(right))
                similarities.append(entry)
        return similarities

    def sentenceThemes(self, tokens: Doc):
        """
        Calculate the length of the theme (number of words before the main
        verb) in a sentence, using is_sent_start to locate the start of a
        sentence and t.sent.root.text to identify the root of the sentence
        parse tree.

        Longer themes tend to correspond to breaks in the cohesion structure
        of a well-written essay. Too many long rhemes thus suggests a lack
        of cohesion.
        """
        currentStart = 0
        offsets = []
        entry = {}
        entry['name'] = 'words2sentenceRoot'
        entry['offset'] = 0
        entry['startToken'] = 0
        entry['value'] = 'theme'
        for t in tokens:
            if t.is_sent_start:
                currentStart = t.i
            if t == t.head or t.dep_ == 'ROOT':
                entry = {}
                entry['length'] = t.idx \
                    + len(t.text_with_ws) \
                    - tokens[currentStart].idx
                entry['offset'] = t.idx
                entry['endToken'] = t.i - 1
                entry['text'] = t.text
                offsets.append(entry)
                entry['name'] = 'words2sentenceRoot'
                entry['offset'] = 0
                entry['startToken'] = currentStart
                entry['value'] = 'theme'
        return offsets

    def syntacticDepth(self, tok: Token, depth=1):
        """
        A simple measure of depth of syntactic embedding
        (number of links to the sentence root)
        """
        if tok.dep_ is None \
           or tok == tok.head \
           or tok.dep_ == 'ROOT':
            return depth
        elif tok.dep_ == 'conj':
            return self.syntacticDepth(tok.head, depth)
        else:
            return self.syntacticDepth(tok.head, depth + 1)

    def weightedSyntacticDepth(self, tok: Token, depth=1):
        """
         We calculate a weighted measure of depth of syntactic
         embedding penalizing sentences for left embedding and
         clauses in subject position. We might be able to improve
         the measure a bit more by considering the syntactic
         concept of heaviness -- longer constituents should as
         a rule be ordered after their shorter sisters within
         a phrase or clause.
        """
        leftBranching = 0
        if tok.i < tok.head.i:
            leftBranching = 1  # penalize for left branching structures
        if tok.text == tok.sent.root.text:
            return depth + leftBranching
        elif tok.dep_ == 'conj':

            # do not penalize depth for coordinate structures
            return self.weightedSyntacticDepth(
                tok.head, depth+leftBranching)
        elif tok.dep_ == 'csubj' or tok.dep_ == 'csubjpass':

            # penalize heavily for embedded subject clauses
            return self.weightedSyntacticDepth(
                tok.head, depth + 3 + leftBranching)
        elif tok.dep_ == 'acl' or tok.dep_ == 'relcl' or tok.dep_ == 'appos' \
                and tok.head.dep_ == 'nsubj' or tok.head.dep_ == 'nsubjpass':

            # penalize for clauses and VPs modifying the subject
            return self.weightedSyntacticDepth(
                tok.head, depth + 2 + leftBranching)
        else:
            return self.weightedSyntacticDepth(
                tok.head, depth + 1 + leftBranching)

    def weightedSyntacticBreadth(self, tok: Token, depth=1):
        """
         We calculate a weighted measure that penalizes tokens
         for being in more loosely connected parts of the text,
         e.g., conjuncts, sentence level modifiers, and the like.
         Some of these are very rare but included for completeness.
         We need to validate that this measure works as expected.
         Note that some of the measures are restricted to right
         branching structures -- we're trying to capture a sense
         of the extent to which the sentence was constructed in
         a flat, additive way, without use of complex embedded
         complement structures.
        """
        if tok.text == tok.sent.root.text \
           or tok == tok.head:
            return depth
        else:
            if tok.dep_ == 'appos' \
               or tok.dep_ == 'advcl' and tok.i > tok.head.i \
               or tok.tag_ == 'SCONJ' and tok.i > tok.head.i \
               or tok.dep_ == 'conj' \
               or tok.dep_ == 'cc' \
               or tok.dep_ == 'discourse' \
               or tok.dep_ == 'dislocated' \
               or tok.dep_ == 'infmod' and tok.i > tok.head.i \
               or tok.dep_ == 'intj' \
               or tok.dep_ == 'list' \
               or tok.dep_ == 'npmod' and tok.i > tok.head.i \
               or tok.dep_ == 'orphan' \
               or tok.dep_ == 'parataxis' \
               or tok.dep_ == 'partmod' and tok.i > tok.head.i \
               or tok.dep_ == 'reparandum' \
               or tok.pos_ == 'PUNCT':
                # appositives
                # adverbial clauses if in theme of sentence
                # conjuncts
                # discourse markers
                # elements moved out of their normal sentence position
                # infinitive modifiers if in theme of sentence
                # interjections
                # list elements
                # adverbial np modifiers
                # orphaned nodes
                # loose adjunction
                # participial modifiers
                # sentence repair
                # punctuation
                return self.weightedSyntacticBreadth(tok.head, depth + 1)
            return self.weightedSyntacticBreadth(tok.head, depth)

    def syntacticDepthsOfThemes(self, Document: Doc):
        """
        The early part of the sentence (before the main verb) prototypically
        contains information that is GIVEN -- i.e., it links to what came
        before in the text, which usually means use of simple pronouns and
        referring expressions rather than long, complex elements. So if we
        caculate a summary statistic for the theme part of the sentence, it
        gives us a measure of the extent to which the writer is keeping the
        rheme simple and hence presumably linked to the rest of the test.
        """
        depths = []
        inTheme = True
        entry = {}
        entry['name'] = 'themeDepth'
        for token in Document:
            depth = int(self.syntacticDepth(token))-1
            if token.is_sent_start:
                inTheme = True
            if depth == 1:
                inTheme = False
                break
            if inTheme:
                entry = {}
                entry['name'] = 'themeDepth'
                entry['offset'] = token.idx
                entry['startToken'] = Document.sent.start
                entry['endToken'] = token.i
                entry['length'] = len(token.text_with_ws)
                entry['value'] = int(depth)
                entry['text'] = token.text
                depths.append(entry)
        return depths

    def syntacticDepthsOfRhemes(self, Document: Doc):
        """
        The theme -- the part of the sentence from the main verb onward --
        is where prototypically new information tends to be put in a sentence.
        Thus if we provide a measure of depth of embedding in this part of the
        sentence, we have a measure of elaboration that focuses on elaboration
        of what is likely to be new content.
        """
        depths = []
        inTheme = True
        for token in Document:
            depth = float(self.syntacticDepth(token))
            if token.is_sent_start:
                inTheme = True
            if depth == 1:
                inTheme = False
            if not inTheme:
                entry = {}
                entry['name'] = 'rhemeDepth'
                entry['offset'] = token.idx
                entry['startToken'] = token.i
                entry['endToken'] = token.i
                entry['length'] = len(token.text_with_ws)
                entry['value'] = int(depth)
                entry['text'] = token.text
                depths.append(entry)
        return depths

    def syntacticProfile(self, Document: Doc, normalized=False):
        """
         Create a syntactic profile by storing a dictionary
         counting the frequency of tag + morphology feature
         combinations and tag-dependency-tag relations
         The resulting profile gives a picture of what
         combination of syntactic patterns appears in the
         student essay. We can either compare overall
         similarity of profiles, or do a simple measure
         of syntactic variety by normalizing the number of
         keys in the syntactic profile against document length
         The individual key/value pairs could be used as predictors
         independently, say to predict writing quality, too, or
         in other ML algorithms
        """
        elements = {}
        for token in Document:
            total = 0
            if token.pos_ not in elements:
                elements[token.pos_] = 1
            else:
                elements[token.pos_] += 1
            total += 1
            for feat in token.morph:
                morphcat = token.pos_
                if token.pos_ == 'AUX' and token.lemma_ == 'be':
                    morphcat += '-BE'
                elif token.pos_ == 'AUX' and token.lemma_ == 'have':
                    morphcat += '-HAVE'
                elif token.pos_ == 'AUX' and token.lemma_ == 'do':
                    morphcat += '-BE'
                elif token.pos_ == 'AUX':
                    morphcat += '-MODAL'

                morphcat += ':' + feat
                if morphcat not in elements:
                    elements[morphcat] = 1
                else:
                    elements[morphcat] += 1
                total += 1
            relation = token.pos_ + '-' + token.dep_ + '-' + token.head.pos_
            if relation not in elements:
                elements[relation] = 1
            else:
                elements[relation] += 1
            total += 1
            if normalized and total > 0:
                for item in elements:
                    elements[item] = elements[item]/total
        return elements

    def syntacticProfileNormed(self, Document: Doc):
        return self.syntacticProfile(Document, normalized=True)
