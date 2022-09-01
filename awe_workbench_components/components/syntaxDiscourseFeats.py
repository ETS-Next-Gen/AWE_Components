# Copyright 2021 Educational Testing Service

import math
import os
import srsly
import imp

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

    TRANSITION_TERMS_PATH = \
        resources.path('awe_workbench_components.json_data',
                       'transition_terms.json')

    TRANSITION_CATEGORIES_PATH = \
        resources.path('awe_workbench_components.json_data',
                       'transition_categories.json')

    transition_terms = {}
    transition_categories = {}

    def package_check(self, lang):
        if not os.path.exists(self.TRANSITION_TERMS_PATH) \
          or not os.path.exists(self.TRANSITION_CATEGORIES_PATH):
            raise LexiconMissingError(
                "Trying to load AWE Workbench Syntaxa and Discourse \
                 Feature Module without supporting datafiles".format(lang)
            )

    def load_lexicons(self, lang):
        self.transition_terms = \
            srsly.read_json(self.TRANSITION_TERMS_PATH)
        self.transition_categories = \
            srsly.read_json(self.TRANSITION_CATEGORIES_PATH)

    def __call__(self, doc):
        # We're using this component as a wrapper to add access
        # to the lexical features. There is no actual parsing of the
        # sentences, except for a scan to label transition terms.
        doc._.transition_word_profile = self.transitionProfile(doc)
        self.quotedText(doc)
        return doc

    ##########################################
    # Define getter functions for attributes #
    ##########################################

    def nParas(self, tokens):
        return len(self.paragraphs(tokens))

    def mnParaLen(self, tokens):
        return summarize(self.paragraphLengths(tokens),
                         summaryType=FType.MEAN)

    def mdParaLen(self, tokens):
        return summarize(self.paragraphLengths(tokens),
                         summaryType=FType.MEDIAN)

    def mxParaLen(self, tokens):
        return summarize(self.paragraphLengths(tokens),
                         summaryType=FType.MAX)

    def minParaLen(self, tokens):
        return summarize(self.paragraphLengths(tokens),
                         summaryType=FType.MIN)

    def stdParaLen(self, tokens):
        return summarize(self.paragraphLengths(tokens),
                         summaryType=FType.STDEV)

    def transCt(self, tokens):
        return tokens._.transition_word_profile[0]

    def transCatCt(self, tokens):
        return len(tokens._.transition_word_profile[1])

    def transTypeCt(self, tokens):
        return len(tokens._.transition_word_profile[2])

    def mdTransDist(self, tokens):
        return summarize(self.transitionDistances(tokens),
                         summaryType=FType.MEDIAN)

    def transterms(self, tokens):
        return tokens._.transition_word_profile[3]

    def mnTransDist(self, tokens):
        return summarize(self.transitionDistances(tokens),
                         summaryType=FType.MEAN)

    def mxTransDist(self, tokens):
        return summarize(self.transitionDistances(tokens),
                         summaryType=FType.MAX)

    def minTransDist(self, tokens):
        return summarize(self.transitionDistances(tokens),
                         summaryType=FType.MIN)

    def stdTransDist(self, tokens):
        return summarize(self.transitionDistances(tokens),
                         summaryType=FType.STDEV)

    def mnSentCoh(self, tokens):
        return summarize(self.interSentenceCohesions(tokens),
                         summaryType=FType.MEAN)

    def mdSentCoh(self, tokens):
        return summarize(self.interSentenceCohesions(tokens),
                         summaryType=FType.MEDIAN)

    def mxSentCoh(self, tokens):
        return summarize(self.interSentenceCohesions(tokens),
                         summaryType=FType.MAX)

    def minSentCoh(self, tokens):
        return summarize(self.interSentenceCohesions(tokens),
                         summaryType=FType.MIN)

    def sdSentCoh(self, tokens):
        return summarize(self.interSentenceCohesions(tokens),
                         summaryType=FType.STDEV)

    def mnSlideCoh(self, tokens):
        return summarize(self.slidingWindowCohesions(tokens),
                         summaryType=FType.MEAN)

    def mdSlideCoh(self, tokens):
        return summarize(self.slidingWindowCohesions(tokens),
                         summaryType=FType.MEDIAN)

    def mxSlideCoh(self, tokoens):
        return summarize(self.slidingWindowCohesions(tokens),
                         summaryType=FType.MAX)

    def minSlideCoh(self, tokens):
        return summarize(self.slidingWindowCohesions(tokens),
                         summaryType=FType.MIN)

    def sdSlideCoh(self, tokens):
        return summarize(self.slidingWindowCohesions(tokens),
                         summaryType=FType.STDEV)

    def nCoref(self, tokens):
        return len(tokens._.coref_chains)

    def mnCorefCL(self, tokens):
        return summarize(self.corefChainLengths(tokens),
                         summaryType=FType.MEAN)

    def mdCorefCL(self, tokens):
        return summarize(self.corefChainLengths(tokens),
                         summaryType=FType.MEDIAN)

    def mxCorefCL(self, tokens):
        return summarize(self.corefChainLengths(tokens),
                         summaryType=FType.MAX)

    def minCorefCL(self, tokens):
        return summarize(self.corefChainLengths(tokens),
                         summaryType=FType.MIN)

    def sdCorefCL(self, tokens):
        return summarize(self.corefChainLengths(tokens),
                         summaryType=FType.STDEV)

    def sentc(self, tokens):
        return len(list(tokens.sents))

    def stopwords(self, tokens):
        return [token.is_stop for token in tokens]

    def mnsentlen(self, tokens):
        return summarize(self.sentenceLens(tokens),
                         summaryType=FType.MEAN)

    def mdsentlen(self, tokens):
        return summarize(self.sentenceLens(tokens),
                         summaryType=FType.MEDIAN)

    def mxsentlen(self, tokens):
        return summarize(self.sentenceLens(tokens),
                         summaryType=FType.MAX)

    def minsentlen(self, tokens):
        return summarize(self.sentenceLens(tokens),
                         summaryType=FType.MIN)

    def stdsentlen(self, tokens):
        return summarize(self.sentenceLens(tokens),
                         summaryType=FType.STDEV)

    def sqsentLens(self, tokens):
        return [math.sqrt(x) for x in self.sentenceLens(tokens)]

    def mnsqsentlen(self, tokens):
        return summarize(tokens._.sqrt_sentence_lengths,
                         summaryType=FType.MEAN)

    def mdsqsentlen(self, tokens):
        return summarize(tokens._.sqrt_sentence_lengths,
                         summaryType=FType.MEDIAN)

    def mxsqsentlen(self, tokens):
        return summarize(tokens._.sqrt_sentence_lengths,
                         summaryType=FType.MAX)

    def minsqsentlen(self, tokens):
        return summarize(tokens._.sqrt_sentence_lengths,
                         summaryType=FType.MIN)

    def stdsqsentlen(self, tokens):
        return summarize(tokens._.sqrt_sentence_lengths,
                         summaryType=FType.STDEV)

    def mnword2root(self, tokens):
        return summarize(self.sentenceRhemes(tokens),
                         summaryType=FType.MEAN)

    def mdword2root(self, tokens):
        return summarize(self.sentenceRhemes(tokens),
                         summaryType=FType.MEDIAN)

    def mxword2root(self, tokens):
        return summarize(self.sentenceRhemes(tokens),
                         summaryType=FType.MAX)

    def minword2root(self, tokens):
        return summarize(self.sentenceRhemes(tokens),
                         summaryType=FType.MIN)

    def sdword2root(self, tokens):
        return summarize(self.sentenceRhemes(tokens),
                         summaryType=FType.STDEV)

    def mnRhemeDepth(self, tokens):
        return summarize(self.syntacticDepthsOfRhemes(tokens),
                         summaryType=FType.MEAN)

    def mdRhemeDepth(self, tokens):
        return summarize(self.syntacticDepthsOfRhemes(tokens),
                         summaryType=FType.MEDIAN)

    def mxRhemeDepth(self, tokens):
        return summarize(self.syntacticDepthsOfRhemes(tokens),
                         summaryType=FType.MAX)

    def minRhemeDepth(self, tokens):
        return summarize(self.syntacticDepthsOfRhemes(tokens),
                         summaryType=FType.MIN)

    def sdRhemeDepth(self, tokens):
        return summarize(self.syntacticDepthsOfRhemes(tokens),
                         summaryType=FType.STDEV)

    def mnThemeDepth(self, tokens):
        return summarize(self.syntacticDepthsOfThemes(tokens),
                         summaryType=FType.MEAN)

    def mdThemeDepth(self, tokens):
        return summarize(self.syntacticDepthsOfThemes(tokens),
                         summaryType=FType.MEDIAN)

    def mxThemeDepth(self, tokens):
        return summarize(self.syntacticDepthsOfThemes(tokens),
                         summaryType=FType.MAX)

    def minThemeDepth(self, tokens):
        return summarize(self.syntacticDepthsOfThemes(tokens),
                         summaryType=FType.MIN)

    def sdThemeDepth(self, tokens):
        return summarize(self.syntacticDepthsOfThemes(tokens),
                         summaryType=FType.STDEV)

    def mnWtDepth(self, tokens):
        return summarize(self.weightedSyntacticDepths(tokens),
                         summaryType=FType.MEAN)

    def mdWtDepth(self, tokens):
        return summarize(self.weightedSyntacticDepths(tokens),
                         summaryType=FType.MEDIAN)

    def mxWtDepth(self, tokens):
        return summarize(self.weightedSyntacticDepths(tokens),
                         summaryType=FType.MAX)

    def minWtDepth(self, tokens):
        return summarize(self.weightedSyntacticDepths(tokens),
                         summaryType=FType.MIN)

    def sdWtDepth(self, tokens):
        return summarize(self.weightedSyntacticDepths(tokens),
                         summaryType=FType.STDEV)

    def mnWtBreadth(self, tokens):
        return summarize(self.weightedSyntacticBreadths(tokens),
                         summaryType=FType.MEAN)

    def mdWtBreadth(self, tokens):
        return summarize(self.weightedSyntacticBreadths(tokens),
                         summaryType=FType.MEDIAN)

    def mxWtBreadth(self, tokens):
        return summarize(self.weightedSyntacticBreadths(tokens),
                         summaryType=FType.MAX)

    def minWtBreadth(self, tokens):
        return summarize(self.weightedSyntacticBreadths(tokens),
                         summaryType=FType.MIN)

    def sdWtBreadth(self, tokens):
        return summarize(self.weightedSyntacticBreadths(tokens),
                         summaryType=FType.STDEV)

    def synVar(self, tokens):
        return len(self.syntacticProfile(tokens))

    def ptscp(self, tokens):
        return [1 if in_past_tense_scope(tok)
                else 0 for tok in tokens]

    def prptscp(self, tokens):
        return sum(self.ptscp(tokens))/len(self.ptscp(tokens))

    def quot(self, tokens):
        return [1 if token._.vwp_quoted
                else 0 for token in tokens]

    def __init__(self, lang="en"):
        super().__init__()
        self.package_check(lang)
        self.load_lexicons(lang)

        # The depth of embedding of a word indicates the
        # complexity of the structure within which it occurs
        Token.set_extension("syntacticDepth",
                            getter=self.syntacticDepth,
                            force=True)

        # However, we may have a more accurate measure of
        # complexity of the context if we consider
        # left-embedding effects, which increase the cognitive cost
        Token.set_extension("weightedSyntacticDepth",
                            getter=self.weightedSyntacticDepth,
                            force=True)

        # We also want to know if sentences are constructed
        # in a purely additive way without any embedding except
        # what can be processed one by one in a simple sequential way
        # Like "Advertisements should be banned because they trick
        # kids so we should stop it and that's really important."
        Token.set_extension("weightedSyntacticBreadth",
                            getter=self.weightedSyntacticBreadth,
                            force=True)

        # List of locations where paragraph breaks occur in the document
        Span.set_extension("paragraph_breaks",
                           getter=self.paragraphs,
                           force=True)
        Doc.set_extension("paragraph_breaks",
                          getter=self.paragraphs,
                          force=True)

        # Number of paragraphs in the document
        Span.set_extension("paragraph_count",
                           getter=self.nParas,
                           force=True)
        Doc.set_extension("paragraph_count",
                          getter=self.nParas,
                          force=True)

        # List of paragraph lengths in document
        Span.set_extension("paragraph_lengths",
                           getter=self.paragraphLengths,
                           force=True)
        Doc.set_extension("paragraph_lengths",
                          getter=self.paragraphLengths,
                          force=True)

        # Mean length of paragraphs in the document
        Span.set_extension("mean_paragraph_length",
                           getter=self.mnParaLen,
                           force=True)
        Doc.set_extension("mean_paragraph_length",
                          getter=self.mnParaLen,
                          force=True)

        # information on simple, compound, complex, and
        # compound/complex sentences
        Span.set_extension('sentence_types',
                           getter=self.sentenceTypes,
                           force=True)
        Doc.set_extension('sentence_types',
                          getter=self.sentenceTypes,
                          force=True)

        # Median length of paragraphs in the document
        Span.set_extension("median_paragraph_length",
                           getter=self.mdParaLen,
                           force=True)
        Doc.set_extension("median_paragraph_length",
                          getter=self.mdParaLen,
                          force=True)

        # Max length of paragraphs in the document
        Span.set_extension("max_paragraph_length",
                           getter=self.mxParaLen,
                           force=True)
        Doc.set_extension("max_paragraph_length",
                          getter=self.mxParaLen,
                          force=True)

        # Min length of paragraphs in the document
        Span.set_extension("min_paragraph_length",
                           getter=self.minParaLen,
                           force=True)
        Doc.set_extension("min_paragraph_length",
                          getter=self.minParaLen,
                          force=True)

        # Standard deviation of paragraph length in the document
        Span.set_extension("stdev_paragraph_length",
                           getter=self.stdParaLen,
                           force=True)
        Doc.set_extension("stdev_paragraph_length",
                          getter=self.stdParaLen,
                          force=True)

        # By default, we do not classify words as transition terms
        # We set the flag true when we identif them later
        if not Token.has_extension('transition'):
            Token.set_extension('transition', default=False)

        if not Token.has_extension('transition_category'):
            Token.set_extension('transition_category', default=None)

        # Document level measure: return full transition word profile data
        Span.set_extension("transition_word_profile",
                           default=None,
                           force=True)
        Doc.set_extension("transition_word_profile",
                          default=None,
                          force=True)

        # Document level measure: total number of transition words and
        # phrases present in text
        Span.set_extension("total_transition_words",
                           getter=self.transCt,
                           force=True)
        Doc.set_extension("total_transition_words",
                          getter=self.transCt,
                          force=True)

        # Document level measure: number of TYPES of transition words
        # present in text
        Span.set_extension("transition_category_count",
                           getter=self.transCatCt,
                           force=True)
        Doc.set_extension("transition_category_count",
                          getter=self.transCatCt,
                          force=True)

        # Document level measure: number of distinct transition words
        # present in text
        Span.set_extension("transition_word_type_count",
                           getter=self.transTypeCt,
                           force=True)
        Doc.set_extension("transition_word_type_count",
                          getter=self.transTypeCt,
                          force=True)

        # Dictionary giving the frequency of transition term categories
        # that appear in the document
        Span.set_extension("transition_words",
                           getter=self.transterms,
                           force=True)
        Doc.set_extension("transition_words",
                          getter=self.transterms,
                          force=True)

        # List of cosine distances between ten-word text windows before/
        # after the transition words and phrases in the document
        Span.set_extension("transition_distances",
                           getter=self.transitionDistances,
                           force=True)
        Doc.set_extension("transition_distances",
                          getter=self.transitionDistances,
                          force=True)

        # Document level measure: mean cosine distance between text windows
        # before/after a transition word or phrase
        Span.set_extension("mean_transition_distance",
                           getter=self.mnTransDist,
                           force=True)
        Doc.set_extension("mean_transition_distance",
                          getter=self.mnTransDist,
                          force=True)

        # Document level measure: median cosine distance between text windows
        # before/after a transition word or phrase
        Span.set_extension("median_transition_distance",
                           getter=self.mdTransDist,
                           force=True)
        Doc.set_extension("median_transition_distance",
                          getter=self.mdTransDist,
                          force=True)

        # Document level measure: maximum cosine distance between text windows
        # before/after a transition word or phrase
        Span.set_extension("max_transition_distance",
                           getter=self.mxTransDist,
                           force=True)
        Doc.set_extension("max_transition_distance",
                          getter=self.mxTransDist,
                          force=True)

        # Document level measure: minimum cosine distance between text windows
        # before/after a transition word or phrase
        Span.set_extension("min_transition_distance",
                           getter=self.minTransDist,
                           force=True)
        Doc.set_extension("min_transition_distance",
                          getter=self.minTransDist,
                          force=True)

        # Document level measure: std. deviation of cosine distance between
        # text windows before/after a transition word or phrase
        Span.set_extension("stdev_transition_distance",
                           getter=self.stdTransDist,
                           force=True)
        Doc.set_extension("stdev_transition_distance",
                          getter=self.stdTransDist,
                          force=True)

        # List of intersentence cohesions (cosine distance between 10-word
        # windows before and after a sentence boundary)
        Span.set_extension("intersentence_cohesions",
                           getter=self.interSentenceCohesions,
                           force=True)
        Doc.set_extension("intersentence_cohesions",
                          getter=self.interSentenceCohesions,
                          force=True)

        # Document level measure: mean similarity between adjacent sentences
        Span.set_extension("mean_sent_cohesion",
                           getter=self.mnSentCoh,
                           force=True)
        Doc.set_extension("mean_sent_cohesion",
                          getter=self.mnSentCoh,
                          force=True)

        # Document level measure: median similarity between adjacent sentences
        Span.set_extension("median_sent_cohesion",
                           getter=self.mdSentCoh,
                           force=True)
        Doc.set_extension("median_sent_cohesion",
                          getter=self.mdSentCoh,
                          force=True)

        # Document level measure: max length similarity between adjacent
        # sentences.
        Span.set_extension("max_sent_cohesion",
                           getter=self.mxSentCoh,
                           force=True)
        Doc.set_extension("max_sent_cohesion",
                          getter=self.mxSentCoh,
                          force=True)

        # Document level measure: min length similarity between adjacent
        # sentences.
        Span.set_extension("min_sent_cohesion",
                           getter=self.minSentCoh,
                           force=True)
        Doc.set_extension("min_sent_cohesion",
                          getter=self.minSentCoh,
                          force=True)

        # Document level measure: st dev of  similarity between adjacent
        # sentences.
        Span.set_extension("stdev_sent_cohesion",
                           getter=self.sdSentCoh,
                           force=True)
        Doc.set_extension("stdev_sent_cohesion",
                          getter=self.sdSentCoh,
                          force=True)

        # List of local cohesion measures calculated by sliding a ten-word
        # window over the text and taking the cosine between that window
        # and the ten words that follow them.
        Span.set_extension("sliding_window_cohesions",
                           getter=self.slidingWindowCohesions,
                           force=True)
        Doc.set_extension("sliding_window_cohesions",
                          getter=self.slidingWindowCohesions,
                          force=True)

        # Document level measure: mean similarity between sliding
        # window of 10 words and the next 10 words
        Span.set_extension("mean_slider_cohesion",
                           getter=self.mnSlideCoh,
                           force=True)
        Doc.set_extension("mean_slider_cohesion",
                          getter=self.mnSlideCoh,
                          force=True)

        # Document level measure: median similarity between sliding
        # window of 10 words and the next 10 words
        Span.set_extension("median_slider_cohesion",
                           getter=self.mdSlideCoh,
                           force=True)
        Doc.set_extension("median_slider_cohesion",
                          getter=self.mdSlideCoh,
                          force=True)

        # Document level measure: max length similarity between sliding
        # window of 10 words and the next 10 words
        Span.set_extension("max_slider_cohesion",
                           getter=self.mxSlideCoh,
                           force=True)
        Doc.set_extension("max_slider_cohesion",
                          getter=self.mxSlideCoh,
                          force=True)

        # Document level measure: min length similarity between sliding
        # window of 10 words and the next 10 words
        Span.set_extension("min_slider_cohesion",
                           getter=self.minSlideCoh,
                           force=True)
        Doc.set_extension("min_slider_cohesion",
                          getter=self.minSlideCoh,
                          force=True)

        # Document level measure: st dev of  similarity between sliding
        # window of 10 words and the next 10 words
        Span.set_extension("stdev_slider_cohesion",
                           getter=self.sdSlideCoh,
                           force=True)
        Doc.set_extension("stdev_slider_cohesion",
                          getter=self.sdSlideCoh,
                          force=True)

        # Document level measure: number of coreferential chains
        Span.set_extension("num_corefs",
                           getter=self.nCoref,
                           force=True)
        Doc.set_extension("num_corefs",
                          getter=self.nCoref,
                          force=True)

        # Get coref chain data using extension ._.coref_chains

        # Document level measure: mean length of coreferential chains
        Span.set_extension("mean_coref_chain_len",
                           getter=self.mnCorefCL,
                           force=True)
        Doc.set_extension("mean_coref_chain_len",
                          getter=self.mnCorefCL,
                          force=True)

        # Document level measure: median length of coreferential chains
        Span.set_extension("median_coref_chain_len",
                           getter=self.mdCorefCL,
                           force=True)
        Doc.set_extension("median_coref_chain_len",
                          getter=self.mdCorefCL,
                          force=True)

        # Document level measure: max length of coreferential chains
        Span.set_extension("max_coref_chain_len",
                           getter=self.mxCorefCL,
                           force=True)
        Doc.set_extension("max_coref_chain_len",
                          getter=self.mxCorefCL,
                          force=True)

        # Document level measure: min length of coreferential chains
        Span.set_extension("min_coref_chain_len",
                           getter=self.minCorefCL,
                           force=True)
        Doc.set_extension("min_coref_chain_len",
                          getter=self.minCorefCL,
                          force=True)

        # Document level measure: st dev of length of coreferential chains
        Span.set_extension("stdev_coref_chain_len",
                           getter=self.sdCorefCL,
                           force=True)
        Doc.set_extension("stdev_coref_chain_len",
                          getter=self.sdCorefCL,
                          force=True)

        # Document level measure: sentence count
        Span.set_extension("sentence_count",
                           getter=self.sentc,
                           force=True)
        Doc.set_extension("sentence_count",
                          getter=self.sentc,
                          force=True)

        # List of sentence lengths for document
        Span.set_extension("sentence_lengths",
                           getter=self.sentenceLens,
                           force=True)
        Doc.set_extension("sentence_lengths",
                          getter=self.sentenceLens,
                          force=True)

        # List of sentence lengths for document
        Span.set_extension("stopwords",
                           getter=self.stopwords,
                           force=True)
        Doc.set_extension("stopwords",
                          getter=self.stopwords,
                          force=True)

        # Document level measure: mean sentence length
        Span.set_extension("mean_sentence_len",
                           getter=self.mnsentlen,
                           force=True)
        Doc.set_extension("mean_sentence_len",
                          getter=self.mnsentlen,
                          force=True)

        # Document level measure: median sentence length
        Span.set_extension("median_sentence_len",
                           getter=self.mdsentlen,
                           force=True)
        Doc.set_extension("median_sentence_len",
                          getter=self.mdsentlen,
                          force=True)

        # Document level measure: max sentence length
        Span.set_extension("max_sentence_len",
                           getter=self.mxsentlen,
                           force=True)
        Doc.set_extension("max_sentence_len",
                          getter=self.mxsentlen,
                          force=True)

        # Document level measure: min sentence length
        Span.set_extension("min_sentence_len",
                           getter=self.minsentlen,
                           force=True)
        Doc.set_extension("min_sentence_len",
                          getter=self.minsentlen,
                          force=True)

        # Document level measure: st dev sentence length
        Span.set_extension("std_sentence_len",
                           getter=self.stdsentlen,
                           force=True)
        Doc.set_extension("std_sentence_len",
                          getter=self.stdsentlen,
                          force=True)

        # List of sqrt sentence lengths for document
        Span.set_extension("sqrt_sentence_lengths",
                           getter=self.sqsentLens,
                           force=True)
        Doc.set_extension("sqrt_sentence_lengths",
                          getter=self.sqsentLens,
                          force=True)

        # Document level measure: mean sentence length
        Span.set_extension("mean_sqrt_sentence_len",
                           getter=self.mnsqsentlen,
                           force=True)
        Doc.set_extension("mean_sqrt_sentence_len",
                          getter=self.mnsqsentlen,
                          force=True)

        # Document level measure: median sentence length
        Span.set_extension("median_sqrt_sentence_len",
                           getter=self.mdsqsentlen,
                           force=True)
        Doc.set_extension("median_sqrt_sentence_len",
                          getter=self.mdsqsentlen,
                          force=True)

        # Document level measure: max sentence length
        Span.set_extension("max_sqrt_sentence_len",
                           getter=self.mxsqsentlen,
                           force=True)
        Doc.set_extension("max_sqrt_sentence_len",
                          getter=self.mxsqsentlen,
                          force=True)

        # Document level measure: min sentence length
        Span.set_extension("min_sqrt_sentence_len",
                           getter=self.minsqsentlen,
                           force=True)
        Doc.set_extension("min_sqrt_sentence_len",
                          getter=self.minsqsentlen,
                          force=True)

        # Document level measure: st dev sentence length
        Span.set_extension("std_sqrt_sentence_len",
                           getter=self.stdsqsentlen,
                           force=True)
        Doc.set_extension("std_sqrt_sentence_len",
                          getter=self.stdsqsentlen,
                          force=True)

        # List of sentence rheme lengths (no. words before sentence root)
        Span.set_extension("words_before_sentence_root",
                           getter=self.sentenceRhemes,
                           force=True)
        Doc.set_extension("words_before_sentence_root",
                          getter=self.sentenceRhemes,
                          force=True)

        # Document level measure: mean number of words before main verb
        Span.set_extension("mean_words_to_sentence_root",
                           getter=self.mnword2root,
                           force=True)
        Doc.set_extension("mean_words_to_sentence_root",
                          getter=self.mnword2root,
                          force=True)

        # Document level measure: median number of words before main verb
        Span.set_extension("median_words_to_sentence_root",
                           getter=self.mdword2root,
                           force=True)
        Doc.set_extension("median_words_to_sentence_root",
                          getter=self.mdword2root,
                          force=True)

        # Document level measure: max number of words before main verb
        Span.set_extension("max_words_to_sentence_root",
                           getter=self.mxword2root,
                           force=True)
        Doc.set_extension("max_words_to_sentence_root",
                          getter=self.mxword2root,
                          force=True)

        # Document level measure: min number of words before main verb
        Span.set_extension("min_words_to_sentence_root",
                           getter=self.minword2root,
                           force=True)
        Doc.set_extension("min_words_to_sentence_root",
                          getter=self.minword2root,
                          force=True)

        # Document level measure: st dev of number of words before main verb
        Span.set_extension("stdev_words_to_sentence_root",
                           getter=self.sdword2root,
                           force=True)
        Doc.set_extension("stdev_words_to_sentence_root",
                          getter=self.sdword2root,
                          force=True)

        # List of rheme depths (depth of dependency embedding for words
        # before sentence root)
        Span.set_extension("syntacticRhemeDepths",
                           getter=self.syntacticDepthsOfRhemes,
                           force=True)
        Doc.set_extension("syntacticRhemeDepths",
                          getter=self.syntacticDepthsOfRhemes,
                          force=True)

        # Document level measure: mean depth of syntactic embedding of subject
        # and other pre-main-verb elements
        Span.set_extension("meanRhemeDepth",
                           getter=self.mnRhemeDepth,
                           force=True)
        Doc.set_extension("meanRhemeDepth",
                          getter=self.mnRhemeDepth,
                          force=True)

        # Document level measure: median depth of syntactic embedding of
        # subject and other pre-main-verb elements
        Span.set_extension("medianRhemeDepth",
                           getter=self.mdRhemeDepth,
                           force=True)
        Doc.set_extension("medianRhemeDepth",
                          getter=self.mdRhemeDepth,
                          force=True)

        # Document level measure: max depth of syntactic embedding of subject
        # and other pre-main-verb elements
        Span.set_extension("maxRhemeDepth",
                           getter=self.mxRhemeDepth,
                           force=True)
        Doc.set_extension("maxRhemeDepth",
                          getter=self.mxRhemeDepth,
                          force=True)

        # Document level measure: min depth of syntactic embedding of subject
        # and other pre-main-verb elements
        Span.set_extension("minRhemeDepth",
                           getter=self.minRhemeDepth,
                           force=True)
        Doc.set_extension("minRhemeDepth",
                          getter=self.minRhemeDepth,
                          force=True)

        # Document level measure: standard deviation of depth of syntactic
        # embedding of subject and other pre-main-verb elements
        Span.set_extension("stdevRhemeDepth",
                           getter=self.sdRhemeDepth,
                           force=True)
        Doc.set_extension("stdevRhemeDepth",
                          getter=self.sdRhemeDepth,
                          force=True)

        # List of theme depths (depth of dependency embedding for
        # sentence root and following words)
        Span.set_extension("syntacticThemeDepths",
                           getter=self.syntacticDepthsOfThemes,
                           force=True)
        Doc.set_extension("syntacticThemeDepths",
                          getter=self.syntacticDepthsOfThemes,
                          force=True)

        # Document level measure: mean depth of syntactic embedding
        # of sentence predicate elements
        Span.set_extension("meanThemeDepth",
                           getter=self.mnThemeDepth,
                           force=True)
        Doc.set_extension("meanThemeDepth",
                          getter=self.mnThemeDepth,
                          force=True)

        # Document level measure: median depth of syntactic embedding of \
        # sentence predicate elements
        Span.set_extension("medianThemeDepth",
                           getter=self.mdThemeDepth,
                           force=True)
        Doc.set_extension("medianThemeDepth",
                          getter=self.mdThemeDepth,
                          force=True)

        # Document level measure: max depth of syntactic embedding of sentence
        # predicate elements
        Span.set_extension("maxThemeDepth",
                           getter=self.mxThemeDepth,
                           force=True)
        Doc.set_extension("maxThemeDepth",
                          getter=self.mxThemeDepth,
                          force=True)

        # Document level measure: min depth of syntactic embedding of sentence
        # predicate elements
        Span.set_extension("minThemeDepth",
                           getter=self.minThemeDepth,
                           force=True)
        Doc.set_extension("minThemeDepth",
                          getter=self.minThemeDepth,
                          force=True)

        # Document level measure: st dev of depth of syntactic embedding of
        # sentence predicate elements
        Span.set_extension("stdevThemeDepth",
                           getter=self.sdThemeDepth,
                           force=True)
        Doc.set_extension("stdevThemeDepth",
                          getter=self.sdThemeDepth,
                          force=True)

        # List of weighted syntactic depths for all words in the sentence
        Span.set_extension('weightedSyntacticDepths',
                           getter=self.weightedSyntacticDepths,
                           force=True)
        Doc.set_extension('weightedSyntacticDepths',
                          getter=self.weightedSyntacticDepths,
                          force=True)

        # Document level measure: mean weighted depth of syntactic embedding
        Span.set_extension("meanWeightedDepth",
                           getter=self.mnWtDepth,
                           force=True)
        Doc.set_extension("meanWeightedDepth",
                          getter=self.mnWtDepth,
                          force=True)

        # Document level measure: median weighted depth of syntactic embedding
        Span.set_extension("medianWeightedDepth",
                           getter=self.mdWtDepth,
                           force=True)
        Doc.set_extension("medianWeightedDepth",
                          getter=self.mdWtDepth,
                          force=True)

        # Document level measure: max weighted depth of syntactic embedding
        Span.set_extension("maxWeightedDepth",
                           getter=self.mxWtDepth,
                           force=True)
        Doc.set_extension("maxWeightedDepth",
                          getter=self.mxWtDepth,
                          force=True)

        # Document level measure: min weighted depth of syntactic embedding
        Span.set_extension("minWeightedDepth",
                           getter=self.minWtDepth,
                           force=True)
        Doc.set_extension("minWeightedDepth",
                          getter=self.minWtDepth,
                          force=True)

        # Document level measure: st dev of weighted depth
        # of syntactic embedding
        Span.set_extension("stdevWeightedDepth",
                           getter=self.sdWtDepth,
                           force=True)
        Doc.set_extension("stdevWeightedDepth",
                          getter=self.sdWtDepth,
                          force=True)

        # List of weighted syntactic breadths for all words in the sentence
        Span.set_extension('weightedSyntacticBreadths',
                           getter=self.weightedSyntacticBreadths,
                           force=True)
        Doc.set_extension('weightedSyntacticBreadths',
                          getter=self.weightedSyntacticBreadths,
                          force=True)

        # Document level measure: mean weighted breadth of syntactic embedding
        Span.set_extension("meanWeightedBreadth",
                           getter=self.mnWtBreadth,
                           force=True)
        Doc.set_extension("meanWeightedBreadth",
                          getter=self.mnWtBreadth,
                          force=True)

        # Document level measure: median weighted breadth
        # of syntactic embedding
        Span.set_extension("medianWeightedBreadth",
                           getter=self.mdWtBreadth,
                           force=True)
        Doc.set_extension("medianWeightedBreadth",
                          getter=self.mdWtBreadth,
                          force=True)

        # Document level measure: max weighted breadth of syntactic embedding
        Span.set_extension("maxWeightedBreadth",
                           getter=self.mxWtBreadth,
                           force=True)
        Doc.set_extension("maxWeightedBreadth",
                          getter=self.mxWtBreadth,
                          force=True)

        # Document level measure: min weighted breadth of syntactic embedding
        Span.set_extension("minWeightedBreadth",
                           getter=self.minWtBreadth,
                           force=True)
        Doc.set_extension("minWeightedBreadth",
                          getter=self.minWtBreadth,
                          force=True)

        # Document level measure: st dev of weighted breadth
        # of syntactic embedding
        Span.set_extension("stdevWeightedBreadth",
                           getter=self.sdWtBreadth,
                           force=True)
        Doc.set_extension("stdevWeightedBreadth",
                          getter=self.sdWtBreadth,
                          force=True)

        # Document level measure: syntactic profile vector
        Span.set_extension("syntacticProfile",
                           getter=self.syntacticProfile,
                           force=True)
        Doc.set_extension("syntacticProfile",
                          getter=self.syntacticProfile,
                          force=True)

        # Document level measure: normed syntactic profile vector
        Span.set_extension("syntacticProfileNormed",
                           getter=self.syntacticProfileNormed,
                           force=True)
        Doc.set_extension("syntacticProfileNormed",
                          getter=self.syntacticProfileNormed,
                          force=True)

        # Document level measure: length of syntactic profile vector
        # (syntactic variety)
        Span.set_extension("syntacticVariety",
                           getter=self.synVar,
                           force=True)
        Doc.set_extension("syntacticVariety",
                          getter=self.synVar,
                          force=True)

        Token.set_extension("pastTenseScope",
                            getter=in_past_tense_scope,
                            force=True)

        # Document level measure: list of tokens within past tense clauses
        Span.set_extension("pastTenseScope",
                           getter=self.ptscp,
                           force=True)
        Doc.set_extension("pastTenseScope",
                          getter=self.ptscp,
                          force=True)

        # Document level measure: proportion of tokens inside a past tense
        # clause/sentence. Past tense clauses are strongly associated with
        # narrative.
        Span.set_extension("propn_past",
                           getter=self.prptscp,
                           force=True)
        Doc.set_extension("propn_past",
                          getter=self.prptscp,
                          force=True)

        Token.set_extension('subjectVerbInversion',
                            getter=self.sv_inversion,
                            force=True)

        # Flag that identifies whether a token falls within an
        # explicitly quoted text segment.
        if not Token.has_extension('vwp_quoted'):
            Token.set_extension('vwp_quoted', default=False)

        if not Span.has_extension('vwp_quoted'):
            Span.set_extension('vwp_quoted', getter=self.quot)
        if not Doc.has_extension('vwp_quoted'):
            Doc.set_extension('vwp_quoted', getter=self.quot)

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

    def transitionTerms(self, document: Doc):
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
                        transitionList.append(
                            [gram,
                             start,
                             trans[0],
                             trans[len(trans)-1],
                             'temporal'])
                        document[loc]._.transition_category = 'temporal'
                    i = i + len(trans)
                    continue

            if tok._.vwp_quoted:
                i += 1
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
                transitionList.append(
                    [gram5,
                     document[i].sent.start,
                     i,
                     i + 5,
                     self.transition_categories[
                         self.transition_terms[gram5]]])
            elif gram4 in self.transition_terms:
                for loc in range(i, i + 5):
                    document[loc]._.transition = True
                    document[loc]._.transition_category = \
                        self.transition_categories[
                            self.transition_terms[gram4]]
                transitionList.append(
                    [gram4,
                     document[i].sent.start,
                     i,
                     i + 4,
                     self.transition_categories[
                         self.transition_terms[gram4]]])
            elif gram3 in self.transition_terms:
                for loc in range(i, i + 4):
                    document[loc]._.transition = True
                    document[loc]._.transition_category = \
                        self.transition_categories[
                            self.transition_terms[gram3]]
                transitionList.append(
                    [gram3,
                     document[i].sent.start,
                     i,
                     i + 3,
                     self.transition_categories[
                         self.transition_terms[gram3]]])
            elif gram2 in self.transition_terms:
                for loc in range(i, i + 3):
                    document[loc]._.transition = True
                    document[loc]._.transition_category = \
                        self.transition_categories[
                            self.transition_terms[gram2]]
                transitionList.append(
                    [gram2,
                     document[i].sent.start,
                     i,
                     i + 5,
                     self.transition_categories[
                         self.transition_terms[gram2]]])
            elif gram1 in self.transition_terms:
                for loc in range(i, i + 2):
                    document[loc]._.transition = True
                    document[loc]._.transition_category = \
                        self.transition_categories[
                            self.transition_terms[gram1]]
                transitionList.append(
                    [gram1,
                     document[i].sent.start,
                     i,
                     i + 1,
                     self.transition_categories[
                         self.transition_terms[gram1]]])
            elif (gram0 in self.transition_terms and document[i].tag_ not in
                  ['NN',
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
                   'RP',
                   'GW',
                   'NOUN',
                   'PROPN',
                   'VERB',
                   'ADJ']):
                # basically we require one-word transition terms
                # to be adverbs or function words
                document[i]._.transition = True
                if gram0 in '?!':
                    transitionList.append(
                        [gram0,
                         document[i].sent.start,
                         i,
                         i,
                         self.transition_categories[
                             self.transition_terms[gram0]]])
                    document[i]._.transition_category = \
                        self.transition_categories[
                            self.transition_terms[gram0]]
                elif (document[i].dep_ == 'cc'
                      or document[i].dep_ == 'advmod'):
                    if document[i].head.dep_ is None \
                       or document[i].head.dep_ == 'ROOT' \
                       or document[i].head.head.dep_ is None \
                       or document[i].head.head.dep_ == 'ROOT':
                        transitionList.append(
                            [gram0,
                             document[i].sent.start,
                             i,
                             i,
                             self.transition_categories[
                                 self.transition_terms[gram0]]])
                        document[i]._.transition_category = \
                            self.transition_categories[
                                self.transition_terms[gram0]]
                elif (document[i].head.dep_ is None
                      or document[i].head.dep_ == 'ROOT'):
                    transitionList.append([gram0, document[i].sent.start,
                                           i,
                                           i,
                                           self.transition_categories[
                                               self.transition_terms[gram0]]])
                    document[i]._.transition_category = \
                        self.transition_categories[
                            self.transition_terms[gram0]]
                elif (document[i].head.dep_ in 'advcl'
                      and document[i].head.head.dep_ is None
                      or document[i].head.head.dep_ == 'ROOT'):
                    for item in document[i].head.subtree:
                        transitionList.append(
                            [gram0,
                             document[i].sent.start,
                             i,
                             i,
                             self.transition_categories[
                                self.transition_terms[gram0]]])
                        document[i]._.transition_category = \
                            self.transition_categories[
                                self.transition_terms[gram0]]
                        break
            if document[i].pos_ == 'SPACE' and '\n' in document[i].text:
                # we treat paragraph breaks as another type of transition cue
                transitionList.append(['NEWLINE',
                                       tok.sent.start,
                                       i,
                                       i,
                                       'PARAGRAPH'])
                document[i]._.transition_category = \
                    self.transition_categories[-1]

            i += 1
        return transitionList

    def transitionProfile(self, document: Doc):
        """
         Output a summary of the frequency of transition words overall,
         by category, and by individual expression (plus the base transition
         list that gives the offsets and categories for each detected
         transition word.)
        """
        transitionList = self.transitionTerms(document)
        total = 0
        catProfile = {}
        detProfile = {}
        for item in transitionList:
            total += 1
            if item[4] not in catProfile:
                catProfile[item[4]] = 1
            else:
                catProfile[item[4]] += 1
            if item[0] not in detProfile:
                detProfile[item[0]] = 1
            else:
                detProfile[item[0]] += 1
        return [total, catProfile, detProfile, transitionList]

    def transitionDistances(self, Document: Doc):
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
                    if (Document[i].has_vector
                        and not Document[i].is_stop
                        and Document[i].tag_ in
                        ['NN',
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
                         'CD']):
                        left.append(Document[i].vector)
                    elif Document[i].tag_ in ['PRP',
                                              'PRP$',
                                              'WDT',
                                              'WP',
                                              'WP$',
                                              'WRB',
                                              'DT']:
                        Resolution = ResolveReference(Document[i], Document)
                        if Resolution is not None and len(Resolution) > 0:
                            left.append(sum([Document[item].vector
                                             for item in Resolution]))
                    if (Document[j].has_vector
                        and not Document[j].is_stop
                        and Document[j].tag_ in
                        ['NN',
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
                         'CD']):
                        right.append(Document[j].vector)
                    elif Document[j].tag_ in ['PRP',
                                              'PRP$',
                                              'WDT',
                                              'WP',
                                              'WP$',
                                              'WRB',
                                              'DT']:
                        Resolution = ResolveReference(Document[j], Document)
                        if Resolution is not None and len(Resolution) > 0:
                            right.append(sum([Document[item].vector
                                              for item in Resolution]))
            if len(left) > 0 and len(right) > 0:
                distances.append(cosine(sum(left), sum(right)))
        return distances

    def paragraphs(self, document: Doc):
        """
         Return a list of starting offsets for the paragraphs in document,
         where we assume a line break IS a paragraph break
        """
        paras = []
        for token in document:
            if token.is_space and token.text.count("\n") > 0:
                paras.append(token.i)
        if '\n' not in document[len(document) - 1].text:
            paras.append(len(document) - 1)
        return paras

    def paragraphLengths(self, document: Doc):
        """
         Return list of lengths of paragraphs in document, where we
         assume a line break IS a paragraph break
        """
        lengths = []
        start = 0
        for token in document:
            if token.is_space and token.text.count("\n") > 0:
                lengths.append(token.i - start)
                start = token.i
        if '\n' not in document[len(document) - 1].text:
            lengths.append(token.i - start)
        return lengths

    def sentenceTypes(self, Document: Doc):
        countSimple = 0
        countSimpleComplexPred = 0
        countSimpleCompoundPred = 0
        countSimpleCompoundComplexPred = 0
        countCompound = 0
        countComplex = 0
        countCompoundComplex = 0
        stypes = []
        for sent in Document.sents:
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
                countSimple += 1
                stypes.append(1)
            elif (not compoundS
                  and not complexS
                  and complexPred
                  and not compoundPred):
                countSimpleComplexPred += 1
                stypes.append(2)
            elif (not compoundS
                  and not complexS
                  and not complexPred
                  and compoundPred):
                countSimpleCompoundPred += 1
                stypes.append(3)
            elif (not compoundS
                  and not complexS
                  and complexPred
                  and compoundPred):
                countSimpleCompoundComplexPred += 1
                stypes.append(4)
            elif compoundS and not complexS:
                countCompound += 1
                stypes.append(5)
            elif complexS and not compoundS:
                countComplex += 1
                stypes.append(6)
            elif compoundS and complexS:
                countCompoundComplex += 1
                stypes.append(7)
            else:
                stypes.append(0)
        return (stypes,
                countSimple,
                countSimpleComplexPred,
                countSimpleCompoundPred,
                countSimpleCompoundComplexPred,
                countCompound,
                countComplex,
                countCompoundComplex)

    def corefChainLengths(self, Document: Doc):
        """
         Calculate statistics for the length of chains of coreferential
         nouns/pronouns identified by coreferee. Longer chains implies
         more development of specific topics in the essay.
        """
        lengths = [len(chain) for chain in Document._.coref_chains]
        return lengths

    def interSentenceCohesions(self, Document: Doc):
        """
         Calculate cohesion between adjacent sentences using vector similarity.
         High mean cosines means successive sentences tend to address the
         same content
        """
        lastSentence = None
        similarities = []
        for sentence in Document.sents:
            if lastSentence is not None \
               and sentence.has_vector \
               and lastSentence.has_vector:
                similarities.append(float(sentence.similarity(lastSentence)))
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
            left = []
            right = []
            for j in range(0, 9):
                if Document[i + j] is None:
                    continue
                if Document[i + j + 10] is None:
                    continue
                if (Document[i + j].has_vector
                    and not Document[i + j].is_stop
                    and Document[i + j].tag_ in
                    ['NN',
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
                     'CD']):
                    left.append(Document[i + j].vector)
                elif Document[i + j].tag_ in ['PRP',
                                              'PRP$',
                                              'WDT',
                                              'WP',
                                              'WP$',
                                              'WRB',
                                              'DT']:
                    Resolution = Document[i + j]._.coref_chains.resolve(
                        Document[i + j])
                    if Resolution is not None and len(Resolution) > 0:
                        left.append(sum([item.vector for item in Resolution]))
                if (Document[i + j + 10].has_vector
                    and not Document[i + j + 10].is_stop
                    and Document[i + j + 10].tag_ in
                    ['NN',
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
                     'CD']):
                    right.append(Document[i + j + 10].vector)
                elif Document[i + j + 10].tag_ in ['PRP',
                                                   'PRP$',
                                                   'WDT',
                                                   'WP',
                                                   'WP$',
                                                   'WRB',
                                                   'DT']:
                    Resolution = Document._.coref_chains.resolve(
                        Document[i + j + 10])
                    if Resolution is not None and len(Resolution) > 0:
                        right.append(sum([item.vector for item in Resolution]))
            if len(left) > 0 and len(right) > 0:
                similarities.append(1 - cosine(sum(left), sum(right)))
        return similarities

    def sentenceLens(self, tokens: Doc):
        """
        Return list specifying lengths of sentences in words.
        """
        sentLens = [len(sent) for sent in tokens.sents]
        return sentLens

    def sentenceRhemes(self, tokens: Doc):
        """
        Calculate the length of the rheme (number of words before the main
        verb) in a sentence, using is_sent_start to locate the start of a
        sentence and t.sent.root.text to identify the root of the sentence
        parse tree.

        Longer rhemes tend to correspond to breaks in the cohesion structure
        of a well-written essay. Too many long rhemes thus suggests a lack
        of cohesion.
        """
        currentStart = 0
        offsets = []
        for t in tokens:
            if t.is_sent_start:
                currentStart = t.i
            if t.text == t.sent.root.text:
                offsets.append(t.i - currentStart)
        return offsets

    def syntacticDepth(self, tok: Token, depth=1):
        """
        A simple measure of depth of syntactic embedding
        (number of links to the sentence root)
        """
        if tok.dep_ is None or tok.dep_ == 'ROOT':
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
        if tok.text == tok.sent.root.text:
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

    def weightedSyntacticDepths(self, Document: Doc):
        """
        By summarizing over senteces using the weighted depth measure,
        we attempt to measure the extent to which the writer has used
        really complex elements, especially in subject position or other
        left-branching contexts where it is likely to impose greater processing
        costs on the reader. This may reflect greater complexity of the
        content being expressed.
        """
        depths = []
        for token in Document:
            depths.append(float(self.weightedSyntacticDepth(token)))
        return depths

    def weightedSyntacticBreadths(self, Document: Doc):
        """
        By summarizing over sentences using the weighted breadth measure,
        we attempt to measure the extent to which the writer has mostly used
        a simple additive style in their sentence construction.
        """
        depths = []
        for token in Document:
            depths.append(float(self.weightedSyntacticBreadth(token)))
        return depths

    def syntacticDepthsOfRhemes(self, Document: Doc):
        """
        The early part of the sentence (before the main verb) prototypically
        contains information that is GIVEN -- i.e., it links to what came
        before in the text, which usually means use of simple pronouns and
        referring expressions rather than long, complex elements. So if we
        caculate a summary statistic for the rheme part of the sentence, it
        gives us a measure of the extent to which the writer is keeping the
        rheme simple and hence presumably linked to the rest of the test.
        """
        depths = []
        inRheme = True
        for token in Document:
            depth = float(self.syntacticDepth(token))-1
            if token.is_sent_start:
                inRheme = True
            if depth == 1:
                inRheme = False
            if inRheme:
                depths.append(depth)
        return depths

    def syntacticDepthsOfThemes(self, Document: Doc):
        """
        The theme -- the part of the sentence from the main verb onward --
        is where prototypically new information tends to be put in a sentence.
        Thus if we provide a measure of depth of embedding in this part of the
        sentence, we have a measure of elaboration that focuses on elaboration
        of what is likely to be new content.
        """
        depths = []
        inRheme = True
        for token in Document:
            depth = float(self.syntacticDepth(token))
            if token.is_sent_start:
                inRheme = True
            if depth == 1:
                inRheme = False
            if not inRheme:
                depths.append(depth)
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

    def sv_inversion(self, tok: Token):
        if (tok.lemma_ in ['be', 'have', 'do']
           or tok.tag_ == 'MD'):
            for child in tok.children:
                if child.dep_ in ['nsubj', 'nsubjpass', 'csubj', 'csubjpass'] \
                   and child.i > tok.i:
                    return True
        return False
