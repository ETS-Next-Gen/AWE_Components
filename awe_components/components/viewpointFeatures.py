#!/usr/bin/env python3
# Copyright 2022, Educational Testing Service

import os
import srsly
import imp

from enum import Enum
from spacy.tokens import Doc, Span, Token
from spacy.language import Language

from scipy.spatial.distance import cosine
# Standard cosine distance metric

from nltk.corpus import wordnet
# (a lot more, but that's what we're currently using it for)

from .utility_functions import *
from ..errors import *
from importlib import resources


def defineMorpholex():
    from awe_components.components.lexicalFeatures import morpholex


@Language.factory("viewpointfeatures", default_config={"fast": False})
def ViewpointFeatures(nlp, name, fast: bool):
    return ViewpointFeatureDef(fast)


class ViewpointFeatureDef:
    """
     This class handles viewpoint and perspective phenomena, including
     recognition of direct vs. indirect speech, finding the implied
     viewpoint of mental state verbs and verbs of saying, recognizing
     stance elements that imply a subjective judgment from a particular
     viewpont, and recognizing other elements that involve explicit
     argumentation.
    """

    with resources.path('awe_lexica.json_data',
                        'stancePerspectiveVoc.json') as filepath:
        STANCE_PERSPECTIVE_PATH = filepath

    with resources.path('awe_lexica.json_data',
                        'morpholex.json') as filepath:
        MORPHOLEX_PATH = filepath

    stancePerspectiveVoc = {}
    morpholex = {}
    is_nominalization = {}

    calculatePerspective = True

    def package_check(self, lang):
        if not os.path.exists(self.STANCE_PERSPECTIVE_PATH):
            raise LexiconMissingError(
                "Trying to load AWE Workbench Syntaxa and Discourse Feature \
                 Module without supporting datafile {}".format(filepath)
            )
        if not os.path.exists(self.MORPHOLEX_PATH):
            raise LexiconMissingError(
                "Trying to load AWE Workbench Syntaxa and Discourse Feature \
                 Module without supporting datafile {}".format(filepath)
            )

    def load_lexicon(self, lang):
        """
         Load the lexicon that contains word class information we will need to
         process perspective and argumentation elements
        """
        self.stancePerspectiveVoc = \
            srsly.read_json(self.STANCE_PERSPECTIVE_PATH)

        # would really rather not reimport, but
        # importing from lexicalFeatures is a bit complicated.
        # TBD: fix this
        self.morpholex = \
            srsly.read_json(self.MORPHOLEX_PATH)

        # Nominalizations are by definition not concrete

        for token in self.morpholex:
            morpholexsegm = \
                self.morpholex[token]['MorphoLexSegm']
            if morpholexsegm.endswith('>ship>') \
               or morpholexsegm.endswith('>hood>') \
               or morpholexsegm.endswith('>ness>') \
               or morpholexsegm.endswith('>less>') \
               or morpholexsegm.endswith('>ion>') \
               or morpholexsegm.endswith('>tion>') \
               or morpholexsegm.endswith('>ity>') \
               or morpholexsegm.endswith('>ty>') \
               or morpholexsegm.endswith('>cy>') \
               or morpholexsegm.endswith('>al>') \
               or morpholexsegm.endswith('>ance>'):
                self.is_nominalization[token] = True
        self.morpholex = None

    def __call__(self, doc):
        """
         Call the spacy component and process the document to register
         viewpoint and stance elements
        """
        for token in doc:
            # Add attributes specified in the stance lexicon
            self.setLexiconAttributes(token, doc)

        # Add general rules for vwp_abstract, above and beyond
        # any special cases set from the stancePerspectiveVoc
        # dictionary
        for token in doc:
            # Low concreteness or in abstract traits
            # or being a nominalization means an abstract word
            if (token._.concreteness is not None
                and token._.concreteness < 3.5) \
               or token._.abstract_trait \
               or token.text.lower() in self.is_nominalization:
                token._.vwp_abstract = True

        for token in doc:

            # Mark implicit subject relationships within each clause
            if isRoot(token):
                self.markImplicitSubject(token, doc)

            # Add corrected antecedents to the parse tree
            if token.pos_ == 'PRON':
                antecedents = ResolveReference(token, doc)
                token._.antecedents = antecedents

        # Mark sentiment properly under scope of negation, using
        # sentiWord weights as base lexical sentiment polarities
        negation_tokens = self.propagateNegation(doc)
        for token in doc:
            if isRoot(token):
                self.traverseTree(token, negation_tokens)

        # Mark words that suggest a spoken/interactive style
        self.spoken_register_markers(doc)

        # Identify where tense shifts present tense to past or vice versa
        self.tenseSequences(doc)

        # Find implied perspectives
        self.perspectiveMarker(doc)

        self.mark_transition_argument_words(doc)

        # Identify the perspectives
        # that apply to individual tokens in the text
        self.set_perspective_spans(doc, self.calculatePerspective)

        if self.calculatePerspective:
            # Identify references especially to animate characters
            characterList, referenceList = self.nominalReferences(doc)
            doc._.nominalReferences = (characterList, referenceList)

            # Identify vocabulary that can be classified as providing concrete
            # detail (say, to a narrative)
            self.concreteDetails(doc)

        # Mark explicitly cued cases of direct speech
        # such as 'This is crazy,' Jenna remarked.
        self.directSpeech(doc)
        
        self.egocentric(doc)
        self.allocentric(doc)

        return doc

    def subjectiveWord(self, token):
        if token.text in ['?', '!'] \
           or token._.vwp_evaluation \
           or (token._.vwp_evaluated_role
               and token.pos_ == 'NOUN') \
           or ((token._.vwp_emotion
               or token._.vwp_character)
               and token._.has_governing_subject
               and token.doc[token._.governing_subject]._.animate) \
           or token._.vwp_hedge \
           or (token._.vwp_emotional_impact
               and (getLogicalObject(token) is None
                    or getLogicalObject(token)._.animate)) \
           or token._.vwp_possibility \
           or (token._.vwp_plan
               and token._.has_governing_subject
               and not in_past_tense_scope(token)) \
           or token._.vwp_future:
            return True
        return False

    def s_o_f(self, tokens):
        return self.statements_of_fact(tokens, ' == 0')

    def statements_of_fact(self, tokens, relation):
        factList = []
        j = 0
        nextRoot = 0
        currentHead = None
        for i in range(0, len(tokens)):
            if i < nextRoot:
                continue

            if tokens[i]._.vwp_quoted:
                continue

            if isRoot(tokens[i]) \
               and not (tokens[i].dep_ == 'conj'
                        and tensed_clause(tokens[i].head)
                        and 'MD' in [child.tag_
                                     for child
                                     in tokens[i].head.children]):
                numSubjective = 0
                currentHead = tokens[i]

                if self.subjectiveWord(tokens[i]) \
                   and len(tokens[i]._.vwp_perspective) == 0:
                    numSubjective += 1

                if len(tokens[i]._.vwp_perspective) > 0 \
                   and (tokens[tokens[i]._.vwp_perspective[0]
                               ].text.lower()
                        in first_person_pronouns
                        or tokens[tokens[i]._.vwp_perspective[0]
                                  ].text.lower()
                        in second_person_pronouns) \
                   and (self.subjectiveWord(tokens[i])):
                    numSubjective += 1

                # Use of potential viewpoint predicates with first or second
                # person viewpoint governors in root position is a mark of
                # subjectivity
                if isRoot(tokens[i]) \
                   and not tokens[i]._.vwp_emotional_impact \
                   and (tokens[i]._.vwp_argument
                        or tokens[i]._.vwp_cognitive
                        or tokens[i]._.vwp_communication
                        or tokens[i]._.vwp_emotion) \
                    and tokens[i]._.has_governing_subject \
                    and (tokens[tokens[i]._.governing_subject
                                ].text.lower() in first_person_pronouns
                         or tokens[tokens[i]._.governing_subject
                                   ].text.lower() in second_person_pronouns):
                    numSubjective += 1

                if isRoot(tokens[i]) \
                   and tokens[i]._.vwp_emotional_impact \
                   and getLogicalObject(tokens[i]) is not None \
                   and (getLogicalObject(tokens[i]).text.lower()
                        in first_person_pronouns
                        or getLogicalObject(tokens[i]).text.lower()
                        in second_person_pronouns):
                    numSubjective += 1

                # Special case: imperatives are implicitly subjective
                # even though there is no explicit evaluation word
                if tokens[i] == self.getHeadDomain(tokens[i]) \
                   and tokens[i].text.lower() == tokens[i].lemma_ \
                   and not in_past_tense_scope(tokens[i]) \
                   and 'expl' not in [left.dep_ for left in tokens[i].lefts] \
                   and getSubject(tokens[i]) is None:
                    numSubjective += 1

                for child in tokens[i].subtree:
                    if child == tokens[i]:
                        continue

                    if child._.vwp_quoted:
                        continue

                    if isRoot(child) \
                       and child.dep_ in ['ROOT', 'conj'] \
                       and child != tokens[i] \
                       and tensed_clause(child) \
                       and getSubject(child) is not None \
                       and not (child.dep_ == 'conj'
                                and 'SCONJ' in [gchild.pos_
                                                for gchild
                                                in tokens[i].children]) \
                       and not (child.dep_ == 'conj'
                                and tensed_clause(child.head)
                                and 'MD' in [gchild.tag_
                                             for gchild
                                             in child.head.children]):
                        currentHead = child
                        break

                    if self.subjectiveWord(child) \
                       and len(child._.vwp_perspective) == 0:
                        numSubjective += 1
                        break

                    if len(child._.vwp_perspective) > 0 \
                       and (tokens[child._.vwp_perspective[0]].text.lower()
                            in first_person_pronouns
                            or tokens[child._.vwp_perspective[0]].text.lower()
                            in second_person_pronouns) \
                       and self.subjectiveWord(child):
                        numSubjective += 1
                        break

                    # Special case: cognitive, communication or
                    # argument predicates as subjects of the
                    # domain head are implicitly subjective
                    if child.dep_ in ['nsubj',
                                      'nsubjpass',
                                      'csubj',
                                      'csubjpass'] \
                       and (len(child._.vwp_perspective) == 0
                            or (tokens[child._.vwp_perspective[0]
                                       ].text.lower()
                                in first_person_pronouns)
                            or (tokens[child._.vwp_perspective[0]
                                       ].text.lower()
                                in second_person_pronouns)) \
                       and not child._.vwp_sourcetext \
                       and (child._.vwp_communication
                            or child._.vwp_cognitive
                            or child._.vwp_argument):
                        numSubjective += 1

                    if child._.vwp_possibility \
                       and child == self.getHeadDomain(child):
                        numSubjective += 1
                        break
                if eval("numSubjective " + relation):
                    start, end = rootTree(currentHead,
                                          currentHead.i,
                                          currentHead.i)
                    entry = \
                        newSpanEntry('objective',
                                     start,
                                     end,
                                     tokens,
                                     tokens[end].idx \
                                     + len(tokens[end].text_with_ws) \
                                     - tokens[start].idx)
                    factList.append(entry)
                    nextRoot = end + 1
                    numSubjective = 0
        return factList

    def statements_of_opinion(self, tokens):
        theList = self.statements_of_fact(tokens, " > 0")
        opinionList = []
        for item in theList:
             newItem = item
             newItem['name'] = 'subjective'
             opinionList.append(newItem)
        return opinionList

    def vwp_emotionword(self, token):
        ''' Return list of emotion words for all tokens in document
        '''
        if token._.vwp_emotion \
           or token._.vwp_emotional_impact:
            return True
        else:
            return False

    def vwp_argumentword(self, token):
        ''' Return list of token indexes for words that meet the standard
            for being explicit argumentation words
        '''
        if token._.vwp_argument \
           or token._.vwp_certainty \
           or token._.vwp_necessity \
           or token._.vwp_probability \
           or token._.vwp_likelihood \
           or token._.vwp_surprise \
           or token._.vwp_qualification \
           or token._.vwp_emphasis \
           or token._.vwp_accuracy \
           or token._.vwp_information \
           or token._.vwp_relevance \
           or token._.vwp_persuasiveness \
           or token._.vwp_reservation \
           or token._.vwp_qualification \
           or token._.vwp_generalization \
           or token._.vwp_illocutionary \
           or token._.vwp_argue:
            return token.i

    def vwp_explicit_argument(self, token):
        ''' Return list of indexes to words that belong to argumentation
            sequences and can be classified as subjective/argument language
        '''
        #return [token.i for token in tokens
        if token._.vwp_argumentation \
           and (token._.vwp_argument
                or token._.vwp_certainty
                or token._.vwp_necessity
                or token._.vwp_probability
                or token._.vwp_likelihood
                or token._.vwp_surprise
                or token._.vwp_risk
                or token._.vwp_cause
                or token._.vwp_tough
                or token._.vwp_qualification
                or token._.vwp_emphasis
                or token._.vwp_accuracy
                or token._.vwp_information
                or token._.vwp_relevance
                or token._.vwp_persuasiveness
                or token._.vwp_reservation
                or token._.vwp_qualification
                or token._.vwp_generalization
                or token._.vwp_illocutionary
                or token._.vwp_manner
                or token._.vwp_importance
                or ((token._.vwp_say
                     or token._.vwp_interpret
                     or token._.vwp_perceive
                     or token._.vwp_think
                     or token._.vwp_argue
                     or (token._.vwp_plan
                         and not token._.vwp_physical
                         and not token._.vwp_social))
                    and (not token.pos_ == 'VERB'
                         or (not token._.has_governing_subject
                             and tensed_clause(token))
                         or (token._.has_governing_subject
                             and (token.doc[
                                  token._.governing_subject
                                  ]._.animate
                                  or token.doc[
                                     token._.governing_subject
                                     ].lemma_ in ['this',
                                                  'that',
                                                  'it']
                                  or token.doc[
                                     token._.governing_subject
                                     ]._.vwp_cognitive
                                  or token.doc[
                                     token._.governing_subject
                                     ]._.vwp_plan
                                  or token.doc[
                                     token._.governing_subject
                                     ]._.vwp_abstract
                                  or token.doc[
                                     token._.governing_subject
                                     ]._.vwp_information
                                  or token.doc[
                                     token._.governing_subject
                                     ]._.vwp_possession
                                  or token.doc[
                                     token._.governing_subject
                                     ]._.vwp_relation
                                  or token.doc[
                                     token._.governing_subject
                                     ]._.vwp_communication))))):
            return True

    extensions = [{"name": "vwp_statements_of_fact",
                   "getter": "s_o_f",
                   "type": "docspan"},
                  {"name": "vwp_statements_of_opinion",
                   "getter": "statements_of_opinion",
                   "type": "docspan"},
                  {"name": "vwp_explicit_argument",
                   "getter": "vwp_explicit_argument",
                   "type": "token"},
                  {"name": "vwp_emotionword",
                   "getter": "vwp_emotionword",
                   "type": "token"},
                  {"name": "vwp_argumentword",
                   "getter": "vwp_argumentword",
                   "type": "token"}
                  ]

    def add_extensions(self):

        """
         Funcion to add extensions with getter functions that allow us
         to access the various lexicons this module is designed to support.
        """

        ##################################################
        # Register extensions for all the categories in  #
        # the stance lexicon                             #
        ##################################################
        for wrd in self.stancePerspectiveVoc:
            for tg in self.stancePerspectiveVoc[wrd]:
                for item in self.stancePerspectiveVoc[wrd][tg]:
                    Token.set_extension('vwp_' + item,
                                        default=False,
                                        force=True)

        #####################################################
        # Register all the extensions in the extension list #
        #####################################################
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

        ########################
        # Viewpoint and stance #
        ########################

        # Index to the word that identifies the perspective that applies
        # to this token
        if not Token.has_extension('vwp_perspective'):
            Token.set_extension('vwp_perspective', default=None)

        # Special index that tracks the perspective of root words for
        # whole sentences. Default is empty list, implicitly the speaker.
        # Only filled when whole-clause viewpoint is signaled explicitly
        if not Token.has_extension('head_perspective'):
            Token.set_extension('head_perspective', default=[])

        if not Token.has_extension('vwp_attribution'):
            Token.set_extension('vwp_attribution', default=False)

        if not Token.has_extension('vwp_source'):
            Token.set_extension('vwp_source', default=False)

        if not Token.has_extension('vwp_cite'):
            Token.set_extension('vwp_cite', default=False)

        if not Token.has_extension('vwp_claim'):
            Token.set_extension('vwp_claim', default=False, force=True)

        if not Token.has_extension('vwp_discussion'):
            Token.set_extension('vwp_discussion',
                                default=False,
                                force=True)

        if not Token.has_extension('vwp_egocentric'):
            Token.set_extension('vwp_egocentric', default=False)

        if not Token.has_extension('vwp_allocentric'):
            Token.set_extension('vwp_allocentric', default=False)

        if not Token.has_extension('antecedents'):
            Token.set_extension('antecedents', default=None, force=True)

        # Mapping of tokens to viewpoints for the whole document
        #
        # Our code creates separate lists by the viewpoint that
        # applies to each token: implicit first person,
        # explicit first person, explicit third person, and
        # for explicit third person, by the offset for the referent
        # that takes that perspective.
        Span.set_extension('vwp_perspective_spans',
                           default=None,
                           force=True)
        Doc.set_extension('vwp_perspective_spans',
                          default=None,
                          force=True)

        # Identification of stance markers that make a text more
        # subjective/opinion-based and less objective
        #
        # Our code creates separate lists by the viewpoint that
        # applies to the stance marker: implicit first person,
        # explicit first person, explicit third person, and
        # for explicit third person, by the offset for the referent
        # that takes that perspective.
        Doc.set_extension('vwp_stance_markers',
                          default=None,
                          force=True)
        Span.set_extension('vwp_stance_markers',
                           default=None,
                           force=True)

        # Identification of propositional attitude predicates associated
        # with specific predicates. E.g., believe or think in 'I believe
        # that this is true', or 'John thinks we are on the right track'.
        #
        # Our code creates separate lists by the viewpoint that
        # applies to the prop. attitude predicate: implicit first person,
        # explicit first person, explicit third person, and
        # for explicit third person, by the offset for the
        # animate entity referent.
        if not Doc.has_extension('vwp_propositional_attitudes'):
            Doc.set_extension('vwp_propositional_attitudes', default=None)

        # Identification of emotion markers that attribute
        # emotional states to agents
        #
        # Our code creates separate lists by the animate entity that
        # the emotion predicate applies to: implicit first person,
        # explicit first person, explicit third person, and
        # for explicit third person, by the offset for the
        # animate entity referent.
        if not Doc.has_extension('vwp_emotion_states'):
            Doc.set_extension('vwp_emotion_states', default=None)

        # Identification of character markers that attribute
        # character attributes to agents
        #
        # Our code creates separate lists by the animate entity that
        # the character attribute applies to: implicit first person,
        # explicit first person, explicit third person, and
        # for explicit third person, by the offset for the referent
        # offset for the animate entity.
        if not Doc.has_extension('vwp_character_traits'):
            Doc.set_extension('vwp_character_traits',
                              default=None)

        if not Doc.has_extension('vwp_social_awareness'):
            Doc.set_extension('vwp_social_awareness',
                              default=None)

        if not Doc.has_extension('concrete_details'):
            Doc.set_extension('concrete_details',
                              default=None)

        # SentiWord sentiment polarity ratings #

        # Rating of positive or negative sentiment
        if not Token.has_extension('vwp_sentiment'):
            Token.set_extension('vwp_sentiment',
                                default=None)

        # Rating of positive or negative sentiment
        if not Token.has_extension('vwp_tone'):
            Token.set_extension('vwp_tone',
                                default=None)

        ##########################
        # Argumentative style    #
        ##########################

        if not Token.has_extension('vwp_argumentation'):
            Token.set_extension('vwp_argumentation', default=False)

        #####################
        # Interactive Style #
        #####################

        # Flag identifying words that cue an interactive style
        if not Token.has_extension('vwp_interactive'):
            Token.set_extension('vwp_interactive', default=False)

        if not Doc.has_extension('nominalReferences'):
            Doc.set_extension('nominalReferences', default=None)

        ################################
        # Direct speech                #
        ################################

        # Flag that says whether a verb of saying (thinking, etc.)
        # is being used as direct speech ('John is happy, I think')
        # rather than as direct speech ('I think that John is happy.')
        if not Token.has_extension('vwp_direct_speech'):
            Token.set_extension('vwp_direct_speech',
                                default=False)

        if not Token.has_extension('vwp_in_direct_speech'):
            Token.set_extension('vwp_in_direct_speech',
                                default=False)

        # List of spans that count as direct speech
        if not Doc.has_extension('vwp_direct_speech_spans'):
            Doc.set_extension('vwp_direct_speech_spans', default=[])
        # version in the standardized format we use for doc level
        # spans
        if not Doc.has_extension('direct_speech_spans'):
            Doc.set_extension('direct_speech_spans', default=[])

        # Flag identifying a nominal that identifies the
        # speaker referenced as 'I/me/my/mine' within a
        # particular stretch of direct speech
        if not Token.has_extension('vwp_speaker'):
            Token.set_extension('vwp_speaker', default=None)

        # List of all tokens (nominals or first person pronouns)
        # that refer to a speaker defined within a particular
        # stretch of direct speech
        if not Token.has_extension('vwp_speaker_refs'):
            Token.set_extension('vwp_speaker_refs', default=None)

        # Flag identifying a nominal (if present) that identifies
        # the speaker referenced as 'you/your/yours' within a
        # particular stretch of direct speech.
        if not Token.has_extension('vwp_addressee'):
            Token.set_extension('vwp_addressee', default=None)

        # List of all tokens (nominals or first person pronouns)
        # that refer to an addressee defined within a particular
        # stretch of direct speech
        if not Token.has_extension('vwp_addressee_refs'):
            Token.set_extension('vwp_addressee_refs', default=None)

        ##########################################################
        # Helper extensions for tracking viewpoint domains over  #
        # sentential scope                                       #
        ##########################################################

        # Flag that identifies whether a token has a governing subject.
        if not Token.has_extension('has_governing_subject'):
            Token.set_extension('has_governing_subject', default=False)

        # Pointer to a token's governing subject.
        if not Token.has_extension('governing_subject'):
            Token.set_extension('governing_subject', default=None)

        # A token's WordNet usage domain.
        if not Token.has_extension('usage'):
            Token.set_extension('usage', default=None)

        # Pointer to a token's governing subject.
        if not Doc.has_extension('vwp_tense_changes'):
            Doc.set_extension('vwp_tense_changes', default=None)

    def __init__(self, fast: bool, lang="en"):
        super().__init__()
        self.package_check(lang)
        self.load_lexicon(lang)
        self.calculatePerspective = not fast
        self.add_extensions()

    def synsets(self, token):
        return wordnet.synsets(token.orth_)

    def setLexiconAttributes(self, token, doc):

        # unigram lexicon match
        if token.lemma_ in self.stancePerspectiveVoc \
           and token.pos_ in self.stancePerspectiveVoc[token.lemma_]:
            for item in (self.stancePerspectiveVoc[
                         token.lemma_][token.pos_]):
                if not Token.has_extension('vwp_' + item):
                    Token.set_extension('vwp_' + item, default=False)
                token._.set('vwp_' + item, True)

                # Rules and special cases. TBD: introduce mechanism
                # for specifying these kinds of exceptions
                if token._.vwp_evaluated_role:
                    token._.vwp_evaluation = True

                if token.dep_ == 'amod' \
                   and token.text.lower() == 'certain':
                    token._.vwp_evaluation = False
                    token._.vwp_likelihood = False
                    token._.vwp_probability = False

                # modal have to
                if token.text.lower() in ['have',
                                          'has',
                                          'dare',
                                          'dares',
                                          'ought',
                                          'going',
                                          'need',
                                          'needs'] \
                   and token.i+1 < len(doc) \
                   and token.nbor().tag_ == 'TO':
                    token._.vwp_evaluation = True

                # plan words with modal complements
                if token.tag_ == 'TO' \
                   and token.head.pos_ in ['NOUN', 'VERB', 'ADJ'] \
                   and token.head.head._.vwp_plan:
                    token._.vwp_evaluation = True

                # plan words asserted as a predicate
                if token._.vwp_plan \
                   and token.dep_ in ['attr', 'oprd']:
                    token.head._.vwp_evaluation = True

        # for now only requiring string match for bigram or trigram
        # entries. TO-DO: use dep_ of head of matched phrase to pick
        # the POS
        if token.i < len(doc) - 1:
            bigram = token.text.lower() + '_' + doc[token.i+1].text.lower()
            if bigram in self.stancePerspectiveVoc:
                for pos in self.stancePerspectiveVoc[bigram]:
                    for item in self.stancePerspectiveVoc[bigram][pos]:
                        if not Token.has_extension('vwp_' + item):
                            Token.set_extension('vwp_'
                                                + item,
                                                default=False)
                        token._.set('vwp_' + item, True)
                        doc[token.i+1]._.set('vwp_' + item, True)
        if token.i < len(doc) - 2:
            trigram = (token.text.lower()
                       + '_'
                       + doc[token.i
                       + 1].text.lower()
                       + '_'
                       + doc[token.i + 2].text.lower())
            if trigram in self.stancePerspectiveVoc:
                for pos in self.stancePerspectiveVoc[trigram]:
                    for item in self.stancePerspectiveVoc[trigram][pos]:
                        if not Token.has_extension('vwp_' + item):
                            Token.set_extension('vwp_'
                                                + item,
                                                default=False)
                        token._.set('vwp_' + item, True)
                        doc[token.i + 1]._.set('vwp_' + item, True)
                        doc[token.i + 2]._.set('vwp_' + item, True)

    def viewpointPredicate(self, token):
        """
         Viewpoint predicates cover a range of concepts, referenced
         using our spacy extensions. vwp_communication covers verbs
         of saying and similar elements in other parts of speech
         vwp_cognitive covers various sorts of mental state predicates.
         vwp_emotion covers emotion predicates where the experiencer
         is syntactically a subject. vwp_emotional_impact covers
         emotion predicates where the experiencer is syntactically
         an object. vwp_plan covers abstract predicates describing
         mental planning, strategy, and execution. vwp_argument covers
         predicates focusing on describing and evaluating argumentation.
        """

        if token._.vwp_communication \
           or token._.vwp_cognitive \
           or token._.vwp_emotion \
           or token._.vwp_emotional_impact \
           or token._.vwp_argument:
            return True
        return False

    def potentialViewpoint(self, token):
        """
         Potential viewpoint expressions are nominals that are either
         (a) animate, (b) in the 'vwp_sourcetext' or 'vwp_communication'
         categories that we loaded from our lexicon.
        """
        if token is not None and (token._.animate or token._.vwp_sourcetext) \
           and not token._.location:
            return True
        return False

    def find_abstract_trait_object(self, tok: Token):
        subj = None
        if 'of' in [child.lemma_ for child in tok.subtree] \
           and (self.viewpointPredicate(tok) or tok._.abstract_trait):
            for child in tok.subtree:
                if child.lemma_ == 'of':
                    subj = getPrepObject(child, ['of'])
                    break
        return subj

    def findViewpoint(self, predicate, tok: Token, lastDep, lastTok, hdoc):
        """
         Locate the explicit or implicit subjects of viewpoint-controlling
         predicates and mark the chain of nodes from the predicate to the
         controlling animate nominal as being within that nominal's viewpoint
         scope.
        """

        subj = None

        # Dative constructions like 'obvious to me' or 'give me
        # the belief' express viewpoint. The dative NP ('me') is
        # the viewpoint of the predicate ('obvious', 'belief')
        if (tok._.vwp_evaluation
            and 'IN' in [child.tag
                         for child in tok.children]) \
           or tok.head._.vwp_ditransitive:
            subj = getDative(tok)

        # Agent phrases of passive sentences can carry viewpoint
        elif 'agent' in [child.dep_ for child in tok.children]:
            for child in tok.children:
                if child.dep_ == 'agent':
                    subj = getPrepObject(child, ['by'])
                    break

        # Control structures can carry viewpoint
        elif (tok.dep_ in ['acl', 'acomp']
              and tok.head._.governing_subject is not None):
            subj = hdoc[tok.head._.governing_subject]
            return subj

        # But more generally it's the subject (or possessive of a noun)
        # that takes viewpoint for viewpint-taking predicates
        if tok._.vwp_emotional_impact:
            subj = getLogicalObject(tok)
            if subj is not None:
                return subj
            else:
                subj = getSubject(tok)

        if subj is None:
            subj = self.find_abstract_trait_object(tok)

        if subj is not None:

            # Recursively find subject of NPs headed by potential
            # viewpoint predicate nouns. E.g., 'Jenna's article
            # stated that this was a  problem.' -> Jenna is viewpoint
            # for 'stated'
            if (self.viewpointPredicate(subj)
                or tok._.abstract_trait) \
               and not subj._.animate \
               and subj != tok \
               and subj != lastTok:
                subj2 = self.findViewpoint(predicate,
                                           subj,
                                           subj.dep_,
                                           tok,
                                           hdoc)
                if subj2 is not None:
                    return subj2
                else:
                    return subj
            else:
                return subj

        # In general, subjectless constructions allow viewpoint to spread
        # to subjacent elements from animate nouns in viewpoint-taking roles
        # like subject (Jenny is starting to believe -> 'Jenny' is viewpoint,
        # same as in 'Jenny believes'.)
        elif (tok.dep_ in ['acomp',
                           'ccomp',
                           'xcomp',
                           'advcl',
                           'advmod',
                           'appos',
                           'attr',
                           'dep',
                           'conj',
                           'infmod',
                           'npadvmod',
                           'oprd',
                           'partmod',
                           'prep',
                           'parataxis',
                           'acl']
              and lastDep != 'ccomp'
              and lastDep != 'csubj'
              and lastDep != 'csubjpass'):
            # Note on restrictions above:
            # Complement clauses (comp, csubj, csubjpass) are opaque
            # to viewpoint from higher in the parse tree. (In: John
            # believes that Sue disagrees with him, 'disagree' must
            # be from Sue's POV, not John's)

            if tok._.vwp_emotional_impact \
               and tok.head._.vwp_raising:
                # Tough-movement verbs but not raising verbs are
                # transparent to viewpoint for emotional impact
                # verbs like 'alarm': 'Max is easy to please'
                # is from Max's viewpoint (Max is pleased) but
                # 'Max is certain to please' is default viewpoint
                # (the speaker's), with an unspecified object
                # viewpoint of who is pleased.
                return None
            if tok.head != tok.head.head:
                subj = self.findViewpoint(predicate,
                                          tok.head,
                                          tok.dep_,
                                          tok,
                                          hdoc)
            return subj

        # Allow light verbs to include their objects in the scope
        # of subject viewpoint (Maria has the right to argue that
        # I am wrong -> 'that I am wrong' is from Maria's POV)
        elif (tok.dep_ in ['dobj', 'nsubjpass']
              and (tok.head._.vwp_cause
                   or tok.head._.vwp_possession
                   or tok.lemma_ in getLightVerbs())):
            subj = getSubject(tok)
            if subj is not None:
                if self.viewpointPredicate(subj) \
                   and not self.potentialViewpoint(subj):
                    subj = self.findViewpoint(predicate,
                                              subj,
                                              tok.dep_,
                                              tok,
                                              hdoc)
                if subj is not None:
                    return subj
                else:
                    return subj
            else:
                subj = self.findViewpoint(predicate,
                                          tok.head,
                                          tok.dep_,
                                          tok,
                                          hdoc)
                return subj
        else:
            return subj

    def registerViewpoint(self, hdoc, token, subj):
        """
         Utility function that resolves antecedents of pronouns
         and then sets the vwp_perspective extension attribute
         for a token to point to the right referent.
        """

        # really just going for consistency here. If it's
        # explicit 1st person perspective, we want to mark
        # everything in the domain as explicit rather than
        # implicit first person perspective
        if token._.governing_subject is not None \
           and hdoc[token._.governing_subject
                    ].text.lower() in first_person_pronouns:
            token._.head_perspective = \
                [token._.governing_subject]

        subj._.vwp_perspective = [subj.i]
        token._.vwp_perspective = [subj.i]

    def markAttribution(self, tok, hdoc):
        if (tok.dep_ == 'nsubj'
            and tok.head.dep_ == 'conj'
            and tok.head.head._.vwp_attribution
            and (tok.head._.vwp_argument
                 or tok.head._.vwp_say
                 or tok.head._.vwp_argue
                 or tok.head._.vwp_think)) \
            or (tok.head._.governing_subject is not None
                and tok.dep_ in ['ccomp', 'csubjpass', 'acl']
                and tensed_clause(tok)
                and ((self.getHeadDomain(tok.head).dep_ is None
                     or isRoot(self.getHeadDomain(tok.head)))
                     or isRoot(self.getHeadDomain(tok.head).head)
                     or tok.head.dep_ == 'conj')
                and (tok.head._.vwp_argument
                     or tok.head._.vwp_say
                     or tok.head._.vwp_argue
                     or tok.head._.vwp_think)) \
            and not tok.left_edge.tag_.startswith('W') \
            and (hdoc[tok.head._.governing_subject].text.lower()
                 not in personal_or_indefinite_pronoun) \
            and (hdoc[tok.head._.governing_subject]._.animate
                 or (hdoc[tok.head._.governing_subject].text.lower()
                     in ['they',
                         'them',
                         'some',
                         'others',
                         'many',
                         'few'])
                 or hdoc[tok.head._.governing_subject].tag_ == 'PROPN'
                 or hdoc[tok.head._.governing_subject].tag_ == 'DET'
                 or hdoc[tok.head._.governing_subject]._.vwp_sourcetext):
            hdoc[tok.head._.governing_subject]._.vwp_source = True
            for child in hdoc[tok.head._.governing_subject].subtree:
                if child.i >= tok.head.i:
                    break
                child._.vwp_source = True
            tok.head._.vwp_attribution = True
            for child in tok.head.subtree:
                if child.i >= tok.head.i:
                    break
                child._.vwp_attribution = True

    def markCitedText(self, tok, hdoc):
        if tok.tag_ == '-LRB-' \
           and (tok.head.tag_ == 'NNP'
                or tok.head.ent_type_ in ['PERSON',
                                          'ORG',
                                          'WORK_OF_ART',
                                          'DATE']
                or tok.head._.vwp_quoted):
            i = tok.i
            while (i < len(hdoc)
                   and hdoc[i].tag_ != '-RRB-'):
                hdoc[i]._.vwp_cite = True
                for child in hdoc[i].subtree:
                    child._.vwp_cite = True
                    tok._.vwp_cite = True
                i += 1
        elif tok.tag_ == '-LRB-':
            for child in tok.head.children:
                if child.tag_ == '-RRB-':
                    for i in range(tok.i + 1, child.i):
                        if hdoc[i].tag_ == 'NNP' \
                           or hdoc[i].ent_type_ in ['PERSON',
                                                    'ORG',
                                                    'WORK_OF_ART',
                                                    'DATE'] \
                           or hdoc[i]._.vwp_quoted:
                            hdoc[i]._.vwp_cite = True
                            tok._.vwp_cite = True
                            for grandchild in hdoc[i].subtree:
                                grandchild._.vwp_cite = True
                            break

    def markAddresseeRefs(self, target, tok, addressee_refs):
        # Mark the addressee for the local domain,
        # which will be the object of the preposition
        # 'to' or the direct object
        if isRoot(target):
            target._.vwp_addressee = []
        for child2 in target.children:
            if child2.dep_ == 'dative' and child2._.animate:
                target._.vwp_addressee = [child2.i]
                tok._.vwp_addressee = [child2.i]
                if child2.i not in addressee_refs:
                    addressee_refs.append(child2.i)
                break
            if child2.dep_ == 'dobj' and child2._.animate:
                target._.vwp_addressee = [child2.i]
                tok._.vwp_addressee = [child2.i]
                if child2.i not in addressee_refs:
                    addressee_refs.append(child2.i)
                break
            if target._.vwp_addressee is not None:
                dativeNoun = getPrepObject(target, ['to'])
                if dativeNoun is not None \
                   and dativeNoun._.animate:
                    target._.vwp_addressee = [dativeNoun.i]
                    dativeNoun._.vwp_addressee = [dativeNoun.i]
                    if child2.i not in addressee_refs:
                        addressee_refs.append(child2.i)
                        break
                dativeNoun = getPrepObject(target, ['at'])
                if dativeNoun is not None \
                   and dativeNoun._.animate:
                    target._.vwp_addressee = [dativeNoun.i]
                    dativeNoun._.vwp_addressee = [dativeNoun.i]
                    if child2.i not in addressee_refs:
                        addressee_refs.append(child2.i)
                    break
        return addressee_refs

    def directSpeech(self, hdoc):
        """
         Scan through the document and find verbs that control
         complement clauses AND contain an immediately dpeendent
         punctuation mark (which is the cue that this is some form
         of direct rather than indirect speech. E.g., 'John said,
         I am happy.'
        """
        lastRoot = None
        currentRoot = None
        speaker_refs = []
        addressee_refs = []
        for tok in hdoc:

            self.markCitedText(tok, hdoc)
            self.markAttribution(tok, hdoc)

            if '\n' in tok.text:
                speaker_refs = []
                addressee_refs = []
                lastRoot = currentRoot
                currentRoot = tok
            else:
                if currentRoot != getRoot(tok):
                    lastRoot = currentRoot
                currentRoot = getRoot(tok)

            # Special case: quote introduced by tag word
            # at end of previous sentence
            target = tok.head
            if tok == currentRoot:
                left = currentRoot.left_edge
                start = currentRoot.left_edge
                while (left.i > 0
                       and (left.tag_ == '_SP'
                            or left.dep_ == 'punct')):
                    left = left.nbor(-1)
                while (start.i + 1 < len(hdoc)
                       and start.tag_ == '_SP'):
                    start = start.nbor()
                target = left

            # If we match the special case with the taq word
            # being a verb or speaking, or else match the
            # general case where the complement of a verb of
            # speaking is a quote, then we mark direct speech
            if (tok == currentRoot
                and quotationMark(start)
                and not left._.vwp_plan
                and not left.head._.vwp_plan
                and (left._.vwp_communication
                     or left._.vwp_cognitive
                     or left._.vwp_argument
                     or left.head._.vwp_communication
                     or left.head._.vwp_cognitive
                     or left.head._.vwp_argument)) \
                or (tok.dep_ in ['ccomp',
                                 'csubjpass',
                                 'acl',
                                 'xcomp',
                                 'intj',
                                 'nsubj']
                    and tok.head.pos_ in ['VERB',
                                          'NOUN',
                                          'ADJ',
                                          'ADV']
                    and not tok.head._.vwp_quoted
                    and not tok.head._.vwp_plan
                    and (tok.head._.vwp_communication
                         or tok.head._.vwp_cognitive
                         or tok.head._.vwp_argument)):

                context = [child for child in target.children]
                if len(context) > 0 and context[0].i > 0:
                    leftnbor = context[0].nbor(-1)
                    if quotationMark(leftnbor):
                        context.append(leftnbor)
                if len(context) > 0 and context[0].i + 1 < len(hdoc):
                    rightnbor = context[0].nbor()
                    if quotationMark(rightnbor):
                        context.append(rightnbor)

                for child in context:

                    # Detect direct speech

                    # Reference to direct speech in previous sentence
                    if (tok == currentRoot and target != tok.head):
                        if target._.has_governing_subject:
                            thisDom = target
                        else:
                            dom = target.head
                            while not dom._.has_governing_subject:
                                dom = dom.head
                            thisDom = dom
                        speaker_refs = []
                        addressee_refs = []
                        if thisDom._.has_governing_subject:
                            for item in ResolveReference(hdoc[thisDom._.governing_subject], hdoc):
                                speaker_refs.append(item)
                            if thisDom in hdoc[thisDom._.governing_subject
                                               ].subtree:
                                thisDom = hdoc[thisDom._.governing_subject]
                            thisDom._.vwp_speaker_refs = speaker_refs

                            # Mark the addressee for the local domain,
                            # which will be the object of the preposition
                            # 'to' or the direct object
                            addressee_refs = \
                                self.markAddresseeRefs(thisDom,
                                                       tok,
                                                       addressee_refs)
                            thisDom._.vwp_addressee_refs = addressee_refs

                        thisDom._.vwp_direct_speech = True
                        break

                    # we need punctuation BETWEEN the head and
                    # the complement for this to be indirect speech
                    elif (child.dep_ == 'punct'
                          and ((child.i < tok.i
                                and tok.i < target.i
                                and tok.dep_ == 'nsubj'
                                and tok.pos_ == 'VERB')
                               or (target.i < child.i
                                   and child.i < tok.i)
                               or (target.i > child.i
                                   and child.i > tok.i))):

                        if isRoot(target):
                            speaker_refs = []
                            addressee_refs = []

                        # Mark the speaker for the local domain, which will
                        # be the subject or agent, if one is available, and
                        # interpret first person pronouns within the local
                        # domain as referring to the subject of the domain
                        # predicate
                        subj = getSubject(target)

                        # TO-DO: add support for controlled subjects
                        # ("This is wrong", Jenna wanted to say.")

                        # Demoted subject of passive sentence
                        if subj is None:
                            subj = getPrepObject(tok, ['by'])

                        # Inverted subject construction
                        # ('spare me, begged the poor mouse')
                        if subj == tok and getObject(target) is not None:
                            subj = getObject(target)

                        # Mark this predicate as involving direct speech
                        # (complement clause with dependent punctuation is
                        # our best indicator of direct speech. That means
                        # we'll run into trouble when students do not punctuate
                        # correctly, but there's nothing to be done about it
                        # short of training a classifier on erroneous student
                        # writing.)

                        if subj is not None \
                           and (subj._.animate
                                or subj._.vwp_sourcetext
                                or subj.text.lower() in ['they', 'it']
                                or subj.pos_ == 'PROPN'):
                            target._.vwp_direct_speech = True
                        else:
                            continue

                        # If the quote doesn't encompass the complement,
                        # this isn't direct speech
                        if quotationMark(child) and not tok._.vwp_quoted:
                            continue

                        # Record the speaker as being the subject
                        if subj is not None:
                            target._.vwp_speaker = [subj.i]
                            tok._.vwp_speaker = [subj.i]
                            if subj.i not in speaker_refs:
                                speaker_refs.append(subj.i)
                            subjAnt = [hdoc[loc] for loc
                                       in ResolveReference(subj, hdoc)]
                            if subjAnt is not None:
                                for ref in subjAnt:
                                    if ref.i not in target._.vwp_speaker:
                                        target._.vwp_speaker.append(ref.i)
                                    if ref.i not in tok._.vwp_speaker:
                                        tok._.vwp_speaker.append(ref.i)
                                    if ref.i not in speaker_refs:
                                        speaker_refs.append(ref.i)

                        elif isRoot(tok.head):
                            target._.vwp_speaker = []
                            tok._.vwp_speaker = []

                        # Mark the addressee for the local domain,
                        # which will be the object of the preposition
                        # 'to' or the direct object
                        addressee_refs = \
                            self.markAddresseeRefs(target, tok, addressee_refs)

                        # If it does not conflict with the syntactic
                        # assignment, direct speech status is inherited
                        # from the node we detect as involving direct
                        # speech.
                        for descendant in target.subtree:
                            if descendant._.vwp_quoted:
                                # TO-DO: add block to prevent inheritance
                                # for embedded direct speech
                                if descendant.text.lower() in \
                                   first_person_pronouns \
                                   and len(list(descendant.children)) == 0:
                                    descendant._.vwp_speaker = \
                                        target._.vwp_speaker
                                    if descendant.i not in speaker_refs \
                                       and descendant.i not in addressee_refs:
                                        speaker_refs.append(descendant.i)
                        target._.vwp_speaker_refs = speaker_refs

                        # direct speech should be treated as quoted even if
                        # it isn't in quotation marks, as in 'This is good,
                        # I thought'
                        for descendant in tok.subtree:
                            descendant._.vwp_quoted = True

                        for descendant in tok.subtree:
                            if descendant._.vwp_quoted:
                                if (descendant.text.lower()
                                    in second_person_pronouns
                                    and len(list(descendant.children)) == 0
                                    and descendant.i not in speaker_refs) \
                                   or (descendant.dep_ == 'vocative'
                                       and descendant.pos_ == 'NOUN'):
                                    descendant._.vwp_addressee = \
                                        tok.head._.vwp_addressee
                                    if descendant.i not in addressee_refs:
                                        addressee_refs.append(descendant.i)
                        target._.vwp_addressee_refs = addressee_refs
                        break

            if (currentRoot is not None
                and lastRoot is not None
                and lastRoot._.vwp_quoted
                and (currentRoot.pos_ == 'VERB'
                     and getSubject(currentRoot) is None
                     and (currentRoot._.vwp_communication
                          or currentRoot._.vwp_cognitive
                          or currentRoot._.vwp_argument))):
                speaker_refs = []
                addressee_refs = []
                currentRoot._.vwp_direct_speech = True
                for child in tok.children:
                    if child.dep_ == 'dobj':
                        subj = child
                        currentRoot._.vwp_direct_speech = True
                        currentRoot._.vwp_addressee = []
                        currentRoot._.vwp_speaker = [subj.i]
                        child._.vwp_speaker = [subj.i]
                        if subj.i not in speaker_refs:
                            speaker_refs.append(subj.i)
                        subjAnt = [loc for loc
                                   in ResolveReference(subj, hdoc)]
                        if subjAnt is not None:
                            for ref in subjAnt:
                                if ref not in currentRoot._.vwp_speaker:
                                    currentRoot._.vwp_speaker.append(ref)
                                if ref not in child._.vwp_speaker:
                                    child._.vwp_speaker.append(ref)
                                if ref not in speaker_refs:
                                    speaker_refs.append(ref)
                        for descendant in lastRoot.subtree:
                            if descendant.text.lower() in \
                               first_person_pronouns:
                                descendant._.vwp_speaker = \
                                    lastRoot._.vwp_speaker
                                if descendant.i not in speaker_refs:
                                    speaker_refs.append(descendant.i)
                            if descendant.text.lower() in \
                               second_person_pronouns \
                               or (descendant.dep_ == 'vocative'
                                   and descendant.pos_ == 'NOUN'):
                                if descendant._.vwp_addressee is None:
                                    descendant._.vwp_addressee = []
                                descendant._.vwp_addressee.append(
                                    descendant.i)
                                if descendant.i not in speaker_refs:
                                    addressee_refs.append(descendant.i)
                        for addressee in addressee_refs:
                            if lastRoot._.vwp_addressee_refs is not None \
                                and addressee not in \
                                    lastRoot._.vwp_addressee_refs:
                                if addressee not in \
                                   lastRoot._.vwp_addressee_refs:
                                    lastRoot._.vwp_addressee_refs.append(
                                       addressee)
                            else:
                                if addressee not in \
                                   lastRoot._.vwp_addressee_refs:
                                    lastRoot._.vwp_addressee_refs = \
                                        [addressee]
                        for speaker in speaker_refs:
                            if lastRoot._.vwp_speaker_refs is not None \
                                and speaker not in \
                                    lastRoot._.vwp_speaker_refs:
                                if speaker not in \
                                   lastRoot._.vwp_speaker_refs:
                                    lastRoot._.vwp_speaker_refs.append(
                                        speaker)
                            else:
                                if speaker not in \
                                   lastRoot._.vwp_speaker_refs:
                                    lastRoot._.vwp_speaker_refs = [speaker]
                        currentRoot._.vwp_speaker = speaker_refs
                        currentRoot._.vwp_addressee = addressee_refs
                        currentRoot._.vwp_speaker_refs = speaker_refs
                        currentRoot._.vwp_addressee_refs = addressee_refs
                        break

            # A quotation following direct speech without identifier
            # can be assumed to be a continuation of the previous
            # direct speech. OR following an immediate introduction
            # by a communication/cognition/argument word

            if currentRoot is not None \
               and lastRoot is not None \
               and currentRoot._.vwp_quoted \
               and lastRoot._.vwp_direct_speech:

                currentRoot._.vwp_direct_speech = True
                if lastRoot._.vwp_speaker is not None \
                   and len(lastRoot._.vwp_speaker) > 0:
                    currentRoot._.vwp_speaker = lastRoot._.vwp_speaker
                if lastRoot._.vwp_addressee is not None \
                   and len(lastRoot._.vwp_addressee) > 0:
                    currentRoot._.vwp_addressee = lastRoot._.vwp_addressee
                if lastRoot._.vwp_speaker_refs is not None \
                   and len(lastRoot._.vwp_speaker_refs) > 0:
                    for item in lastRoot._.vwp_speaker_refs:
                        if item not in speaker_refs:
                            speaker_refs.append(item)
                if lastRoot._.vwp_addressee_refs is not None \
                   and len(lastRoot._.vwp_addressee_refs) > 0:
                    for item in lastRoot._.vwp_addressee_refs:
                        if item not in addressee_refs:
                            addressee_refs.append(item)
                for descendant in tok.subtree:
                    # Direct speech status is inherited
                    # TO-DO: add block to prevent inheritance
                    # for embedded direct speech
                    if descendant.text.lower() in first_person_pronouns:
                        descendant._.vwp_speaker = speaker_refs
                        if descendant.i not in speaker_refs:
                            speaker_refs.append(descendant.i)
                    if descendant.text.lower() in second_person_pronouns \
                       or (descendant.dep_ == 'vocative'
                           and descendent.pos_ == 'NOUN'):
                        descendant._.vwp_addressee = lastRoot._.vwp_addressee
                        if descendant.i not in addressee_refs:
                            addressee_refs.append(descendant.i)
                currentRoot._.vwp_speaker_refs = speaker_refs
                currentRoot._.vwp_addressee_refs = addressee_refs
                tok.head._.vwp_addressee_refs = addressee_refs
                tok.head._.vwp_speaker_refs = speaker_refs

            # Quoted text that contains first or second person
            # pronouns can be presumed to be direct speech
            if (isRoot(tok) and tok._.vwp_quoted) \
               or (tok._.vwp_quoted and '\n' in tok.head.text):
                if tok._.vwp_speaker is None:
                    tok._.vwp_speaker = []
                if tok._.vwp_addressee is None:
                    tok._.vwp_addressee = []

                if len(speaker_refs) > 0:
                    tok._.vwp_direct_speech = True

                subtree = tok.subtree
                if isRoot(tok):
                    subtree = tok.head.subtree

                for descendant in subtree:
                    if descendant.text.lower() in first_person_pronouns:
                        if descendant.i not in speaker_refs:
                            speaker_refs.append(descendant.i)
                        if descendant.i not in tok._.vwp_speaker:
                            tok._.vwp_speaker.append(descendant.i)
                        tok._.vwp_direct_speech = True
                        tok._.vwp_speaker = [descendant.i]
                        tok._.vwp_speaker_refs = speaker_refs
                    if descendant.text.lower() in second_person_pronouns \
                       or (descendant.dep_ == 'vocative'
                           and descendant.pos_ == 'NOUN'):
                        if descendant.i not in addressee_refs:
                            addressee_refs.append(descendant.i)
                        if descendant.i not in tok._.vwp_addressee:
                            if descendant.i not in tok._.vwp_addressee:
                                tok._.vwp_addressee.append(descendant.i)
                        tok._.vwp_direct_speech = True
                        tok._.vwp_addressee = [descendant.i]
                        tok._.vwp_addressee_refs = addressee_refs

                currentRoot._.vwp_speaker_refs = speaker_refs
                currentRoot._.vwp_addressee_refs = addressee_refs

                # TO-DO: Text with no specified viewpoint following
                # direct speech (including internal mental state predicates
                # like feel or believe used as direct speech tags) may be a
                # continuation of the direct speech segment. But we can't
                # resolve such cases without using a deeper NLP module that
                # is aware of other indicators of text macrostructure to
                # resolve implicit stance to the speaker or to some implied
                # speaker in a smaller text scope.

        self.set_direct_speech_spans(hdoc)

    def perspectiveMarker(self, hdoc):
        """
         Find the viewpoint nominal that governs each viewpoint predicate
          in the document. For instance, in 'John is unlikely to win', there
          is no explicit viewpoint nominal, so the viewpoint is implicitly
          the speaker. But in 'According to Bill, John is unlikely to win',
          'Bill' governs 'unlikely', so the judgment of unlikelihood is
          attributed to Bill and not to the speaker.
        """

        subj = None

        for token in hdoc:

            # Evaluated role words when root of a domain take the
            # perspective of that domain (e.g., John deserves a raise)
            if token._.vwp_evaluated_role \
               and isRoot(token):
                token._.vwp_perspective = token._.head_perspective
                continue

            # Set viewpoint for emotional impact predicates
            # with an animate logical object
            if token._.vwp_emotional_impact \
               and getLogicalObject(token) is not None:
                obj = getLogicalObject(token)
                if obj._.animate:
                    self.registerViewpoint(hdoc, token, obj)
                    obj._.vwp_perspective = token._.vwp_perspective
                if obj is not None \
                   and 'poss' in [child.dep_
                                  for child
                                  in obj.children
                                  if child._.animate]:
                    for child in obj.children:
                        if child.dep_ == 'poss' \
                           and child._.animate:
                            self.registerViewpoint(hdoc, token, child)
                            child._.vwp_perspective = token._.vwp_perspective
                            token._.governing_subject = child.i
                            continue
                continue

            # Special case (evaluation predicates may take dative arguments)
            # Treat the object of the preposition as the viewpoint controller
            if token._.vwp_evaluation \
               or token._.vwp_emotion \
               or token._.vwp_emotional_impact:
                subj = getPrepObject(token, ['to', 'for'])
                if subj is not None \
                   and (subj._.animate or subj._.vwp_sourcetext):
                    self.registerViewpoint(hdoc, token, subj)

                    # If this is adjectival, we need to mark the matrix
                    # verb/subject as being in the head's perspective
                    # because the to-phrase has purely local scope
                    if token.dep_ in ['acomp', 'oprd', 'attr']:
                        token.head._.vwp_perspective = \
                            self.getHeadDomain(token.head
                                               )._.head_perspective
                        if getSubject(token) is not None:
                            getSubject(token)._.vwp_perspective = \
                                self.getHeadDomain(token.head
                                                   )._.head_perspective

                    continue
                elif (self.getHeadDomain(token)._.vwp_perspective is not None
                      and len(self.getHeadDomain(token
                                                 )._.vwp_perspective) > 0):
                    token._.vwp_perspective = \
                        self.getHeadDomain(token)._.vwp_perspective
                    continue

            # Special case: Sentence-modifying prepositional phrases
            # to 'for me', 'to me', 'according to me' establish viewpoint
            # for their objects for that clause
            # Prefer sentence-initial if they appear there. Only use
            # late in the sentence if no viewpoint has yet been established
            if (token.i < token.head.i
                or token.head._.vwp_perspective is None
                or (token.head._.vwp_perspective is not None
                    and len(token.head._.vwp_perspective) == 0)) \
               and (isRoot(token)
                    or token.dep_ == 'ccomp'
                    or token.dep_ == 'csubj'
                    or token.dep_ == 'csubjpass'
                    or token.dep_ == 'dep'
                    or token.dep_ == 'nsubj'):
                subj = getPrepObject(token, ['accord', 'to', 'for'])
                if token.dep_ != 'dep' \
                   and subj is not None \
                   and not subj._.location \
                   and (subj._.animate or subj._.vwp_sourcetext):
                    self.registerViewpoint(hdoc, token, subj)
                    self.registerViewpoint(hdoc, token.head, subj)
                    self.getHeadDomain(token)._.head_perspective = [subj.i]

            # Special case (npadvmod expressions like 'way'
            # in expressions like 'The way I see it, ...')
            # Treat the viewpoint of the npadvmod as viewpoint
            # for the sentence.
            if token.dep_ == 'npadvmod' and token.lemma_ in ['way', 'manner'] \
               and 'relcl' in [child.dep_ for child in token.children]:
                for child in token.children:
                    if child.dep_ == 'relcl':
                        subj = self.findViewpoint(
                            child, child, '', token, hdoc)
                        if subj is not None \
                           and (subj._.animate or subj._.vwp_sourcetext):
                            self.registerViewpoint(hdoc, token, subj)
                            self.getHeadDomain(token)._.head_perspective = \
                                [subj.i]
                            self.getHeadDomain(token)._.head_perspective = \
                                [subj.i]
                            break

            # Nouns used as predicates take the matrix viewpoint
            # for their domain
            if token.dep_ in ['attr', 'dobj'] \
               and token._.has_governing_subject:
                token._.vwp_perspective = token._.head_perspective
                continue

            # Evaluation words as subjects take the viewpoint of the
            # explicit or implicit speaker/thinker/sayer
            if token._.vwp_evaluation \
               and token.dep_ in ['nsubj', 'nsubpass']:
                # root clause, speaker perspective
                if token.head == token.head.head \
                   or (token.head.dep_ == 'conj'
                       and token.head.head == token.head.head.head):
                    token._.vwp_perspective = token._.head_perspective
                    continue
                else:
                    subj = self.findViewpoint(token,
                                              token.head,
                                              '',
                                              token,
                                              hdoc)
                    if subj is not None \
                       and (subj._.animate or subj._.vwp_sourcetext):
                        self.registerViewpoint(hdoc, token, subj)
                    continue

            # Subjective assessment words that are minor syntactic sisters
            # to the head of the domain take the viewpoint of the
            # explicit or implicit speaker/thinker/sayer
            if (token._.vwp_evaluation
                or token._.vwp_evaluated_role
                or token._.vwp_hedge
                or token._.vwp_possibility
                or token._.vwp_future
                or token._.subjectVerbInversion) \
               and token.dep_ in ['det',
                                  'aux',
                                  'auxpass',
                                  'neg',
                                  'amod',
                                  'compound',
                                  'advmod',
                                  'npadvmod',
                                  'cc',
                                  'punct']:
                token._.vwp_perspective = \
                    self.getHeadDomain(token.head)._.head_perspective

                # root clause, speaker perspective
                if isRoot(token.head):
                    token._.vwp_perspective = \
                        token.head._.head_perspective

                if token.head.dep_ == 'conj' \
                   and isRoot(token.head.head):
                    token._.vwp_perspective = \
                        token.head.head._.head_perspective
                continue

            # Abstract nouns as (prepositional) objects of viewpoint predicates
            # e.g., I believe in censorship
            if token.pos_ == 'NOUN' \
               and token._.vwp_abstract \
               and token.dep_ == 'pobj' \
               and (token.head.head._.vwp_argument
                    or token.head.head._.vwp_cognitive
                    or token.head.head._.vwp_communication) \
               and token.head.head._.governing_subject is not None \
               and hdoc[token.head.head._.governing_subject]._.animate:
                self.registerViewpoint(hdoc,
                                       token.head.head,
                                       hdoc[
                                            token.head.head._.governing_subject
                                            ])
                self.registerViewpoint(hdoc,
                                       token,
                                       hdoc[
                                           token.head.head._.governing_subject
                                           ])

            # Abstract nouns as (direct) objects of viewpoint predicates
            # e.g., I detest censorship
            if token._.vwp_abstract \
               and token.dep_ == 'dobj' \
               and (token.head._.vwp_argument
                    or token.head._.vwp_cognitive
                    or token.head._.vwp_communication) \
               and token.head._.governing_subject is not None \
               and hdoc[token.head._.governing_subject]._.animate:
                self.registerViewpoint(hdoc,
                                       token.head,
                                       hdoc[token.head._.governing_subject])
                self.registerViewpoint(hdoc,
                                       token,
                                       hdoc[token.head._.governing_subject])
                continue

            # Viewpoint predicates
            if self.viewpointPredicate(token):
                # Set viewpoint for non-emotional impact predicates
                # with an animate governing subject
                if not token._.vwp_emotional_impact \
                   and token._.governing_subject is not None:
                    if token._.vwp_abstract \
                       and hdoc[token._.governing_subject]._.animate:
                        self.registerViewpoint(hdoc,
                                               token,
                                               hdoc[token._.governing_subject])
                    if token._.governing_subject is not None \
                       and 'poss' in [child.dep_
                                      for child
                                      in hdoc[token._.governing_subject
                                              ].children
                                      if child._.animate]:
                        for child in hdoc[token._.governing_subject].children:
                            if child.dep_ == 'poss' \
                               and child._.animate:
                                self.registerViewpoint(hdoc, token, child)
                                token._.governing_subject = child.i
                                continue
                    continue

            # Possessives for viewpoint predicates define viewpoint
            if token._.animate and token.dep_ == 'poss' \
               and (token.head._.vwp_cognitive
                    or token.head._.vwp_communication
                    or token.head._.vwp_argument
                    or token.head._.vwp_plan
                    or token.head._.vwp_emotion
                    or token.head._.vwp_emotional_impact):
                self.registerViewpoint(hdoc, token, token)
                self.registerViewpoint(hdoc, token.head, token)

            # General case -- find and register viewpoint expressions
            if self.viewpointPredicate(token):

                subj = self.findViewpoint(token,
                                          token,
                                          '',
                                          token,
                                          hdoc)
                if subj is not None \
                   and not token._.vwp_emotional_impact \
                   and (subj._.animate or subj._.vwp_sourcetext):
                    self.registerViewpoint(hdoc, token, subj)

        for token in hdoc:

            # Viewpoint predicates that are objects of certain
            # sentence-level PPs establish viewpoint for the
            # whole clause
            if token._.vwp_perspective is not None:
                if token.dep_ == 'pobj' \
                   and token.head.dep_ == 'prep' \
                   and (token.head.head.dep_ is None
                        or isRoot(token.head.head)
                        or token.head.head.dep_ in ['ccomp',
                                                    'csubj',
                                                    'csubjpass']):
                    if not isRoot(token.head.head):
                        token.head._.vwp_perspective = \
                            token.head.head.head._.vwp_perspective
                    token.head.head._.vwp_perspective = token._.vwp_perspective

            # Viewpoint predicates that are adverbial sentence-level modifiers
            # establish viewpoint for the whole clause
            while (token.head.pos_ != 'NOUN'
                   and isRoot(token)
                   and token._.vwp_perspective is not None
                   and ((token.dep_ == 'advcl'
                         or token.dep_ == 'advmod'
                         or token.dep_ == 'acomp'
                         or token.dep_ == 'npadvmod')
                        and not tough_complement(token)
                        and not raising_complement(token))):
                if token.head._.vwp_perspective is None \
                   or len(token.head._.vwp_perspective) == 0:
                    token.head._.vwp_perspective = token._.vwp_perspective
                    token = token.head
                else:
                    break
                continue

        # Spread the viewpoint assignment to all tokens within
        # the scope of the viewpoint markers we've found
        self.percolateViewpoint(getRoots(hdoc))

        for token in hdoc:
            if token._.vwp_perspective is None:
                token._.vwp_perspective = \
                    self.getHeadDomain(token)._.head_perspective
        return hdoc

    def markImplicitSubject(self, item, hdoc):
        """
         Use the dependency relations among words in the parse to
         identify the understood subject of predicates.
        """

        if item._.has_governing_subject:
            return item._.governing_subject

        found = False
        subj = getSubject(item)
        obj = getObject(item)

        ###################################################################
        # default governing subject is the explicit syntactic subject     #
        ###################################################################
        if subj is not None:
            item._.has_governing_subject = True
            item._.governing_subject = subj.i
            # We do not set found to True because this is potentially
            # overrideable

        ###################################################################
        # Assign subject of a relative clause to the modified noun        #
        # if subject is explicitly or implicitly a relative pronoun       #
        # but to the explicit subject otherwise                           #
        ###################################################################
        if item.dep_ == 'relcl' \
           and tensed_clause(item):
            # examples: the idea that upset him
            # the idea that he suggested /
            if subj and subj.tag_ in ['WDT', 'WP']:
                subj = item.head
                item._.has_governing_subject = True
                item._.governing_subject = subj.i
                found = True
            # the idea he gave me /
            # the idea that I like /
            # the idea I like
            elif subj:
                item._.has_governing_subject = True
                item._.governing_subject = subj.i
                found = True

        # infinitival adjectival clauses (my plan to win,
        # I have a plan to win, I developed a plan to win)
        # take their governing subject from the matrix object
        # in a double-object construction and from the matrix
        # subject in other object constructions
        if not found and item.dep_ == 'acl' \
           and not tensed_clause(item):
            if item.head._.has_governing_subject:
                item._.has_governing_subject = True
                item._.governing_subject = \
                    item.head._.governing_subject
                found = True
            elif item.dep_ in ['dobj', 'pobj']:
                if matrixobj is not None \
                   and item != matrixobj:
                    item._has_governing_subject = True
                    item._.governing_subject = matrixobj.i
                    found = True
                elif matrixsubj is not None:
                    item._.has_governing_subject = True
                    item._.governing_subject = matrixsubj.i
                    found = True

        if item.dep_ == 'conj' and getSubject(item) is not None:
            found = True

        ###############################################################
        # Conjuncts inherit their subject                             #
        # from the first conjunct                                     #
        ###############################################################
        dependency = item.dep_
        matrixitem = item
        while (matrixitem != matrixitem.head
               and matrixitem.dep_ == 'conj'
               and matrixitem != matrixitem.head
               and (getSubject(matrixitem) is None
                    or getSubject(matrixitem).tag_ in ['WDT', 'WP'])):
            matrixitem = matrixitem.head

        # Setup.
        if matrixitem.head._.has_governing_subject:
            if hdoc[matrixitem.head._.governing_subject
                    ]._.has_governing_subject \
               and not hdoc[matrixitem.head._.governing_subject
                            ]._.animate \
               and (hdoc[matrixitem.head._.governing_subject
                         ]._.concreteness is None
                    or hdoc[matrixitem.head._.governing_subject
                            ]._.concreteness < 4):
                matrixsubj = \
                    hdoc[hdoc[matrixitem.head._.governing_subject
                              ]._.governing_subject]
            else:
                matrixsubj = hdoc[matrixitem.head._.governing_subject]
        else:
            matrixsubj = getSubject(matrixitem.head)

            # Correction to get the head noun
            # not the subj. relative pronoun for
            # a relative clause
            if matrixsubj is not None \
               and matrixsubj.tag_ in ['WDT', 'WP']:
                matrixsubj = matrixsubj.head.head

        matrixobj = getObject(matrixitem.head)

        # Now inherit the governing subject from the first conjunct
        if item.dep_ == 'conj' \
           and matrixitem._.has_governing_subject:
            item._.has_governing_subject = True
            item._.governing_subject = \
                matrixitem._.governing_subject
            found = True

        ###################################################################
        # Embedded non-finite, subjectless predicates take the            #
        # nearest c-commanding argument as their subject, i.e.,           #
        # matrix object if present, matrix subject otherwise              #
        ###################################################################
        if not tensed_clause(item) \
           and matrixitem.dep_ in ['attr',
                                   'oprd',
                                   'prep',
                                   'dobj',
                                   'acomp',
                                   'ccomp',
                                   'pcomp',
                                   'xcomp',
                                   'dep']:

            if item._.has_governing_subject:
                found = True

            # Matrix object is the governing subject
            elif (matrixsubj is not None
                  and matrixobj is not None
                  and matrixobj.tag_ not in ['WDT', 'WP']
                  and item != matrixobj):
                # Exception: indirect objects (first in two
                # dobj children) are not subjects of the
                # true direct object
                if item.dep_ != 'dobj':
                    if not containsDistinctReference(item, matrixobj, hdoc) \
                       or item.dep_ in ['ccomp', 'xcomp']:
                        item._.has_governing_subject = True
                        item._.governing_subject = matrixobj.i
                    else:
                        item._.has_governing_subject = False
                    found = True

            # Otherwise, matrix subject is the governing subject
            # with some restrictions on adjectives and nouns
            # (either they license raising/tough-movement constructions
            #  or, for nouns, we allow implicit subjects only in special
            # cases involving light verbs and abstract traits)
            elif (matrixsubj is not None
                  and (matrixobj is None or item != matrixobj)
                  and (item.pos_ in ['VERB',
                                     'AUX',
                                     'ADP']
                       or (item.pos_ in ['ADJ',
                                         'NOUN',
                                         'PROPN',
                                         'PRON',
                                         'CD',
                                         'NUM']
                           and (item.head._.vwp_tough
                                or item.head._.vwp_raising
                                or item.head._.vwp_seem
                                or item.head._.vwp_cognitive
                                or item.head._.vwp_emotion
                                or item.head._.vwp_character
                                or item.dep_ == 'attr'
                                or item.dep_ == 'oprd'
                                or (item.dep_ == 'dobj'
                                    and (item._.abstract_trait
                                         or item.head.lemma_
                                         in getLightVerbs())))))):

                item._.has_governing_subject = True
                if item.pos_ == 'ADP':
                    item._.governing_subject = matrixitem.head.i
                else:
                    if not containsDistinctReference(item, matrixsubj, hdoc) \
                       or item.dep_ in ['ccomp', 'xcomp']:
                        item._.has_governing_subject = True
                        item._.governing_subject = matrixsubj.i
                    else:
                        item._.has_governing_subject = False
                    found = True

            if matrixsubj is not None \
               and matrixobj is not None \
               and (matrixobj.head.lemma_ in getLightVerbs()
                    or matrixobj.head._.vwp_cognitive
                    or matrixobj.head._.vwp_emotion):
                matrixobj._.has_governing_subject = True
                matrixobj._.governing_subject = matrixsubj.i

        # The governing subject of an attributive adjective
        # or a preposition or a clause modifying a noun
        # is the noun is modifies
        if not found \
           and item.dep_ in ['amod', 'nmod', 'prep', 'acl'] \
           and item.head.pos_ in ['NOUN', 'PROPN', 'PRON', 'DET', 'CD', 'NUM']:
            item._.has_governing_subject = True
            item._.governing_subject = item.head.i
            found = True

        # The governing subject of a preposition or adverb
        # modifying a verb is that verb's governing subject,
        # except for sentence adverbs (which are listed by type)
        # and transition words
        if not found \
           and (item.dep_ in ['prep', 'advmod', 'neg', 'prt']
                or item.pos_ == 'ADV'):
            if item.dep_ in ['neg'] \
               or item._.vwp_hedge \
               or item._.transition:
                item._.has_governing_subject = True
                item._.governing_subject = item.head.i
                found = True
            elif (item.head.pos_ in ['VERB', 'AUX', 'MD']
                  and item.head._.has_governing_subject
                  and not item._.vwp_necessity
                  and not item._.vwp_certainty
                  and not item._.vwp_probability
                  and not item._.vwp_importance
                  and not item._.vwp_risk
                  and not item._.vwp_expectancy
                  and not item._.vwp_emphasis
                  and not item._.vwp_information
                  and not item._.transition):
                item._.has_governing_subject = True
                item._.governing_subject = item.head._.governing_subject
                found = True
            # otherwise it is the modified word
            else:
                item._.has_governing_subject = True
                item._.governing_subject = item.head.i
                found = True

        # Prepositional and object complements that are gerunds w/o
        # a marked subject take the subject of the closest ancestor
        # with a subject or object commanding them
        if not found \
           and item.dep_ in ['ccomp', 'pcomp'] \
           and item.tag_ == 'VBG' \
           and getSubject(item) is None:
            last = item
            head = item.head
            while (head != head.head
                   and getObject(head) is None
                   and getSubject(head) is None):
                last = head
                head = head.head
            obj = getObject(head)
            subj = getSubject(head)
            if obj is not None and obj != last:
                item._.has_governing_subject = True
                item._.governing_subject = obj.i
                found = True
            elif subj is not None:
                item._.has_governing_subject = True
                item._.governing_subject = subj.i
                found = True

        # Participles also take the subject of the clause they modify
        if not found \
           and item.dep_ in ['advcl'] \
           and item.tag_ == 'VBG' \
           and (getSubject(item) is None
                or item.pos_ != 'VERB'):
            last = item
            head = item.head
            while (head != head.head
                   and getSubject(head) is None):
                last = head
                head = head.head
            obj = getObject(head)
            subj = getSubject(head)
            if obj is not None and obj != last and obj._.has_governing_subject:
                item._.has_governing_subject = True
                item._.governing_subject = obj._.governing_subject
                found = True
            elif subj is not None and subj._.has_governing_subject:
                item._.has_governing_subject = True
                item._.governing_subject = subj._.governing_subject
                found = True

        for child in item.children:
            self.markImplicitSubject(child, hdoc)

        return item._.governing_subject

    def isHeadDomain(self, node):
        """
        Definition of what nodes count as head domains for viewpoint
        """

        # e.g., 'make us lazier'
        if node.dep_ in ['ccomp', 'oprd'] \
           and takesBareInfinitive(node.head):
            return True

        # e.g., 'than ever before'
        if node.dep_ in ['pcomp'] and node.pos_ == 'ADV':
            return True

        # e.g., 'with out the invention of the internet'
        if node.dep_ == 'pcomp' \
           and node.tag_ == 'RP':
            return True

        # conjuncts are head domain if the word they're conjoined with is
        if node.dep_ == 'conj' \
           and self.isHeadDomain(node.head):
            return True

        if not isRoot(node) \
           and node.dep_ not in ['ccomp', 'pcomp', 'csubj', 'csubjpass'] \
           and not (tensed_clause(node)
                    and node.dep_ in ['advcl', 'relcl']) \
           and not (tensed_clause(node)
                    and node.dep_ == 'conj'
                    and not self.isHeadDomain(node.head)) \
           and not (node.dep_ == 'acl'
                    and node.tag_ in ['VBD', 'VBN']) \
           and not (node.dep_ == 'acl'
                    and node.tag_ in ['VBG']
                    and node.head.pos_ == 'NOUN') \
           and not (node.dep_ in ['amod']
                    and (node._.vwp_evaluation
                         or node._.vwp_cognitive
                         or node._.vwp_communication
                         or node._.vwp_argument)) \
            or (node.pos_ == 'VERB'
                and not tensed_clause(node)
                and node.dep_ in ['csubj', 'csubjpass', 'ccomp', 'pcomp']):
            return True
        return False

    def getHeadDomain(self, node):
        """
        Find the head of a viewpoint domain.
        """
        start = node.text

        # Complements and relative clauses are opaque to viewpoint
        while self.isHeadDomain(node) and node != node.head:
            node = node.head
        return node

    def percolateViewpoint(self, nodes: list, barriers=[]):
        """
         Viewpoint is defined relative to head domains (basically,
         tensed clauses). We need to associate each node with the
         viewpoint of its immediately-dominating head domain, which
         this function does by assigning each token to the perspective
         associated with its domain head. Normally, this will be the
         subject of each tensed clause, if the verb is a viewpoint
         predicate. E.g., in 'Jenna believes that Mary is happy that
         Bill is going to school', the proposition 'Bill is going to
         school' is assessed from Mary's perspective, whereas the
         proposition 'Mary is happy that Bill is going to school'
          is assessed from Jenna's perspective.
        """

        for node in nodes:

            if node.head == self.getHeadDomain(node.head) \
               and node.head.dep_ == 'conj' \
               and node.head.i not in barriers:
                barriers.append(node.head.i)
            elif (node == self.getHeadDomain(node)
                  and node.dep_ == 'conj'
                  and node.head.i not in barriers):
                barriers.append(node.i)
            elif (node.dep_ in ['csubj', 'csubjpass']
                  and node.head.i not in barriers):
                barriers.append(node.i)

            if node._.vwp_perspective is None:
                if not isRoot(self.getHeadDomain(node)) \
                   and (node != self.getHeadDomain(node)
                        or node.dep_ not in ['det',
                                             'aux',
                                             'auxpass',
                                             'neg',
                                             'amod',
                                             'advmod',
                                             'npadvmod',
                                             'cc',
                                             'punct']):

                    if self.getHeadDomain(node).i not in barriers:
                        node._.vwp_perspective = \
                            self.getHeadDomain(node).head._.vwp_perspective
                elif node == self.getHeadDomain(node):
                    node._.vwp_perspective = \
                        self.getHeadDomain(node.head)._.head_perspective
                else:
                    node._.vwp_perspective = \
                        self.getHeadDomain(node)._.head_perspective
            else:
                if (node._.governing_subject is None
                   or len(node._.vwp_perspective)) == 0 \
                   and node.head.lemma_ in getLightVerbs():
                    node.head._.vwp_perspective = node._.vwp_perspective
            for child in node.children:

                if child._.vwp_perspective is not None:
                    self.percolateViewpoint([child], barriers)
                    continue

                if child.dep_ not in ['det',
                                      'aux',
                                      'auxpass',
                                      'neg',
                                      'amod',
                                      'advmod',
                                      'npadvmod',
                                      'cc',
                                      'punct']:

                    found = False
                    if self.getHeadDomain(child).i not in barriers:
                        if node._.vwp_perspective is not None:
                            child._.vwp_perspective = node._.vwp_perspective
                        elif (not found and node != self.getHeadDomain(node)
                              and node.i < self.getHeadDomain(node).i):
                            if child._.vwp_perspective is not None \
                               and not isRoot(self.getHeadDomain(node)) \
                               and len(child._.vwp_perspective) == 0:

                                child._.vwp_perspective = \
                                         self.getHeadDomain(
                                             node).head._.vwp_perspective
                            elif child._.vwp_perspective is None:
                                child._.vwp_perspective = \
                                    self.getHeadDomain(child
                                                       )._.head_perspective
                        elif (child.i < node.i
                              and child.i < self.getHeadDomain(node).i):
                            if child._.vwp_perspective is None:
                                child._.vwp_perspective = \
                                    self.getHeadDomain(child
                                                       )._.head_perspective
                else:
                    child._.vwp_perspective = \
                        self.getHeadDomain(child)._.head_perspective

                self.percolateViewpoint([child], barriers)

    def set_direct_speech_spans(self, hdoc):
        """
          Find direct speech spans (recognized because the head word
          of the span has the extension attribute vwp_direct_speech
          set true) and record the token index for the speaker,
          the start of the direct speech segment, and the end of
          the direct speech segment.
        """
        dspans = []
        for token in hdoc:
            if token._.vwp_direct_speech:
                speaker = token._.vwp_speaker_refs
                addressee = token._.vwp_addressee_refs
                left = token.left_edge
                right = token.right_edge

                # Edge case: quotation mark for a following quote
                # Adjust span boundaries to include the quotation
                # mark in the following not the preceding span
                if left.i > 0 \
                   and quotationMark(hdoc[left.i - 1]) \
                   and hdoc[left.i]._.vwp_quoted:
                    left = hdoc[left.i - 1]
                if right.i + 1 < len(hdoc) and right.i > 1:
                    if quotationMark(hdoc[right.i]) \
                       and not hdoc[right.i - 1]._.vwp_quoted:
                        hdoc[right.i]._.vwp_quoted = True
                        right = hdoc[right.i - 1]

                # Define spans
                if speaker is not None \
                   and addressee is not None \
                   and len(speaker) > 0 \
                   and len(addressee) > 0:
                    dspans.append([speaker,
                                  addressee,
                                  left.i,
                                  right.i])
                elif speaker is not None and len(speaker) > 0:
                    dspans.append([speaker,
                                  [],
                                  left.i,
                                  right.i])
                elif addressee is not None and len(addressee) > 0:
                    dspans.append([[],
                                  addressee,
                                  left.i,
                                  right.i])
        lastSpeaker = None
        lastAddressee = None
        lastItem = None
        newSpans = []
        # Merge lists of references so the same list
        # is assigned to spans with references
        # included in a later list
        for span in reversed(dspans):
            newItem = [span[0], span[1], [[span[2], span[3]]]]
            if lastSpeaker is None:
                newItem[0] = list(sorted(span[0]))
                newItem[1] = list(sorted(span[1]))
                if newItem not in newSpans:
                    newSpans.append(newItem)
            else:
                included = True
                for item in span[0]:
                    if item not in lastSpeaker:
                        included = False
                if included:
                    newSpans[len(newSpans) - 1][2].insert(0,
                                                          [span[2],
                                                              span[3]])
                else:
                    newItem[0] = list(sorted(span[0]))
                    newItem[1] = list(sorted(span[1]))
                    if newItem not in newSpans:
                        newSpans.append(newItem)
            lastSpeaker = span[0]
            lastAddressee = span[1]
            lastItem = newItem
        hdoc._.vwp_direct_speech_spans = list(reversed(newSpans))
        # add the token level flag and do the standard format output       
        for span in newSpans:
            (speaker, addressee, subspans) = span
            for subspan in subspans:
                [left, right] = subspan
                entry = newSpanEntry('direct speech',
                                     left,
                                     right,
                                     hdoc,
                                     [speaker, addressee])
                hdoc._.direct_speech_spans.append(entry)
            locs = span[2]
            for loc in locs:
                leftEdge = loc[0]
                rightEdge = loc[1]
                for item in hdoc[leftEdge:rightEdge]:
                    if item._.vwp_quoted:
                        item._.vwp_in_direct_speech = True

    def egocentric(self, hdoc):
        """
         Viewpoint domains that contain evaluation language like should
         or perhaps with explicit or implicit first-person viewpoint
         count as egocentic. 'Unfortunately, Jenna came in last' ->
         egocentric since implicitly it is the speaker who views the
         event as unfortunate
        """
        count = 0
        domainList = []
        for token in hdoc:
            if (token._.vwp_evaluation or token._.vwp_hedge) \
               and len(token._.vwp_perspective) == 0 \
               and self.getHeadDomain(token).i not in domainList:
                domainList.append(self.getHeadDomain(token).i)
            for perspective in token._.vwp_perspective:
                if ((token._.vwp_evaluation or token._.vwp_hedge)
                    and (hdoc[perspective].text.lower()
                         in first_person_pronouns)
                    and (self.getHeadDomain(token).i
                         not in domainList)):
                    domainList.append(self.getHeadDomain(token).i)
        for token in hdoc:
            if self.getHeadDomain(token).i in domainList:
                token._.vwp_egocentric = True

    def allocentric(self, doc):
        count = 0
        domainList = []
        for token in doc:
            include = True
            if len(token._.vwp_perspective) == 0:
                include = False
            else:
                for perspective in token._.vwp_perspective:
                    if perspective is None \
                       or (doc[perspective].text.lower()
                           in first_person_pronouns) \
                       or (doc[perspective].text.lower()
                           in second_person_pronouns):
                        include = False
            if include:
                token._.vwp_allocentric = True

    def set_perspective_spans(self, hdoc, calculatePerspective=True):
        """
         Create structures identifying which tokens have been
         assigned to which point of view

         'implicit' means there is no explicit viewpoint noun
         controlling this item. That means that it is probably,
         implicitly, the speaker.

         'explicit_1' means that there is an explicit first person
         pronoun controlling this item, in a context where the first
         person pronoun should be interpreted as the speaker

         'explicit_2' means that there is an explicit second person
         pronoun controlling this item, in a context where the second
         person pronoun should be interpreted as the addressee

         'explicit_3' means that there is an explicit third person
         viewpoint nominal controlling this item. We list individual
         token offsets under the offset of the controlling viewpoint
         nominal.

         We create the following data structures
         1. a basic perspective dictionary that shows whose viewpoint
            each token is interpreted from
         2. a stance marker dictionary that shows which words cue
            subjective evaluation of propositions from particular points
            of view.
         3. An emotion marker dictionary that shows which emotions are
            associated with which animate nominals in the text.
         4. a character marker dictionary that shows which character
            traits are associated with which animate nominals in the text.
         5. a propositional attitude dictionary that identifies words
            like think, feel, etc. that express attitudes toward propositions
            by agents.

        """
        pspans = {}
        pspans['implicit'] = {}
        pspans['explicit_1'] = []
        pspans['explicit_2'] = []
        pspans['explicit_3'] = {}

        stance_markers = {}
        stance_markers['implicit'] = {}
        stance_markers['explicit_1'] = []
        stance_markers['explicit_2'] = []
        stance_markers['explicit_3'] = {}

        emotional_markers = {}
        emotional_markers['explicit_1'] = []
        emotional_markers['explicit_2'] = []
        emotional_markers['explicit_3'] = {}

        character_markers = {}
        character_markers['explicit_1'] = []
        character_markers['explicit_2'] = []
        character_markers['explicit_3'] = {}

        propositional_attitudes = {}
        propositional_attitudes['implicit'] = []
        propositional_attitudes['implicit_3'] = []
        propositional_attitudes['explicit_1'] = []
        propositional_attitudes['explicit_2'] = []
        propositional_attitudes['explicit_3'] = {}

        theory_of_mind_sentences = []

        # print_parse_tree(hdoc)

        for token in hdoc:

            referentID = ResolveReference(token, hdoc)
            if token.i not in referentID:
                referentID.append(token.i)

            self.mark_argument_words(token, hdoc)
            hdeps = [child.dep_ for child in token.head.children]
            if calculatePerspective:
                stance_markers, pspans = \
                    self.stance_perspective(token,
                                            hdoc,
                                            referentID,
                                            stance_markers,
                                            pspans,
                                            hdeps)
                emotional_markers = \
                    self.emotional_impact(token,
                                          hdoc,
                                          emotional_markers)
            if token._.has_governing_subject:
                if calculatePerspective:
                    character_markers = \
                        self.character_traits(token,
                                              hdoc,
                                              referentID,
                                              character_markers)
                    emotional_markers = \
                        self.emotion_predicates(token,
                                                hdoc,
                                                emotional_markers)
                    theory_of_mind_sentences = \
                        self.theory_of_mind_sentences(token,
                                                      hdoc,
                                                      theory_of_mind_sentences)
            propositional_attitudes = \
                self.propositional_attitudes(token,
                                             hdoc,
                                             propositional_attitudes,
                                             hdeps)
        propositional_attitudes = \
            self.cleanup_propositional_attitudes(propositional_attitudes,
                                                 hdoc)
        hdoc._.vwp_perspective_spans = pspans
        hdoc._.vwp_stance_markers = stance_markers
        hdoc._.vwp_character_traits = character_markers
        hdoc._.vwp_emotion_states = emotional_markers
        hdoc._.vwp_propositional_attitudes = propositional_attitudes
        hdoc._.vwp_social_awareness = theory_of_mind_sentences

    def mark_transition_argument_words(self, hdoc):
    
        tp = hdoc._.transition_word_profile
        for item in tp[3]:
            if item[4] not in ['temporal', 'PARAGRAPH']:
                if item[2] == item[3]:
                    hdoc[item[2]]._.vwp_argument = True
                    hdoc[item[2]]._.transition = True
                else:
                    for i in range(item[2], item[3] + 1):
                        if i >= len(hdoc):
                            break
                        hdoc[i]._.vwp_argument = True
                        hdoc[i]._.transition = True

    def mark_argument_words(self, token, hdoc):

        tp = hdoc._.transition_word_profile
        for item in tp[3]:
            if item[4] not in ['temporal', 'PARAGRAPH']:
                if item[2] == item[3]:
                    hdoc[item[2]]._.vwp_argumentation = True
                    hdoc[item[2]]._.transition = True
                else:
                    for i in range(item[2], item[3] + 1):
                        hdoc[i]._.vwp_argumentation = True
                        hdoc[i]._.transition = True
        if (token._.vwp_cognitive
           or token._.vwp_communication
           or token._.vwp_argument
           or token._.vwp_evaluation) \
           and token.dep_ == 'pobj' \
           and token.head.head._.abstract_trait:
            if token.head.head.dep_ == 'pobj':
                if (token.head.head.head.head._.vwp_cognitive
                    or token.head.head.head.head._.vwp_communication
                    or token.head.head.head.head._.vwp_argument
                    or token.head.head.head.head._.vwp_information
                    or token.head.head.head.head._.vwp_cognitive
                    or (token.head.head.head.dep_ != 'conj'
                        and (token.head.head.head.head._.vwp_abstract
                             or token.head.head.head.head._.vwp_possession
                             or token.head.head.head.head._.vwp_cause
                             or token.head.head.head.head._.vwp_relation))):
                    token._.vwp_argumentation = True
                    token.head._.vwp_argumentation = True
                    token.head.head._.vwp_argumentation = True
                    token.head.head.head._.vwp_argumentation = True
                    token.head.head.head.head._.vwp_argumentation = True
                    for child in token.head.head.children:
                        if child.pos_ in ['DET'] \
                           or child.tag_ in ['WP', 'WP$', 'JJR', 'JJS']:
                            child._.vwp_argumentation = True

            elif token.head.head.dep_ == 'dobj':
                if (token.head.head.head._.vwp_cognitive
                    or token.head.head.head._.vwp_communication
                    or token.head.head.head._.vwp_argument
                    or token.head.head.head._.vwp_information
                    or token.head.head.head._.vwp_communication
                    or token.head.head.head._.vwp_cognitive
                    or (token.head.head.dep_ != 'conj'
                        and (token.head.head.head._.vwp_abstract
                             or token.head.head.head._.vwp_possession
                             or token.head.head.head._.vwp_cause
                             or token.head.head.head._.vwp_relation))):
                    token._.vwp_argumentation = True
                    token.head._.vwp_argumentation = True
                    token.head.head._.vwp_argumentation = True
                    token.head.head.head._.vwp_argumentation = True
                    for child in token.head.head.children:
                        if child.pos_ in ['DET'] \
                           or child.tag_ in ['WP', 'WP$', 'JJR', 'JJS']:
                            child._.vwp_argumentation = True
        if token.dep_ == 'pobj' \
           and token.head is not None \
           and token.head.dep_ == 'prep' \
           and token.head.head is not None \
           and token.head.head.head is not None \
           and token.head.head.dep_ == 'advmod' \
           and (token._.vwp_cognitive
                or token._.vwp_communication
                or token._.vwp_argument) \
           and (token.head.head.head._.vwp_cognitive
                or token.head.head.head._.vwp_communication
                or token.head.head.head._.vwp_argument):
            if (token._.vwp_cognitive
                or token._.vwp_communication
                or token._.vwp_argument
                or token._.vwp_information
                or token._.vwp_communication
                or token._.vwp_cognitive
                or (token.dep_ != 'conj'
                    and (token._.vwp_abstract
                         or token._.vwp_possession
                         or token._.vwp_cause
                         or token._.vwp_relation))):
                token._.vwp_argumentation = True
            token.head._.vwp_argumentation = True
            token.head.head._.vwp_argumentation = True
            token.head.head.head._.vwp_argumentation = True

        if token.dep_ == 'pobj' \
           and token.head is not None \
           and token.head.dep_ == 'prep' \
           and token.head.head is not None \
           and token.head.head.pos_ in ['VERB',
                                        'NOUN',
                                        'ADJ',
                                        'ADV']:
            for child in token.head.head.children:
                if child._.vwp_evaluation \
                   or child._.vwp_hedge \
                   or child._.vwp_cognitive \
                   or child._.vwp_communication \
                   or child._.vwp_argument:
                    if (token._.vwp_cognitive
                        or token._.vwp_communication
                        or token._.vwp_argument
                        or token._.vwp_information
                        or token._.vwp_communication
                        or token._.vwp_cognitive
                        or (child.dep_ != 'conj'
                            and (token._.vwp_abstract
                                 or token._.vwp_possession
                                 or token._.vwp_cause
                                 or token._.vwp_relation))):
                        token._.vwp_argumentation = True
                        token.head._.vwp_argumentation = True
                        child._.vwp_argumentation = True
                    if token.dep_ != 'conj' \
                       and token.head.dep_ != 'conj' \
                       and (token._.vwp_cognitive
                            or token.head.head._.vwp_communication
                            or token.head.head._.vwp_argument
                            or token.head.head._.vwp_information
                            or token.head.head._.vwp_communication
                            or token.head.head._.vwp_cognitive
                            or (token.head.dep_ != 'conj'
                                and (token.head.head._.vwp_abstract
                                     or token.head.head._.vwp_possession
                                     or token.head.head._.vwp_cause
                                     or token.head.head._.vwp_relation))):
                        token.head.head._.vwp_argumentation = True
                        child._.vwp_argumentation = True

        if token.dep_ in ['pobj', 'advmod'] \
           and isRoot(token.head.head) \
           and (token._.vwp_cognitive
                or token._.vwp_communication
                or token._.vwp_argument
                or token._.vwp_evaluation
                or token._.vwp_hedge):
            for child in token.head.head.children:
                if (child._.vwp_cognitive
                    or child._.vwp_communication
                    or child._.vwp_argument
                    or child._.vwp_evaluation
                    or child._.vwp_hedge
                    or child._.vwp_information
                    or child._.vwp_communication
                    or child._.vwp_cognitive
                    or (child.dep_ != 'conj'
                        and (child._.vwp_abstract
                             or child._.vwp_possession
                             or child._.vwp_cause
                             or child._.vwp_relation))):
                    token._.vwp_argumentation = True
                    child._.vwp_argumentation = True

        if token._.has_governing_subject \
           and (token._.vwp_evaluation
                or token._.vwp_hedge) \
           and (hdoc[token._.governing_subject]._.vwp_argument
                or hdoc[token._.governing_subject]._.vwp_information
                or hdoc[token._.governing_subject]._.vwp_communication
                or hdoc[token._.governing_subject]._.vwp_cognitive
                or (token.dep_ != 'conj'
                    and (hdoc[token._.governing_subject]._.vwp_abstract
                         or hdoc[token._.governing_subject]._.vwp_possession
                         or hdoc[token._.governing_subject]._.vwp_relation))):
            token._.vwp_argumentation = True
            hdoc[token._.governing_subject]._.vwp_argumentation = True
            for child in hdoc[token._.governing_subject].children:
                if child.lemma_ not in core_temporal_preps \
                    and (child.pos_ in ['DET', 'AUX']
                         or child.tag_ in function_word_tags
                         or (child.tag_ == 'RB'
                             and (child._.vwp_evaluation
                                  or child._.vwp_hedge))
                         or child.lemma_ in personal_or_indefinite_pronoun
                         or child._.vwp_argument
                         or child._.vwp_information
                         or child._.vwp_communication
                         or child._.vwp_cognitive
                         or child._.vwp_evaluation
                         or child._.vwp_hedge
                         or (child.dep_ != 'conj'
                             and (child._.vwp_abstract
                                  or child._.vwp_possession
                                  or child._.vwp_cause
                                  or child._.vwp_relation
                                  or child.dep_ in ['neg']))):
                    child._.vwp_argumentation = True

        if token.dep_ == 'amod' \
           and token._.vwp_evaluation \
           and token.head.dep_ in subject_or_object_nom \
           and (isRoot(token.head.head)
                or token._.vwp_argument
                or token._.vwp_communication
                or token._.vwp_cognitive):
            token._.vwp_argumentation = True

        if token.dep_ in adjectival_predicates \
           and (token._.vwp_evaluation
                or token._.vwp_raising
                or token._.vwp_hedge
                or token._.vwp_argument
                or token._.vwp_communication
                or token._.vwp_cognitive):
            for child in token.head.children:
                if child.dep_ in clausal_complements:
                    if child._.vwp_argument \
                       or child._.vwp_information \
                       or child._.vwp_communication \
                       or child._.vwp_cognitive \
                       or child._.vwp_evaluation \
                       or child._.vwp_hedge \
                       or (child.dep_ != 'conj'
                           and (child._.vwp_abstract
                                or child._.vwp_possession
                                or child._.vwp_possession
                                or child._.vwp_cause
                                or child._.vwp_relation
                                or child.dep_ in ['neg'])):
                        token._.vwp_argumentation = True
                        child._.vwp_argumentation = True
                    for grandchild in child.children:
                        if child._.vwp_information \
                           or grandchild._.vwp_communication \
                           or grandchild._.vwp_cognitive \
                           or (grandchild.dep_ != 'conj'
                                and (grandchild._.vwp_abstract
                                     or grandchild._.vwp_possession
                                     or grandchild._.vwp_evaluation
                                     or grandchild._.vwp_hedge
                                     or grandchild._.vwp_possession
                                     or grandchild.dep_ in ['neg'])):
                            token._.vwp_argumentation = True
                            grandchild._.vwp_argumentation = True

        if token.dep_ == 'amod' \
           and (token._.vwp_evaluation
                or token._.vwp_hedge) \
           and token.head.dep_ == 'pobj' \
           and token.head.head.dep_ == 'prep' \
           and token.head.head.head is not None \
           and token.head.head.head.pos_ in ['VERB', 'NOUN', 'ADJ', 'ADV']:
            for child in token.head.head.head.children:
                if child._.vwp_evaluation \
                   or child._.vwp_hedge \
                   or child._.vwp_cognitive \
                   or child._.vwp_communication \
                   or child._.vwp_argument:
                    token._.vwp_argumentation = True
                    if token.head._.vwp_evaluation \
                       or token.head._.vwp_hedge \
                       or token.head._.vwp_cognitive \
                       or token.head._.vwp_communication \
                       or token.head._.vwp_argument:
                        token.head._.vwp_argumentation = True
                        token.head.head._.vwp_argumentation = True
                    if token.head.head.head._.vwp_evaluation \
                       or token.head.head.head._.vwp_hedge \
                       or token.head.head.head._.vwp_cognitive \
                       or token.head.head.head._.vwp_communication \
                       or token.head.head.head._.vwp_argument:
                        token.head.head.head._.vwp_argumentation = True
                    child._.vwp_argumentation = True

        if token.dep_ == 'amod' \
           and (token._.vwp_evaluation
                or token._.vwp_hedge) \
           and (in_modal_scope(token.head)
                or not token.head._.in_past_tense_scope) \
           and (token.head._.vwp_argument
                or token.head._.vwp_information
                or token.head._.vwp_communication
                or token.head._.vwp_cognitive
                or (token.dep_ != 'conj'
                    and (token.head._.vwp_abstract
                         or token.head._.vwp_possession
                         or token.head._.vwp_relation))) \
           and token.head._.is_academic:
            token._.vwp_argumentation = True
            token.head._.vwp_argumentation = True

        if token.lemma_ in quantifying_determiners \
           and (token.head._.vwp_argument
                or token.head._.vwp_information
                or token.head._.vwp_communication
                or token.head._.vwp_cognitive
                or (token.dep_ != 'conj'
                    and (token.head._.vwp_abstract
                         or token.head._.vwp_possession
                         or token.head._.vwp_cause
                         or token.head._.vwp_relation
                         or token.head._.abstract_trait))):
            for child in token.head.children:
                if token != child:
                    if child._.vwp_cognitive \
                       or child._.vwp_communication \
                       or child._.vwp_argument \
                       or child._.vwp_hedge \
                       or child._.vwp_evaluation:
                        token._.vwp_argumentation = True
                        token.head._.vwp_argumentation = True
                        child._.vwp_argumentation = True

        if ((token._.vwp_evaluation or token._.vwp_hedge)
            and token.head.dep_ not in ['conj']
            and (token.head.head._.vwp_argument
                 or token.head.head._.vwp_cognitive
                 or token.head.head._.vwp_communication)):
            token._.vwp_argumentation = True
            token.head.head._.vwp_argumentation = True

        if (token._.vwp_cognitive
           or token._.vwp_communication
           or token._.vwp_argument
           or token._.vwp_evaluation
           or token._.vwp_hedge):
            if token.dep_ in subject_or_object_nom:
                for child in token.head.children:
                    if child.dep_ in clausal_complements:
                        token._.vwp_argumentation = True
                        token.head._.vwp_argumentation = True
                        for grandchild in child.children:
                            if grandchild.dep_ == 'mark':
                                grandchild._.vwp_argumentation = True

            if token.dep_ in ['aux', 'auxpass', 'advmod', 'npadvmod'] \
               and (isRoot(token.head)
                    and (token.head._.vwp_cognitive
                         or token.head._.vwp_communication
                         or token.head._.vwp_argument
                         or token.head._.vwp_hedge
                         or token.head._.vwp_evaluation)):
                token._.vwp_argumentation = True
                token.head._.vwp_argumentation = True
                for child in token.head.children:
                    if child.pos_ in ['AUX', 'ADV']:
                        child._.vwp_argumentation = True

            if token.dep_ in ['aux', 'auxpass', 'advmod', 'npadvmod'] \
               and (token._.vwp_evaluation or token._.vwp_hedge):
                for child in token.head.children:
                    if child.dep_ != 'conj':
                        for grandchild in child.children:
                            if (grandchild.dep_ != 'conj'
                                and (grandchild._.vwp_cognitive
                                     or grandchild._.vwp_communication
                                     or grandchild._.vwp_argument
                                     or grandchild._.vwp_hedge
                                     or grandchild._.vwp_evaluation)):
                                token._.vwp_argumentation = True
                                grandchild._.vwp_argumentation = True
                            for ggrandchild in grandchild.children:
                                if (ggrandchild.dep_ != 'conj'
                                    and (ggrandchild._.vwp_cognitive
                                         or ggrandchild._.vwp_communication
                                         or ggrandchild._.vwp_argument
                                         or ggrandchild._.vwp_hedge
                                         or ggrandchild._.vwp_probability
                                         or ggrandchild._.vwp_evaluation)):
                                    token._.vwp_argumentation = True
                                    ggrandchild._.vwp_argumentation = True

            if token.dep_ in ['aux', 'auxpass', 'advmod', 'npadvmod']:
                for child in token.children:
                    if child.dep_ in adjectival_predicates \
                       and (child._.vwp_cognitive
                            or child._.vwp_communication
                            or child._.vwp_argument
                            or child._.vwp_hedge
                            or child._.vwp_evaluation):
                        token._.vwp_argumentation = True
                        child._.vwp_argumentation = True

            if token.dep_ is None \
               or isRoot(token) \
               or (isRoot(token.head)
                   and token.dep_ == 'attr') \
               or token.head._.vwp_cognitive \
               or token.head._.vwp_communication \
               or token.head._.vwp_argument \
               or token.head._.vwp_hedge \
               or token.head._.vwp_evaluation:
                for child in token.children:
                    if clausal_subject_or_complement(child):
                        token._.vwp_argumentation = True
                        token.head._.vwp_argumentation = True
                        for child in token.head.children:
                            if child.lemma_ not in core_temporal_preps \
                               and (child.pos_ in ['DET', 'AUX']
                                    or child.tag_ in function_word_tags
                                    or (child.tag_ == 'RB'
                                        and (child._.vwp_evaluation
                                             or child._.vwp_hedge))
                                    or child.lemma_
                                    in personal_or_indefinite_pronoun
                                    or child._.vwp_argument
                                    or child._.vwp_information
                                    or child._.vwp_communication
                                    or child._.vwp_cognitive
                                    or (child.dep_ != 'conj'
                                        and (child._.vwp_abstract
                                             or child._.vwp_possession
                                             or child._.vwp_relation
                                             or child._.vwp_cause
                                             or child.dep_ in ['neg']))):
                                child._.vwp_argumentation = True

                        if token.i + 1 < len(token.doc) \
                           and token.nbor(1) is not None \
                           and token.nbor(1).dep_ == 'mark':
                            token.nbor(1)._.vwp_argumentation = True

        if token.dep_ == 'amod' \
           and (token._.vwp_evaluation
                or token._.vwp_hedge) \
           and (in_modal_scope(token.head)
                or not token.head._.in_past_tense_scope) \
           and (token.head._.vwp_cognitive
                or token.head._.vwp_communication
                or token.head._.vwp_argument
                or token.head._.vwp_hedge
                or token.head._.vwp_evaluation):
            token._.vwp_argumentation = True
            token.head._.vwp_argumentation = True

        if token.dep_ == 'amod' \
           and (token._.vwp_evaluation
                or token._.vwp_hedge) \
           and (in_modal_scope(token.head.head)
                or not token.head.head._.in_past_tense_scope) \
           and (token.head.head._.vwp_cognitive
                or token.head.head._.vwp_communication
                or token.head.head._.vwp_argument
                or token.head.head._.vwp_hedge
                or token.head.head._.vwp_evaluation):
            token._.vwp_argumentation = True
            token.head.head._.vwp_argumentation = True

        if token.dep_ == 'amod' \
           and (token._.vwp_evaluation
                or token._.vwp_hedge):
            if (token.head.dep_ == 'pobj'
                and (in_modal_scope(token.head.head.head)
                     or not token.head.head.head._.in_past_tense_scope)
                and token.head.head.head is not None
                and (token.head.head.head._.vwp_cognitive
                     or token.head.head.head._.vwp_communication
                     or token.head.head.head._.vwp_argument
                     or token.head.head.head._.vwp_evaluation)):
                token._.vwp_argumentation = True
                token.head.head.head._.vwp_argumentation = True

        if token.dep_ == 'prep' \
           and (in_modal_scope(token.head)
                or not token.head._.in_past_tense_scope) \
           and (token.head._.vwp_argument
                or token.head._.vwp_information
                or token.head._.vwp_communication
                or token.head._.vwp_cognitive
                or (token.dep_ != 'conj'
                    and (token.head._.vwp_abstract
                         or token.head._.vwp_possession
                         or token.head._.vwp_cause
                         or token.head._.vwp_relation))):
            for child in token.children:
                if (child.dep_ == 'pobj'
                    and (child._.vwp_cognitive
                         or child._.vwp_communication
                         or child._.vwp_argument
                         or child._.vwp_evaluation)):
                    token._.vwp_argumentation = True
                    token.head._.vwp_argumentation = True
                    child._.vwp_argumentation = True

        if (token.dep_ in complements
            and (token.head._.vwp_argument
                 or token.head._.vwp_information
                 or token.head._.vwp_communication
                 or token.head._.vwp_cognitive
                 or (token.dep_ != 'conj'
                     and (token.head._.vwp_abstract
                          or token.head._.vwp_possession
                          or token.head._.vwp_cause
                          or token.head._.vwp_relation)))):
            for child in token.children:
                if child.dep_ == 'dobj' \
                   and (child._.vwp_cognitive
                        or child._.vwp_communication
                        or child._.vwp_argument
                        or child._.vwp_hedge
                        or child._.vwp_evaluation):
                    token._.vwp_argumentation = True
                    token.head._.vwp_argumentation = True
                    child._.vwp_argumentation = True

        if (token._.vwp_cognitive
           or token._.vwp_communication
           or token._.vwp_argument):
            for offset in getLinkedNodes(token):
                if (token.text.lower() != hdoc[offset].text.lower()
                    and (token.head.dep_ is None
                         or isRoot(token.head)
                         or isRoot(hdoc[offset].head))):
                    if hdoc[offset].dep_ == 'prep':
                        for child in hdoc[offset].children:
                            if child._.vwp_cognitive \
                               or child._.vwp_communication \
                               or child._.vwp_argument:
                                token._.vwp_argumentation = True
                                hdoc[offset]._.vwp_argumentation = True
                                child._.vwp_argumentation = True
                    if (hdoc[offset]._.vwp_argument
                        or hdoc[offset]._.vwp_evaluation
                        or hdoc[offset]._.vwp_hedge
                        or hdoc[offset]._.vwp_information
                        or hdoc[offset]._.vwp_communication
                        or hdoc[offset]._.vwp_cognitive
                        or (hdoc[offset].dep_ != 'conj'
                            and (hdoc[offset]._.vwp_abstract
                                 or hdoc[offset]._.vwp_possession
                                 or hdoc[offset]._.vwp_cause
                                 or hdoc[offset]._.vwp_relation
                                 or hdoc[offset].dep_ in complements))):
                        if token._.is_academic \
                           or hdoc[offset]._.is_academic:
                            token._.vwp_argumentation = True
                            hdoc[offset]._.vwp_argumentation = True
                            for child in hdoc[offset].children:
                                if (child._.vwp_evaluation
                                    or child._.vwp_hedge
                                    or child._.vwp_information
                                    or child._.vwp_communication
                                    or child._.vwp_cognitive
                                    or (child.dep_ != 'conj'
                                        and (child._.vwp_abstract
                                             or child._.vwp_possession
                                             or child._.vwp_cause
                                             or child._.vwp_relation))):
                                    child._.vwp_argumentation = True
                                    for grandchild in child.children:
                                        if ((grandchild.pos_ in ['DET', 'AUX']
                                             or grandchild.tag_
                                             in function_word_tags
                                             or (grandchild.tag_ == 'RB'
                                                 and (grandchild._.
                                                      vwp_evaluation
                                                      or grandchild._.
                                                      vwp_hedge))
                                             or grandchild.lemma_
                                             in personal_or_indefinite_pronoun
                                             )):
                                            grandchild._.vwp_argumentation\
                                                = True

        if token._.vwp_evaluation:
            for offset in getLinkedNodes(token):
                if (hdoc[offset]._.vwp_argument
                    or hdoc[offset]._.vwp_information
                    or hdoc[offset]._.vwp_communication
                    or hdoc[offset]._.vwp_cognitive
                    or (hdoc[offset].dep_ != 'conj'
                        and (hdoc[offset]._.vwp_abstract
                             or hdoc[offset]._.vwp_possession
                             or hdoc[offset]._.vwp_cause
                             or hdoc[offset]._.vwp_relation))):
                    if token._.is_academic or hdoc[offset]._.is_academic:
                        token._.vwp_argumentation = True
                        hdoc[offset]._.vwp_argumentation = True

        if (token._.vwp_certainty
           or token._.vwp_probability
           or token._.vwp_qualification
           or token._.vwp_emphasis
           or token._.vwp_outcome
           or token._.vwp_surprise
           or token._.vwp_illocutionary
           or token._.vwp_generalization) \
           and token.dep_ == 'advmod' \
           and isRoot(token.head):
            token._.vwp_argumentation = True
            if token.head.is_stop:
                token.head._.vwp_argumentation = True

        if token.head._.vwp_argumentation:
            if token.head.head.dep_ == 'prep':
                token.head.head._.vwp_argumentation = True

        if token._.vwp_argumentation:
            if token.head.dep_ == 'prep':
                token.head._.vwp_argumentation = True

            if token.i + 1 < len(token.doc) \
               and token.nbor(1) is not None \
               and not token.nbor(1).lemma_ in core_temporal_preps \
               and ((token.nbor(1).pos_ in ['DET']
                     and token.nbor(1).lemma_ in quantifying_determiners)
                    or token.nbor(1).tag_ in function_word_tags
                    or (token.nbor(1).tag_ == 'RB'
                        and (token.nbor(1)._.vwp_evaluation
                             or token.nbor(1)._.vwp_hedge))
                    or token.nbor(1)._.vwp_argument
                    or token.nbor(1)._.vwp_information
                    or token.nbor(1)._.vwp_communication
                    or token.nbor(1)._.vwp_cognitive
                    or token.nbor(1)._.vwp_abstract
                    or token.nbor(1)._.vwp_possession
                    or token.nbor(1)._.vwp_cause
                    or token.nbor(1)._.vwp_relation
                    or token.nbor(1).dep_ in ['neg']):
                token.nbor(1)._.vwp_argumentation = True

            for child in token.children:
                if child.lemma_ not in core_temporal_preps \
                 and child.dep_ != 'conj' \
                 and (child.pos_ in ['DET', 'AUX']
                      or child.tag_ in function_word_tags
                      or (child.tag_ == 'RB'
                          and (child._.vwp_evaluation
                               or child._.vwp_hedge))
                      or child.lemma_ in personal_or_indefinite_pronoun
                      or child._.vwp_argument
                      or child._.vwp_information
                      or child._.vwp_communication
                      or child._.vwp_cognitive
                      or (child.dep_ != 'conj'
                          and (child._.vwp_possession
                               or child._.vwp_relation
                               or child._.vwp_abstract
                               or child._.vwp_cause
                               or child.dep_ in ['neg']))):
                    child._.vwp_argumentation = True
                    for grandchild in child.children:
                        if grandchild.tag_ in ['TO',
                                               'MD',
                                               'IN',
                                               'SCONJ',
                                               'WRB',
                                               'WDT',
                                               'ADP',
                                               'WP',
                                               'WP$',
                                               'EX',
                                               'ADP',
                                               'JJR',
                                               'JJS',
                                               'RBR',
                                               'RBS'] \
                           or (grandchild.tag_ == 'RB'
                               and (grandchild._.vwp_evaluation
                                    or grandchild._.vwp_hedge)):
                            grandchild._.vwp_argumentation = True
                            break
                    if token.tag_ == 'NOUN':
                        for grandchild in token.children:
                            if (token._.vwp_hedge
                                or token._.vwp_evaluation
                                or token._.vwp_argument
                                or token._.vwp_information
                                or token._.vwp_communication
                                or token._.vwp_cognitive
                                or (grandchild.dep_ != 'conj'
                                    and (token._.vwp_abstract
                                         or token._.vwp_possession
                                         or token._.vwp_cause
                                         or token._.vwp_relation))):
                                grandchild._.vwp_argumentation = True

            if token._.vwp_argumentation \
               and token.tag_ in ['RB', 'MD', 'SCONJ']:
                for child in token.head.children:
                    if child.lemma_ not in core_temporal_preps \
                       and (child.tag_ in function_word_tags
                            or (child.tag_ == 'RB'
                                and (child._.vwp_evaluation
                                     or child._.vwp_hedge))
                            or child.lemma_ in personal_or_indefinite_pronoun
                            or child._.vwp_argument
                            or child._.vwp_information
                            or child._.vwp_communication
                            or child._.vwp_cognitive
                            or (child.dep_ != 'conj'
                                and (child._.vwp_abstract
                                     or child._.vwp_possession
                                     or child._.vwp_relation
                                     or child._.vwp_cause
                                     or child.dep_ in ['neg']))):
                        child._.vwp_argumentation = True
                        if (child.i + 1 < len(child.doc) and
                            (child.nbor(1)._.vwp_argument
                             or child.nbor(1)._.vwp_information
                             or child.nbor(1)._.vwp_communication
                             or child.nbor(1)._.vwp_cognitive
                             or child.nbor(1)._.vwp_abstract
                             or child.nbor(1)._.vwp_possession
                             or child.nbor(1)._.vwp_cause
                             or child.nbor(1)._.vwp_relation
                             or child.nbor(1).dep_ in ['neg'])):
                            child.nbor(1)._.vwp_argumentation = True
                        for grandchild in child.children:
                            if grandchild._.vwp_argument \
                               or grandchild._.vwp_information \
                               or grandchild._.vwp_communication \
                               or grandchild._.vwp_cognitive \
                               or (grandchild.dep_ != 'conj'
                                   and (grandchild._.vwp_abstract
                                        or grandchild._.vwp_possession
                                        or grandchild._.vwp_relation
                                        or grandchild._.vwp_cause
                                        or grandchild.dep_ in ['neg'])):
                                grandchild._.vwp_argumentation = True

    def stance_perspective(self,
                           token,
                           hdoc,
                           referentID,
                           stance_markers,
                           pspans,
                           hdeps):

        # We mark words as argument-related if they involve
        # RELATIONSHIPS with other words from key categories
        # in our argument/perspective lexicon

        # If the subject and the verb disagree on viewpoint, correct
        # the verb to agree with the subject
        if token._.vwp_perspective is None \
           and token.dep_ in ['nsubj', 'nsubjpass', 'csubj', 'csubjpass']:
            if token._.vwp_perspective != token.head._.vwp_perspective:
                token.head._.vwp_perspective = token._.vwp_perspective

        # Cleanup for stray cases where no viewpoint was assigned to ROOT
        if isRoot(token) \
           and token._.vwp_perspective is None:
            for child in token.children:
                if child._.vwp_perspective is not None:
                    token._.vwp_perspective = child._.vwp_perspective
                    break

        # Cleanup -- prepositions should be assigned to the same
        # viewpoint as their head
        if token.dep_ == 'prep' \
           and token._.vwp_perspective is None:
            token._.vwp_perspective = token.head._.vwp_perspective

        if token._.vwp_perspective is not None \
           and len(token._.vwp_perspective) == 0:
            controller = self.getHeadDomain(token).i
            csubj = hdoc[controller]._.governing_subject
            if (csubj is None or not hdoc[csubj]._.animate) \
               and getSubject(hdoc[controller]) is not None \
               and getSubject(hdoc[controller])._.animate:
                csubj = getSubject(hdoc[controller]).i
            if csubj is not None \
               and not hdoc[csubj]._.animate \
               and getSubject(hdoc[csubj]) is not None \
               and getSubject(hdoc[csubj])._.animate:
                csubj = getSubject(hdoc[csubj]).i

            if self.isHeadDomain(hdoc[controller]) \
               and len(getRoot(hdoc[controller])._.vwp_perspective) == 0:
                controller = getRoot(hdoc[controller]).i
            if csubj is not None \
               and hdoc[csubj].text.lower() in first_person_pronouns:
                domain = 'explicit_1'
                controller = csubj
            elif (csubj is not None
                  and hdoc[csubj].text.lower()
                  in second_person_pronouns):
                domain = 'explicit_2'
                controller = csubj
            else:
                domain = 'implicit'

            if domain not in pspans:
                pspans[domain] = {}
            if token._.transition \
               and token._.vwp_argumentation \
               and isRoot(token):
                if domain in ['explicit_1', 'explicit_2']:
                    if token.i not in pspans[domain]:
                        pspans[domain].append(token.i)
                        pspans[domain] = sorted(pspans[domain].copy())
                else:
                    if str(token.i) not in pspans[domain]:
                        pspans[domain][str(token.i)] = []
                    if token.i not in pspans[domain][str(token.i)]:
                        pspans[domain][str(token.i)].append(token.i)
                        pspans[domain][str(token.i)] = \
                            sorted(pspans[domain][str(token.i)].copy())
            else:
                if domain in ['explicit_1', 'explicit_2']:
                    if token.i not in pspans[domain]:
                        pspans[domain].append(token.i)
                        pspans[domain] = sorted(pspans[domain].copy())
                elif token.text.lower() in first_person_pronouns:
                    if token.i not in pspans['explicit_1']:
                        pspans['explicit_1'].append(token.i)
                        pspans['explicit_1'] = \
                            sorted(pspans['explicit_1'].copy())
                elif token.text.lower() in second_person_pronouns:
                    if token.i not in pspans['explicit_2']:
                        pspans['explicit_2'].append(token.i)
                        pspans['explicit_2'] = \
                            sorted(pspans['explicit_2'].copy())
                else:
                    if str(controller) not in pspans[domain]:
                        pspans[domain][str(controller)] = []
                    if token.i not in pspans[domain][str(controller)]:
                        pspans[domain][str(controller)].append(token.i)
                        pspans[domain][str(controller)] = \
                            sorted(pspans[domain][str(controller)].copy())
            deps = [child.dep_ for child in token.children]
            if token._.transition \
               and token._.vwp_argumentation \
               and isRoot(token):
                if domain in ['explicit_1', 'explicit_2']:
                    if token.i not in stance_markers[domain]:
                        stance_markers[domain].append(token.i)
                        stance_markers[domain] = sorted(
                            stance_markers[domain].copy())
                else:
                    if str(token.i) not in stance_markers[domain]:
                        stance_markers[domain][str(token.i)] = []
                    if token.i not in stance_markers[domain][str(token.i)]:
                        stance_markers[domain][str(token.i)].append(token.i)
                        stance_markers[domain][str(token.i)] = \
                            sorted(stance_markers[
                                domain][str(token.i)].copy())
            elif (token._.vwp_evaluation
                  or token._.vwp_hedge
                  or token._.subjectVerbInversion
                  or ((token._.vwp_argument
                      or token._.vwp_cognitive
                      or token._.vwp_communication
                      or (token._.vwp_raising
                          and token._.vwp_probability))
                      and ('csubj' in deps
                           or 'ccomp' in deps
                           or 'xcomp' in deps
                           or 'acomp' in deps
                           or 'advcl' in deps
                           or 'prep' in deps
                           or 'acl' in deps
                           or 'csub' in hdeps
                           or 'ccomp' in hdeps
                           or 'xcomp' in hdeps
                           or 'acomp' in hdeps
                           or 'advcl' in hdeps
                           or 'prep' in hdeps
                           or 'acl' in hdeps))
                  or (token._.vwp_character
                      and token._.governing_subject is not None
                      and hdoc[token._.governing_subject]._.animate)
                  or (token._.vwp_evaluated_role
                      and token._.governing_subject is not None
                      and hdoc[token._.governing_subject]._.animate)):
                if token.i not in stance_markers[domain]:
                    if token.dep_ == 'amod' \
                       and (token._.vwp_evaluation
                            or token._.vwp_hedge):
                        if domain in ['explicit_1', 'explicit_2']:
                            if token.i not in stance_markers[domain]:
                                stance_markers[domain].append(token.i)
                                stance_markers[domain] = \
                                    sorted(stance_markers[domain].copy())
                        else:
                            if str(token.i) not in stance_markers[domain]:
                                stance_markers[domain][str(token.i)] = []
                            if token.i not in \
                               stance_markers[domain][str(token.i)]:
                                stance_markers[domain][
                                    str(token.i)].append(token.i)
                                stance_markers[domain][str(token.i)] = \
                                    sorted(stance_markers[
                                        domain][str(token.i)].copy())
                    else:
                        if domain in ['explicit_1', 'explicit_2']:
                            if token.i not in stance_markers[domain]:
                                stance_markers[domain].append(token.i)
                                stance_markers[domain] = sorted(
                                    stance_markers[domain].copy())
                        else:
                            if str(controller) not in stance_markers[domain]:
                                stance_markers[domain][str(controller)] = []
                            if token.i not in stance_markers[
                               domain][str(controller)]:
                                stance_markers[domain][
                                    str(controller)].append(token.i)
                                stance_markers[domain][str(controller)] = \
                                    sorted(stance_markers[
                                         'implicit'][str(controller)].copy())
        elif token is not None and token._.vwp_perspective is not None:
            for item in token._.vwp_perspective:
                controller = hdoc[item]
                if controller.text.lower() in first_person_pronouns \
                   and len(list(controller.children)) == 0 \
                   and not controller._.vwp_quoted:
                    if 'explicit_1' not in pspans:
                        pspans['explicit'] = []
                    if controller.i not in pspans['explicit_1'] \
                       and controller.i not in pspans['explicit_2']:
                        pspans['explicit_1'].append(controller.i)
                    for referent in referentID:
                        if referent not in pspans['explicit_1'] \
                           and referent not in pspans['explicit_2']:
                            pspans['explicit_1'].append(referent)
                            pspans['explicit_1'] = \
                                sorted(pspans['explicit_1'].copy())
                    if token._.vwp_evaluation \
                       or token._.vwp_hedge \
                       or token._.vwp_character \
                       or token._.vwp_evaluated_role:
                        if 'explicit_1' not in stance_markers:
                            stance_markers['explicit_1'] = []
                        if token.i not in stance_markers['explicit_1']:
                            stance_markers['explicit_1'].append(token.i)
                            stance_markers['explicit_1'] = \
                                sorted(stance_markers['explicit_1'].copy())

                elif (controller.text.lower() in second_person_pronouns
                      and len(list(controller.children)) == 0
                      and not controller._.vwp_quoted):
                    if 'explicit_2' not in pspans:
                        pspans['explicit'] = []
                    if controller.i not in stance_markers['explicit_2'] \
                       and controller.i not in stance_markers['implicit']:
                        stance_markers['explicit_2'].append(controller.i)
                    for referent in referentID:
                        if referent not in pspans['explicit_2'] \
                           and referent not in pspans['implicit']:
                            pspans['explicit_2'].append(referent)
                            pspans['explicit_2'] = sorted(pspans['explicit_2'])
                    if token._.vwp_evaluation \
                       or token._.vwp_hedge \
                       or token._.vwp_character \
                       or token._.vwp_evaluated_role:
                        if 'explicit_2' not in stance_markers:
                            stance_markers['explicit_2'] = []
                        if token.i not in stance_markers['explicit_2']:
                            stance_markers['explicit_2'].append(token.i)
                            stance_markers['explicit_2'] = \
                                sorted(stance_markers['explicit_2'].copy())
                else:
                    if token.text.lower() in first_person_pronouns:
                        if token.i not in pspans['explicit_1']:
                            pspans['explicit_1'].append(token.i)
                            pspans['explicit_1'] = sorted(
                                pspans['explicit_1'].copy())
                    if token.text.lower() in second_person_pronouns:
                        if token.i not in pspans['explicit_2']:
                            pspans['explicit_2'].append(token.i)
                            pspans['explicit_2'] = sorted(
                                pspans['explicit_2'].copy())
                    else:
                        if token.text.lower() in first_person_pronouns:
                            if token.i not in pspans['explicit_1']:
                                pspans['explicit_1'].append(token.i)
                                pspans['explicit_1'] = \
                                    sorted(pspans['explicit_1'].copy())
                            if 'explicit_1' not in stance_markers:
                                stance_markers['explicit_1'] = []
                            if token.i not in stance_markers['explicit_1']:
                                stance_markers['explicit_1'].append(token.i)
                                stance_markers['explicit_1'] = \
                                    sorted(stance_markers['explicit_1'].copy())
                        elif token.text.lower() in second_person_pronouns:
                            if token.i not in pspans['explicit_2']:
                                pspans['explicit_2'].append(token.i)
                                pspans['explicit_2'] = \
                                    sorted(pspans[
                                        'explicit_2'].copy())
                            if 'explicit_2' not in stance_markers:
                                stance_markers['explicit_2'] = []
                            if token.i not in stance_markers['explicit_2']:
                                stance_markers['explicit_2'].append(token.i)
                                stance_markers['explicit_2'] = \
                                    sorted(stance_markers['explicit_2'].copy())
                        else:
                            if 'explicit_3' not in pspans:
                                pspans['explicit_3'] = {}
                            if int(item) not in pspans['explicit_3']:
                                pspans['explicit_3'][int(item)] = []
                            for referent in referentID:
                                if referent not in \
                                   pspans['explicit_3'][int(item)] \
                                   and referent not in pspans['explicit_1'] \
                                   and referent not in pspans['explicit_2']:
                                    pspans['explicit_3'][
                                        int(item)].append(referent)
                                    pspans['explicit_3'][int(item)] = \
                                        sorted(pspans[
                                            'explicit_3'][int(item)].copy())
                        if token._.vwp_evaluation \
                           or token._.vwp_hedge \
                           or token._.vwp_character \
                           or token._.vwp_evaluated_role:
                            if 'explicit_3' not in stance_markers:
                                stance_markers['explicit_3'] = {}
                            if int(item) not in stance_markers['explicit_3']:
                                stance_markers['explicit_3'][int(item)] = []
                            if token.i not in stance_markers['explicit_3'] \
                               and token.i not in \
                               stance_markers['explicit_1'] \
                               and token.i not in \
                               stance_markers['explicit_2']:
                                stance_markers['explicit_3'][
                                    int(item)].append(token.i)
                                stance_markers['explicit_3'][int(item)] = \
                                    sorted(stance_markers[
                                        'explicit_3'][int(item)].copy())
        return stance_markers, pspans

    def emotional_impact(self, token, hdoc, emotional_markers):
        if token._.vwp_emotional_impact \
           and getLogicalObject(token) is not None:
            controllers = ResolveReference(getLogicalObject(token), hdoc)
            charname = ''
            for i, controller in enumerate(controllers):
                if i == 0:
                    charname = hdoc[controller].text.capitalize()
                elif i == len(controllers)-1:
                    charname += ' and ' \
                        + hdoc[controller].text.capitalize()
                else:
                    charname += ', ' \
                        + hdoc[controller].text.capitalize()

            for controller in controllers:
                if getLogicalObject(token)._.animate:
                    if hdoc[controller].text.lower() in first_person_pronouns \
                       and len(list(
                           hdoc[
                               controller].children)) == 0 \
                       and not hdoc[controller]._.vwp_quoted:
                        if token.i not in emotional_markers['explicit_1']:
                            emotional_markers['explicit_1'].append(token.i)
                    elif (hdoc[controller].text.lower()
                          in second_person_pronouns
                          and len(list(hdoc[controller].children)) == 0
                          and not hdoc[controller]._.vwp_quoted):
                        if token.i not in emotional_markers['explicit_2']:
                            emotional_markers['explicit_2'].append(token.i)
                    else:
                        if controller not in \
                           emotional_markers['explicit_3']:
                            emotional_markers[
                                'explicit_3'][charname] = []
                            if token.i not in \
                               emotional_markers[
                                   'explicit_3'][charname]:
                                emotional_markers[
                                    'explicit_3'][
                                        charname].append(token.i)
                        else:
                            if token.i not in \
                               emotional_markers[
                                   'explicit_3'][charname]:
                                if token.i not in \
                                   emotional_markers[
                                       'explicit_3'][charname]:
                                    emotional_markers[
                                         'explicit_3'][
                                             charname].append(token.i)
        return emotional_markers

    def character_traits(self, token, hdoc, referentID, character_markers):
        if (token.pos_ == 'NOUN' and token._.vwp_evaluated_role) \
           or token._.vwp_character:
            if token._.has_governing_subject:
                controllers = ResolveReference(
                    hdoc[token._.governing_subject], hdoc)
                charname = ''
                for i, controller in enumerate(controllers):
                    if i == 0:
                        charname = hdoc[controller].text.capitalize()
                    elif i == len(controllers)-1:
                        charname += ' and ' \
                            + hdoc[controller].text.capitalize()
                    else:
                        charname += ', ' + hdoc[controller].text.capitalize()
                for controller in controllers:
                    if hdoc[controller]._.animate:
                        if hdoc[controller].text.lower() in \
                           first_person_pronouns \
                           and len(list(
                               hdoc[controller].children)) == 0:
                            if referentID not in \
                               character_markers['explicit_1']:
                                for referent in referentID:
                                    if referent not in \
                                       character_markers[
                                           'explicit_1']:
                                        character_markers[
                                            'explicit_1'].append(
                                                referent)
                        elif hdoc[controller].text.lower() \
                            in second_person_pronouns \
                            and len(list(
                                hdoc[controller].children)) == 0:
                            if referentID not in \
                               character_markers['explicit_2']:
                                for referent in referentID:
                                    if referent not in \
                                       character_markers['explicit_2']:
                                        character_markers[
                                            'explicit_2'].append(
                                                referent)
                        else:
                            if charname not in \
                               character_markers['explicit_3']:
                                character_markers[
                                    'explicit_3'][charname] = []
                                for referent in referentID:
                                    if referent not in controllers:
                                        if referent not in \
                                           character_markers[
                                                'explicit_3'][charname]:
                                            character_markers[
                                                'explicit_3'][
                                                    charname].append(
                                                        referent)
                            else:
                                if referentID not in \
                                   character_markers[
                                       'explicit_3'][charname]:
                                    for referent in referentID:
                                        if referent \
                                           not in controllers:
                                            if referent not in \
                                               character_markers[
                                                   'explicit_3'][charname]:
                                                character_markers[
                                                    'explicit_3'][
                                                        charname].append(
                                                            referent)
                    break
        return character_markers

    def emotion_predicates(self, token, hdoc, emotional_markers):
        if token._.vwp_emotion:
            if token._.has_governing_subject:
                controllers = ResolveReference(
                    hdoc[token._.governing_subject], hdoc)
                for i, controller in enumerate(controllers):
                    if i == 0:
                        charname = \
                            hdoc[controller].text.capitalize()
                    elif i == len(controllers)-1:
                        charname += ' and '\
                            + hdoc[controller].text.capitalize()
                    else:
                        charname += ', ' + hdoc[controller].text.capitalize()
                for controller in controllers:
                    if hdoc[controller]._.animate:
                        if hdoc[controller].text.lower() \
                            in first_person_pronouns \
                            and len(list(
                              hdoc[controller].children)) == 0:
                            if token.i not in \
                               emotional_markers['explicit_1'] \
                                and not \
                                    hdoc[controller]._.vwp_quoted:
                                if token.i not in \
                                   emotional_markers['explicit_1']:
                                    emotional_markers[
                                        'explicit_1'].append(token.i)
                        elif hdoc[controller].text.lower() \
                            in second_person_pronouns \
                            and len(list(
                              hdoc[controller].children)) == 0:
                            if token.i not in \
                               emotional_markers['explicit_2'] \
                               and not hdoc[controller]._.vwp_quoted:
                                if token.i not in \
                                   emotional_markers['explicit_2']:
                                    emotional_markers[
                                        'explicit_2'].append(token.i)
                        else:
                            if charname not in \
                               emotional_markers['explicit_3']:
                                emotional_markers[
                                    'explicit_3'][charname] = []
                                if token.i not in \
                                   emotional_markers[
                                       'explicit_3'][charname]:
                                    emotional_markers[
                                        'explicit_3'][
                                            charname].append(token.i)
                            else:
                                if token.i not in emotional_markers[
                                   'explicit_3'][charname]:
                                    if token.i not in emotional_markers[
                                       'explicit_3'][
                                           charname]:
                                        emotional_markers[
                                            'explicit_3'][
                                                charname].append(token.i)
        return emotional_markers

    def propDomain(self, token, controller, hdoc):
        if controller is None:
            controller = token._.governing_subject
        dom = self.getHeadDomain(token)
        if controller is not None:
            ctrlr = hdoc[controller]
            while (ctrlr.dep_ is not None
                   and ctrlr != ctrlr.head
                   and not isRoot(ctrlr)
                   and ctrlr.pos_ != 'VERB'):
                ctrlr = ctrlr.head
            if ctrlr.i != dom.i:
                dom = ctrlr.head
        start = list(dom.subtree)[0].i
        end = list(dom.subtree)[len(list(dom.subtree))-1].i
        return ([start, end], controller, token.i)

    def cleanup_propositional_attitudes(self, propositional_attitudes, hdoc):
        for item in propositional_attitudes['implicit_3']:
            if item in propositional_attitudes['implicit']:
                propositional_attitudes['implicit'].remove(item)
        for item in propositional_attitudes['explicit_1']:
            if item in propositional_attitudes['implicit']:
                propositional_attitudes['implicit'].remove(item)
            if item in propositional_attitudes['implicit_3']:
                propositional_attitudes['implicit_3'].remove(item)
        for item in propositional_attitudes['explicit_2']:
            if item in propositional_attitudes['implicit']:
                propositional_attitudes['implicit'].remove(item)
            if item in propositional_attitudes['implicit_3']:
                propositional_attitudes['implicit_3'].remove(item)
        for domain in propositional_attitudes['explicit_3']:
            for item in propositional_attitudes['explicit_3'][domain]:
                if item in propositional_attitudes['implicit']:
                    propositional_attitudes['implicit'].remove(item)
                if item in propositional_attitudes['implicit_3']:
                    propositional_attitudes['implicit_3'].remove(item)

        # register which tokens are part of claims or discussions
        for item in propositional_attitudes['implicit']:
            if isinstance(item, tuple):
                for offset in range(item[0][0], item[0][1]):
                    hdoc[offset]._.vwp_claim = True
        for item in propositional_attitudes['explicit_1']:
            if isinstance(item, tuple):
                for offset in range(item[0][0], item[0][1]):
                    hdoc[offset]._.vwp_claim = True
        for item in propositional_attitudes['explicit_2']:
            if isinstance(item, tuple):
                for offset in range(item[0][0], item[0][1]):
                    hdoc[offset]._.vwp_claim = True
        for item in propositional_attitudes['implicit_3']:
            if isinstance(item, tuple):
                for offset in range(item[0][0], item[0][1]):
                    hdoc[offset]._.vwp_discussion = True
        for domain in propositional_attitudes['explicit_3']:
            if isinstance(item, tuple):
                for item in propositional_attitudes['explicit_3'][domain]:
                    for offset in range(item[0][0], item[0][1]):
                        hdoc[offset]._.vwp_discussion = True
                        
        # Now let's clean things up to put propositional attitudes
        # in the format we're standardizing to for sentence scale ranges
        reformatted = []
        for item in propositional_attitudes['implicit']:
            [[left, right], controller, proposition] = item
            entry = newSpanEntry('propositional attitudes',
                                 left,
                                 right,
                                 hdoc,
                                 'implicit')
            reformatted.append(entry)
        for item in propositional_attitudes['explicit_1']:
            [[left, right], controller, proposition] = item
            entry = newSpanEntry('propositional attitudes',
                                 left,
                                 right,
                                 hdoc,
                                 'explicit')
            reformatted.append(entry)
        for item in propositional_attitudes['explicit_2']:
            [[left, right], controller, proposition] = item
            entry = newSpanEntry('propositional attitudes',
                                 left,
                                 right,
                                 hdoc,
                                 'explicit_2')
            reformatted.append(entry)
        for domain in propositional_attitudes['explicit_3']:
            for item in propositional_attitudes['explicit_3'][domain]:
                [[left, right], controller, proposition] = item
                entry = newSpanEntry('propositional attitudes',
                                     left,
                                     right,
                                     hdoc,
                                     'explicit_3')
                reformatted.append(entry)
        return reformatted


    def propositional_attitudes(self,
                                token,
                                hdoc,
                                propositional_attitudes,
                                hdeps):

        if ((token.head._.vwp_say
            or token.head._.vwp_think
            or token.head._.vwp_perceive
            or token.head._.vwp_interpret
            or token.head._.vwp_argue
            or token.head._.vwp_argument
            or token.head._.vwp_emotion
            or token.head._.vwp_evaluation
            or (token._.transition
                and token._.transition_category is not None
                and token._.transition_category not in ['temporal',
                                                        'PARAGRAPH'])
            or any([(child._.vwp_say
                    or child._.vwp_think
                    or child._.vwp_perceive
                    or child._.vwp_interpret
                    or child._.vwp_argue
                    or child._.vwp_argument
                    or child._.vwp_emotion
                    or (child._.transition
                        and child._.transition_category is not None
                        and child._.transition_category not in ['temporal',
                                                                'PARAGRAPH'])
                    or child._.vwp_evaluation)
                   for child in token.head.children
                   if child.dep_ in ['acomp', 'attr', 'oprd']])
            or (token.head.text.lower() in ['am',
                                            'are',
                                            'is',
                                            'was',
                                            'were',
                                            'be',
                                            'being',
                                            'been']
                and token.head._.governing_subject is not None
                and hdoc[
                    token.head._.governing_subject]._.vwp_evaluation)
            or (token.head.text.lower() in ['am',
                                            'are',
                                            'is',
                                            'was',
                                            'were',
                                            'be',
                                            'being',
                                            'been'])
                and 'attr' not in hdeps
                and 'acomp' not in hdeps)
           and ((token.dep_ in ['ccomp', 'csubjpass', 'acl', 'xcomp', 'oprd']
                 and tensed_clause(token))
                or token.dep_ in 'relcl' and token.head.dep_ in ['nsubj',
                                                                 'nsubjpass',
                                                                 'attr']
                or (token.dep_ in ['ccomp', 'xcomp', 'oprd']
                    and ('dobj' in hdeps
                         or 'nsubjpass' in hdeps))
                or (token.dep_ == 'advcl'
                    and 'for' in [child.lemma_ for child in token.children
                                  if child.dep_ == 'mark'])
                or (token.dep_ == 'prep'
                    and 'pcomp' in [child.dep_ for child in token.children
                                    if tensed_clause(child)]))
           and ((getSubject(token.head) is None
                 and token.head._.governing_subject is None)
                or (token.head._.governing_subject is not None
                    and hdoc[token.head._.governing_subject].lemma_ == 'it'
                    and (token.head._.vwp_raising
                         or token.head._.vwp_tough
                         or token.head.text.lower()
                         in ['am',
                             'are',
                             'is',
                             'was',
                             'were',
                             'be',
                             'being',
                             'been']))
                or (token.head._.governing_subject is not None
                    and (hdoc[token.head._.governing_subject]._.animate
                         or hdoc[token.head._.governing_subject
                                 ]._.vwp_sourcetext
                         or hdoc[token.head._.governing_subject
                                 ]._.vwp_say
                         or hdoc[token.head._.governing_subject
                                 ]._.vwp_think
                         or hdoc[token.head._.governing_subject
                                 ]._.vwp_perceive
                         or hdoc[token.head._.governing_subject
                                 ]._.vwp_interpret
                         or hdoc[token.head._.governing_subject
                                 ]._.vwp_argue
                         or hdoc[token.head._.governing_subject
                                 ]._.vwp_argument
                         or hdoc[token.head._.governing_subject
                                 ]._.vwp_emotion
                         or (hdoc[token.head._.governing_subject
                                  ]._.transition
                             and hdoc[
                                      token.head._.governing_subject
                                      ]._.transition_category
                             is not None
                             and hdoc[
                                      token.head._.governing_subject
                                      ]._.transition_category
                             not in ['temporal', 'PARAGRAPH'])
                         or hdoc[token.head._.governing_subject
                                 ]._.vwp_evaluation)))) \
                or (token.dep_ in ['attr', 'oprd']
                    and (token.head._.vwp_say
                         or token.head._.vwp_think
                         or token.head._.vwp_perceive
                         or token.head._.vwp_interpret
                         or token.head._.vwp_argue
                         or token.head._.vwp_argument
                         or token.head._.vwp_emotion
                         or (token._.transition
                             and token._.transition_category is not None
                             and token._.transition_category
                             not in ['temporal', 'PARAGRAPH'])
                         or token.head._.vwp_evaluation)
                    and token.head._.governing_subject is not None
                    and (hdoc[token.head._.governing_subject].dep_ == 'csubj'
                         or hdoc[token.head._.governing_subject
                                 ].tag_ == '_SP')):
            domHead = self.propDomain(token.head,
                                      token.head._.governing_subject,
                                      hdoc)
            if token.head._.governing_subject is None:
                if token.head.pos_ == 'VERB':
                    # Imperatives
                    if isRoot(token) \
                       and token.head.lemma_ == token.head.text.lower() \
                       and not all(['Tense=' in str(child.morph)
                                    for child in token.head.children]):
                        if domHead not in \
                           propositional_attitudes['explicit_2']:
                            propositional_attitudes[
                                'explicit_2'].append(domHead)
                        elif is_definite_nominal(token.head):
                            if domHead not in \
                               propositional_attitudes['implicit_3']:
                                propositional_attitudes[
                                    'implicit_3'].append(domHead)
                    # Controlled verbs
                    else:
                        if is_definite_nominal(token.head):
                            if domHead not in \
                               propositional_attitudes['implicit_3']:
                                propositional_attitudes[
                                    'implicit_3'].append(domHead)
                            elif token.head.pos_ == 'NOUN':
                                if domHead not in \
                                   propositional_attitudes['implicit_3']:
                                    propositional_attitudes[
                                        'implicit_3'].append(domHead)
                        else:
                            if domHead not in \
                               propositional_attitudes['implicit']:
                                propositional_attitudes[
                                    'implicit'].append(domHead)
                # nominals w/o subject
                elif token.head.pos_ in ['ADJ']:
                    if domHead not in propositional_attitudes['implicit']:
                        propositional_attitudes['implicit'].append(domHead)
                elif token.head.pos_ in ['NOUN']:
                    if is_definite_nominal(token.head):
                        if domHead not in \
                           propositional_attitudes['implicit_3']:
                            propositional_attitudes[
                                'implicit_3'].append(domHead)
                    else:
                        if domHead not in propositional_attitudes['implicit']:
                            propositional_attitudes['implicit'].append(domHead)
            else:
                controllers = ResolveReference(
                    hdoc[token.head._.governing_subject], hdoc)
                for controller in controllers:
                    domHead = self.propDomain(token.head, controller, hdoc)
                    if (hdoc[
                        token.head._.governing_subject].lemma_ == 'it'
                        and (token.head._.vwp_raising
                             or token.head._.vwp_tough
                             or token.head._.vwp_evaluation
                             or token.head.dep_ == 'xcomp'
                             or any([child._.vwp_evaluation
                                     for child in token.head.children]))):
                        if token.head.pos_ in ['NOUN']:
                            if is_definite_nominal(token.head):
                                if token.head.i not in \
                                   propositional_attitudes['implicit_3']:
                                    propositional_attitudes[
                                        'implicit_3'].append(token.head.i)
                            else:
                                if token.head.i not in \
                                   propositional_attitudes['implicit']:
                                    propositional_attitudes[
                                        'implicit'].append(domHead)
                        else:
                            if token.head.i not in \
                               propositional_attitudes['implicit']:
                                propositional_attitudes[
                                    'implicit'].append(domHead)
                    elif (hdoc[controller].text.lower()
                          in first_person_pronouns
                          and len(list(hdoc[controller].children)) == 0):
                        if token.head.i not in \
                           propositional_attitudes['explicit_1'] \
                           and not hdoc[controller]._.vwp_quoted:
                            propositional_attitudes[
                                'explicit_1'].append(domHead)
                    elif (hdoc[controller].text.lower()
                          in second_person_pronouns
                          and len(list(hdoc[controller].children)) == 0):
                        if domHead not in propositional_attitudes[
                           'explicit_2'] \
                           and not hdoc[controller]._.vwp_quoted:
                            propositional_attitudes[
                                'explicit_2'].append(domHead)
                    else:
                        if hdoc[controller]._.vwp_say \
                           or hdoc[controller]._.vwp_think \
                           or hdoc[controller]._.vwp_perceive \
                           or hdoc[controller]._.vwp_interpret \
                           or hdoc[controller]._.vwp_argue \
                           or hdoc[controller]._.vwp_argument \
                           or hdoc[controller]._.vwp_emotion \
                           or (hdoc[controller]._.transition
                               and (hdoc[controller]._.transition_category
                                    is not None)
                               and (hdoc[controller]._.transition_category
                                    not in ['temporal', 'PARAGRAPH'])) \
                           or hdoc[controller]._.vwp_evaluation:
                            if is_definite_nominal(token.head):
                                if domHead not in \
                                   propositional_attitudes['implicit_3']:
                                    propositional_attitudes[
                                        'implicit_3'].append(domHead)
                            else:
                                if domHead not in \
                                   propositional_attitudes['implicit']:
                                    propositional_attitudes[
                                        'implicit'].append(domHead)
                        else:
                            if controller not in \
                               propositional_attitudes['explicit_3']:
                                propositional_attitudes[
                                    'explicit_3'][controller] = []
                                if domHead not in \
                                   propositional_attitudes[
                                       'explicit_3'][controller]:
                                    propositional_attitudes[
                                        'explicit_3'][
                                            controller].append(domHead)
                            else:
                                if domHead not in \
                                   propositional_attitudes[
                                       'explicit_3'][controller]:
                                    propositional_attitudes[
                                        'explicit_3'][
                                            controller].append(
                                                domHead)

        elif (token.dep_ == 'amod'
              and token.head.dep_ in ['attr', 'oprd']
              and (token._.vwp_say
                   or token._.vwp_think
                   or token._.vwp_perceive
                   or token._.vwp_interpret
                   or token._.vwp_argue
                   or token._.vwp_argument
                   or token._.vwp_emotion
                   or (token._.transition
                       and token._.transition_category is not None
                       and token._.transition_category
                       not in ['temporal', 'PARAGRAPH'])
                   or token._.vwp_evaluation)
              and token._.governing_subject is not None
              and (hdoc[token._.governing_subject].dep_ == 'csubj'
                   or hdoc[token._.governing_subject].tag_ == '_SP')):

            controllers = ResolveReference(
                hdoc[token.head.head._.governing_subject], hdoc)
            for controller in controllers:
                domHead = self.propDomain(token.head.head, controller, hdoc)
                if (hdoc[token.head.head._.governing_subject].lemma_ == 'it'
                   and (token.head.head._.vwp_raising
                        or token.head.head._.vwp_tough
                        or token.head.head._.vwp_evaluation
                        or token.head.head.dep_ == 'xcomp')):
                    if token.head.head.pos_ == 'NOUN':
                        if is_definite_nominal(token.head.head):
                            if domHead not in \
                               propositional_attitudes['implicit_3']:
                                propositional_attitudes[
                                    'implicit_3'].append(domHead)
                        else:
                            if domHead not in \
                               propositional_attitudes['implicit']:
                                propositional_attitudes[
                                    'implicit'].append(domHead)
                    else:
                        if domHead not in propositional_attitudes['implicit']:
                            propositional_attitudes[
                                'implicit'].append(domHead)
                elif (hdoc[controller].text.lower()
                      in first_person_pronouns
                      and len(list(hdoc[controller].children)) == 0):
                    if domHead not in propositional_attitudes['explicit_1'] \
                       and not hdoc[controller]._.vwp_quoted:
                        propositional_attitudes['explicit_1'].append(domHead)
                elif (hdoc[controller].text.lower() in second_person_pronouns
                      and len(list(hdoc[controller].children)) == 0):
                    if domHead not in propositional_attitudes['explicit_2'] \
                       and not hdoc[controller]._.vwp_quoted:
                        propositional_attitudes['explicit_2'].append(domHead)
                else:
                    if hdoc[controller]._.vwp_say \
                       or hdoc[controller]._.vwp_think \
                       or hdoc[controller]._.vwp_perceive \
                       or hdoc[controller]._.vwp_interpret \
                       or hdoc[controller]._.vwp_argue \
                       or hdoc[controller]._.vwp_argument \
                       or hdoc[controller]._.vwp_emotion \
                       or (hdoc[controller]._.transition
                           and hdoc[controller]._.transition_category
                           is not None
                           and hdoc[controller]._.transition_category
                           not in ['temporal', 'PARAGRAPH']) \
                       or hdoc[controller]._.vwp_evaluation:
                        if is_definite_nominal(token.head.head):
                            if domHead not in \
                               propositional_attitudes['implicit_3']:
                                propositional_attitudes[
                                    'implicit_3'].append(domHead)
                        else:
                            if domHead not in \
                               propositional_attitudes['implicit']:
                                propositional_attitudes[
                                    'implicit'].append(domHead)
                    else:
                        if controller not in \
                           propositional_attitudes['explicit_3']:
                            propositional_attitudes[
                                'explicit_3'][controller] = []
                            if domHead not in propositional_attitudes[
                               'explicit_3'][controller]:
                                propositional_attitudes[
                                    'explicit_3'][controller].append(domHead)
                        else:
                            if domHead not in propositional_attitudes[
                               'explicit_3'][controller]:
                                propositional_attitudes[
                                    'explicit_3'][
                                        controller].append(domHead)
        elif (token.dep_ == 'acl'
              and token.head.head.lemma_ == 'for'
              and token.head.head.head.pos_ == 'ADJ'
              and token.head.head.head._.vwp_emotion
              and token.head.head.head._.governing_subject
              is not None):
            controllers = ResolveReference(
                hdoc[token.head.head.head._.governing_subject], hdoc)
            for controller in controllers:
                domHead = self.propDomain(
                    token.head.head.head, controller, hdoc)
                if (hdoc[
                    token.head.head.head._.governing_subject].lemma_ == 'it'
                    and (token.head.head.head._.vwp_raising
                         or token.head.head.head._.vwp_tough
                         or token.head.head.head._.vwp_evaluation
                         or token.head.head.head.dep_ == 'xcomp')):
                    if token.head.head.head.pos_ == 'NOUN':
                        if is_definite_nominal(token.head.head.head):
                            if domHead not in propositional_attitudes[
                               'implicit_3']:
                                propositional_attitudes[
                                    'implicit_3'].append(domHead)
                        else:
                            if domHead not in propositional_attitudes[
                               'implicit']:
                                propositional_attitudes[
                                    'implicit'].append(domHead)
                    else:
                        if domHead not in propositional_attitudes['implicit']:
                            propositional_attitudes['implicit'].append(domHead)
                elif (hdoc[controller].text.lower() in first_person_pronouns
                      and len(list(hdoc[controller].children)) == 0):
                    if domHead not in propositional_attitudes['explicit_1'] \
                       and not hdoc[controller]._.vwp_quoted:
                        propositional_attitudes['explicit_1'].append(domHead)
                elif (hdoc[controller].text.lower() in second_person_pronouns
                      and len(list(hdoc[controller].children)) == 0):
                    if domHead not in propositional_attitudes['explicit_2'] \
                       and not hdoc[controller]._.vwp_quoted:
                        propositional_attitudes['explicit_2'].append(domHead)
                else:
                    if hdoc[controller]._.vwp_say \
                       or hdoc[controller]._.vwp_think \
                       or hdoc[controller]._.vwp_perceive \
                       or hdoc[controller]._.vwp_interpret \
                       or hdoc[controller]._.vwp_argue \
                       or hdoc[controller]._.vwp_argument \
                       or hdoc[controller]._.vwp_emotion \
                       or (hdoc[controller]._.transition
                           and hdoc[controller]._.transition_category
                           is not None
                           and hdoc[controller]._.transition_category
                           not in ['temporal', 'PARAGRAPH']) \
                       or hdoc[controller]._.vwp_evaluation:
                        if is_definite_nominal(token.head.head.head):
                            if domHead not in \
                               propositional_attitudes['implicit_3']:
                                propositional_attitudes[
                                    'implicit_3'].append(domHead)
                        else:
                            if domHead not in \
                               propositional_attitudes['implicit']:
                                propositional_attitudes[
                                    'implicit'].append(domHead)
                    else:
                        if controller not in \
                           propositional_attitudes['explicit_3']:
                            propositional_attitudes[
                                'explicit_3'][controller] = []
                            if domHead not in \
                               propositional_attitudes[
                                   'explicit_3'][controller]:
                                propositional_attitudes[
                                    'explicit_3'][controller].append(domHead)
                        else:
                            if domHead not in propositional_attitudes[
                               'explicit_3'][controller]:
                                propositional_attitudes[
                                    'explicit_3'][
                                        controller].append(domHead)
        elif (token._.vwp_say
              or token._.vwp_think
              or token._.vwp_perceive
              or token._.vwp_interpret
              or token._.vwp_argue
              or token._.vwp_argument
              or token._.vwp_emotion
              or (token._.vwp_evaluation
                  and not token._.vwp_manner)
              or token.dep_ == 'neg'
              or token.lemma_ == 'no'
              or (token._.transition
                  and token._.transition_category is not None
                  and token._.transition_category
                  not in ['temporal', 'PARAGRAPH'])
              or token._.vwp_generalization
              or token._.vwp_illocutionary
              or token._.vwp_probability
              or token._.vwp_reservation
              or token._.vwp_emphasis):
            domHead = self.propDomain(token.head.head, None, hdoc)
            if token.dep_ in ['acomp', 'attr', 'oprd'] \
               and getSubject(token.head) is not None \
               and getSubject(token.head).dep_ == 'csubj':
                if token._.governing_subject is not None \
                   and hdoc[token._.governing_subject]._.animate:
                    domain = hdoc[token._.governing_subject].i
                    domHead = self.propDomain(token.head.head.head,
                                              domain,
                                              hdoc)
                    if hdoc[token._.governing_subject].text.lower() \
                       in first_person_pronouns:
                        if domHead not in propositional_attitudes[
                           'explicit_1']:
                            propositional_attitudes[
                                'explicit_1'].append(domHead)
                    elif (hdoc[token._.governing_subject].text.lower()
                          in second_person_pronouns):
                        if domHead not in propositional_attitudes[
                           'explicit_2']:
                            propositional_attitudes[
                                'explicit_2'].append(domHead)
                    else:
                        if domain not in propositional_attitudes['explicit_3']:
                            propositional_attitudes[
                                'explicit_3'][domain] = []
                        if domHead not in propositional_attitudes[
                           'explicit_3'][domain]:
                            propositional_attitudes[
                                'explicit_3'][domain].append(domHead)
                elif token._.governing_subject is not None:
                    domain = hdoc[token._.governing_subject].i
                    domHead = self.propDomain(token.head.head, domain, hdoc)
                    if domHead not in propositional_attitudes['implicit']:
                        propositional_attitudes['implicit'].append(domHead)
            if token.pos_ == 'ADV' \
               and token.head.pos_ in ['VERB', 'AUX'] \
               and (self.getHeadDomain(token).dep_ is None
                    or isRoot(self.getHeadDomain(token))
                    or self.getHeadDomain(token).head.tag_ == '_SP'):
                if domHead not in propositional_attitudes['implicit']:
                    propositional_attitudes['implicit'].append(domHead)
            elif (token.pos_ in 'ADV'
                  and token.head.dep_ in ['acomp', 'attr', 'oprd']
                  and token.head.head.pos_ in ['VERB', 'AUX']
                  and (self.getHeadDomain(token.head.head).dep_ is None
                       or isRoot(self.getHeadDomain(token.head.head)))):
                if domHead not in propositional_attitudes['implicit']:
                    propositional_attitudes['implicit'].append(domHead)

            elif (token.dep_ in ['xcomp',
                                 'ccomp',
                                 'csubj',
                                 'csubjpass',
                                 'advcl',
                                 'acl']
                  and token.head._.vwp_evaluation):
                if domHead not in propositional_attitudes['implicit']:
                    propositional_attitudes['implicit'].append(domHead)

            elif (token.dep_ == 'neg'
                  and (self.getHeadDomain(token).dep_ is None
                       or isRoot(self.getHeadDomain(token)))):
                if domHead not in propositional_attitudes['implicit']:
                    propositional_attitudes['implicit'].append(domHead)
            elif (token.dep_ == 'det'
                  and token.lemma_ == 'no'
                  and isRoot(self.getHeadDomain(token))):
                if domHead not in propositional_attitudes['implicit']:
                    propositional_attitudes['implicit'].append(domHead)
            elif (token.dep_ in ['aux', 'auxpass']
                  and token._.vwp_evaluation
                  and token.head.pos_ in ['VERB', 'AUX']
                  and isRoot(self.getHeadDomain(token))):
                if domHead not in propositional_attitudes['implicit']:
                    propositional_attitudes['implicit'].append(domHead)
            elif token._.governing_subject is not None:
                match = False
                for item in hdoc[token._.governing_subject].subtree:
                    if item.dep_ in ['ccomp', 'acl']:
                        match = True
                        break
                if match:
                    domHead = self.propDomain(token.head.head.head, None, hdoc)
                    if (hdoc[token._.governing_subject].lemma_ == 'it'
                        and (token._.vwp_raising
                             or token._.vwp_tough
                             or token._.vwp_evaluation
                             or token.dep_ == 'xcomp')):
                        if token.head.head.head.pos_ == 'NOUN':
                            if domHead not in \
                               propositional_attitudes['implicit_3']:
                                propositional_attitudes[
                                   'implicit_3'].append(domHead)
                            else:
                                if domHead not in \
                                   propositional_attitudes['implicit']:
                                    propositional_attitudes[
                                       'implicit'].append(domHead)
                        else:
                            if domHead not in \
                               propositional_attitudes['implicit']:
                                propositional_attitudes[
                                    'implicit'].append(domHead)
                    else:
                        if token.head.head.head._.governing_subject \
                           is not None:
                            controllers = ResolveReference(
                                hdoc[token.head.head.head._.governing_subject
                                     ], hdoc)
                            for controller in controllers:
                                if hdoc[controller].text.lower() \
                                   in first_person_pronouns \
                                   and len(list(
                                      hdoc[controller].children)) == 0:
                                    if domHead not in \
                                       propositional_attitudes['explicit_1'] \
                                       and not hdoc[controller]._.vwp_quoted:
                                        propositional_attitudes[
                                            'explicit_1'].append(domHead)
                                elif (hdoc[controller].text.lower()
                                      in second_person_pronouns
                                      and len(list(
                                          hdoc[controller].children)) == 0):
                                    if domHead not in \
                                        propositional_attitudes['explicit_2'] \
                                       and not hdoc[controller]._.vwp_quoted:
                                        propositional_attitudes[
                                            'explicit_2'].append(domHead)
                                else:
                                    if domHead not in \
                                       propositional_attitudes['implicit_3']:
                                        propositional_attitudes[
                                            'implicit_3'].append(domHead)
        if is_definite_nominal(token) \
           and isRoot(self.getHeadDomain(token)) \
           and (token._.vwp_say
                or token._.vwp_think
                or token._.vwp_perceive
                or token._.vwp_interpret
                or token._.vwp_argue
                or token._.vwp_argument
                or token._.vwp_emotion
                or (token._.transition
                    and token._.transition_category is not None
                    and token._.transition_category
                    not in ['temporal', 'PARAGRAPH'])
                or token._.vwp_evaluation):
            rt = self.propDomain(token.head.head, None, hdoc)
            if tensed_clause(self.getHeadDomain(token)
               or self.getHeadDomain(token)._.pos_ != 'VERB'):
                if rt not in propositional_attitudes['implicit_3'] \
                   and rt not in propositional_attitudes['implicit']:
                    propositional_attitudes['implicit_3'].append(rt)
            else:
                if rt not in propositional_attitudes['explicit_2'] \
                   and rt not in propositional_attitudes['explicit_2']:
                    propositional_attitudes['explicit_2'].append(rt)

        return propositional_attitudes

    def theory_of_mind_sentences(self, token, hdoc, theory_of_mind_sentences):
        if token._.vwp_cognitive or token._.vwp_communication \
           or token._.vwp_emotion or token._.vwp_emotional_impact:
            for child in token.children:
                if ('ccomp' == child.dep_
                   or 'csubj' == child.dep_
                   or 'acl' == child.dep_
                   or 'relcl' == child.dep_
                   or 'advcl' == child.dep_
                   or 'oprd' == child.dep_
                   or 'xcomp' == child.dep_
                   or 'acomp' == child.dep_
                   or 'prep' == child.dep_) \
                   and not child._.vwp_quoted:
                    controllers = ResolveReference(
                        hdoc[token._.governing_subject], hdoc)
                    for controller in controllers:
                        agent = hdoc[controller]
                        if child._.has_governing_subject:
                            childSubj = hdoc[child._.governing_subject]
                            childControllers = ResolveReference(
                                hdoc[child._.governing_subject], hdoc)
                            if agent._.animate \
                               and childControllers is not None \
                               and controller not in childControllers \
                               and childSubj._.animate \
                               and not (agent.text.lower()
                                        in first_person_pronouns
                                        and childSubj.text.lower()
                                        in first_person_pronouns) \
                               and not (agent.text.lower()
                                        in second_person_pronouns
                                        and childSubj.text.lower()
                                        in second_person_pronouns):

                                entry = \
                                    newSpanEntry('social awareness',
                                                 token.sent.start,
                                                 token.sent.end,
                                                 hdoc,
                                                 'theory of mind sentence')
                                if entry not in theory_of_mind_sentences:
                                    theory_of_mind_sentences.append(
                                        entry)
                                break
                    break
        return theory_of_mind_sentences

    def negativePredicate(self, item: Token):
        """
         This function identifies lexical predicates that
         function as equivalent to negation when combined
         with other elements. This list may not be complete
         -- to double check later.
        """
        if item.lemma_ in ['lack',
                           'fail',
                           'failure',
                           'absence',
                           'shortage',
                           'false',
                           'wrong',
                           'inaccurate',
                           'incorrect']:
            return True
        return False

    def findClause(self, token: Token):
        if token.tag_.startswith('V') \
           or isRoot(token):
            return token
        return self.findClause(token.head)

    def getFirstChild(self, token: Token):
        for child in token.children:
            return child
        else:
            return None

    def propagateNegation(self, doc: Doc):
        """
         Set baseline sentiment on lexical elements.
         Identify negation elements in the parse tree (to use
         to reverse sentiment polarities where negated)
        """
        negation_tokens = []
        for tok in doc:
            tok._.vwp_sentiment = tok._.sentiword
            # we combine data from the sentiword and polarity ratings
            # to come up with a more reliable estimate of tone, positive
            # or negative. We trust those estimates a lot more for words
            # marked as subjective (vwp_evaluation, vwp_hege, or in assessments
            # from spacytextblob). In that case, we take the absolutely
            # larger or the sentiword or polarity rating. Otherwise, we
            # take the minimum, because sentiword in particular seems to
            # be biased toward positive ratings for words that are actually
            # neutral.
            if tok._.vwp_evaluation \
               or tok._.vwp_hedge \
               or tok.text in doc._.assessments:
                if tok._.polarity < 0 or tok._.sentiword < 0:
                    tok._.vwp_tone = min(tok._.polarity, tok._.sentiword)
                elif tok._.polarity > 0 and tok._.sentiword > 0:
                    tok._.vwp_tone = max(tok._.polarity, tok._.sentiword)
                else:
                    tok._.vwp_tone = (tok._.polarity + tok._.sentiword) / 2
            else:
                tok._.vwp_tone = min(tok._.polarity, tok._.sentiword)

            # rule order fixes to the tone variable are generally a bad idea,
            # but these are so common that fixing them gets rid of a lot of
            # distraction when displaying positive and negative tone words
            # able certain pretty kind mean fun
            if tok.text.lower() in ['able', 'ready'] \
               and tok.i + 1 < len(doc) \
               and tok.nbor(1) is not None \
               and tok.nbor(1).text.lower() == 'to':
                tok._.vwp_tone = 0.0
            elif (tok.text.lower() == 'fun'
                  and tok.i + 1 < len(doc)
                  and tok.nbor(1) is not None
                  and tok.nbor(1).text.lower() == 'of'):
                tok._.vwp_tone = -1*tok._.vwp_tone
                tok.nbor(1)._.vwp_tone = tok._.vwp_tone
            elif tok.text.lower() == 'certain' and tok.i < tok.head.i:
                tok._.vwp_tone = 0.0
            elif tok.text.lower() == 'pretty' and tok.pos_ == 'ADV':
                tok._.vwp_tone = 0.0
            elif tok.text.lower() in ['kind', 'right'] and tok.pos_ == 'NOUN':
                tok._.vwp_tone = 0.0
            elif tok.text.lower() == 'mean' and tok.pos_ == 'VERB':
                tok._.vwp_tone = 0.0

            penultimate = None
            antepenultimate = None
            if tok.dep_ == 'neg' \
               or self.negativePredicate(tok) \
               or (tok.dep_ == 'preconj'
                   and tok.text.lower() == 'neither') \
               or tok.text.lower() == 'hardly' \
               or tok.text.lower() == 'no' \
               or (antepenultimate is not None
                   and antepenultimate.text.lower() == 'less'
                   and penultimate is not None
                   and penultimate.text.lower() == 'than'):
                newTk = self.findClause(tok)
                if newTk not in negation_tokens:
                    negation_tokens.append(newTk)
                else:
                    negation_tokens.remove(newTk)

            if (antepenultimate is not None
                and antepenultimate.text.lower() == 'less'
                or antepenultimate is not None
                and antepenultimate.text.lower() == 'more') \
               and penultimate is not None \
               and penultimate.text.lower() == 'than':
                antepenultimate.norm_ = tok.norm_

            antepenultimate = penultimate
            penultimate = tok
        return negation_tokens

    def traverseTree(self, token: Token, negation_tokens: list):
        """
         Traverse tree and call function to reverse sentiment polarity
         when negated
        """

        pos_degree_mod = ['totally',
                          'completely',
                          'perfectly',
                          'eminently',
                          'fairly',
                          'terrifically',
                          'amazingly',
                          'astonishingly',
                          'breathtakingly',
                          'genuinely',
                          'truly',
                          'sublimely',
                          'marvelously',
                          'fantastically',
                          'pretty',
                          'wondrously',
                          'exquisitely',
                          'thoroughly',
                          'profoundly',
                          'incredibly',
                          'greatly',
                          'extraordinarily',
                          'stunningly',
                          'incomparably',
                          'greatly',
                          'incomparable',
                          'stunning',
                          'great',
                          'greater',
                          'greatest',
                          'complete',
                          'ultimate',
                          'absolute',
                          'incredible',
                          'total',
                          'extraordinary',
                          'perfect',
                          'terrific',
                          'astonishing',
                          'breathtaking',
                          'genuine',
                          'true',
                          'sublime',
                          'fantastic',
                          'marvelous',
                          'wonderful',
                          'thorough',
                          'profound',
                          'unsurpassed',
                          'phenomenal',
                          'inestimable',
                          'prodigious',
                          'monumental',
                          'monumentally',
                          'tremendous',
                          'tremendously',
                          'prodigiously',
                          'inestimably',
                          'phenomenally']

        if token.pos_ in ['ADJ', 'ADV'] \
           and token.text.lower() in pos_degree_mod \
           and token._.vwp_tone is not None \
           and token.head._.vwp_tone is not None \
           and token._.vwp_tone > 0 \
           and token.head._.vwp_tone < 0:
            token._.vwp_tone = -1 * token._.vwp_tone

        # negation takes scope over everything it governs, i.e., its
        # children and its sister nodes and their children
        # so if we have a negation, we need to apply it not just to
        # the node and its children, but to all children of the node
        # that is that child's head
        if token in negation_tokens:
            if token is not None:
                neg = self.getFirstChild(token)
                self.spread_reverse_polarity(token)
                negation_tokens.remove(token)

        # once we have handled negation at this scope level,
        # handle any embedded negations
        if len(negation_tokens) > 0:
            for child in token.children:
                if child != token:
                    negation_tokens = \
                        self.traverseTree(child, negation_tokens)

        return negation_tokens

    def spread_reverse_polarity(self, tok: Token):
        """
         This function traverses the children of a node and marks
         # them all as negated, except for most right children of
         # nouns, which need to be treated as not affected
         by negation.
        """

        # negate the node
        tok._.vwp_sentiment = -1 * tok._.vwp_sentiment

        lastChild = None

        # now traverse the children of the node
        for child in tok.children:

            if child != tok and not tensed_clause(child):

                # and recurse over children of the children
                self.spread_reverse_polarity(child)
                lastChild = child

    def spoken_register_markers(self, doc: Doc):
        # TBD: develop a parallel marker of formal/written
        # register
        for token in doc:

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
                            if token.text.lower() not in ['think']:
                                token._.usage = dom.lemma_names()[0]
                        break
                    break
            except Exception as e:
                print('No Wordnet synset found for ', token, e)

            # Use of first person pronouns
            if token.text.lower() in first_person_pronouns:
                token._.vwp_interactive = True

            # Use of second person pronouns
            elif token.text.lower() in second_person_pronouns:
                token._.vwp_interactive = True

            # Use of wh-questions
            elif (token.dep_ in ['WP', 'WRB']
                  and token.head.dep_ not in
                  ['relcl',
                   'ccomp',
                   'csubj',
                   'csubjpass',
                   'xcomp',
                   'acl']):
                token._.vwp_interactive = True
            elif (token.dep_ == 'WP$'
                  and token.head.head.dep_ not in
                  ['relcl',
                   'ccomp',
                   'csubj',
                   'csubjpass',
                   'xcomp',
                   'acl']):
                token._.vwp_interactive = True

            # Use of contractions
            elif (token.text in
                  ['\'s',
                   '\'ve',
                   '\'d',
                   '\'ll',
                   '\'m',
                   '\'re',
                   'n\'t',
                   '\'cause',
                   'gotta',
                   'oughta',
                   'sposeta',
                   'gonna',
                   'couldnt',
                   'shouldnt',
                   'wouldnt',
                   'mightnt',
                   'woulda',
                   'didnt',
                   'doesnt',
                   'dont',
                   'werent',
                   'wasnt',
                   'aint',
                   'sposta',
                   'sposeta',
                   'ain\t',
                   'cain\'t']):
                token._.vwp_interactive = True

            elif (token.pos_ == 'PRON'
                  and token.i + 1 < len(doc)
                  and doc[token.i + 1].text in ['\'s',
                                                '\'re',
                                                '\'d',
                                                '\'m',
                                                '\'ll',
                                                '\'ve']):
                token._.vwp_interactive = True

            # Preposition stranding
            elif (token.dep_ == 'IN'
                  and 'pobj' not in [child.dep_
                                     for child in token.children]
                  and 'prep' not in [child.dep_
                                     for child in token.children]):
                token._.vwp_interactive = True

            # Anaphoric use of auxiliaries
            elif (token.pos == 'AUX'
                  and 'VERB' not in [child.pos_ for child
                                     in token.head.children]):
                token._.vwp_interactive = True

            # pronominal contractives
            elif ((token.lemma_ in ['be',
                                    'have',
                                    'do',
                                    'ai',
                                    'wo',
                                    'sha']
                  or token.pos_ == 'AUX')
                  and token.i+1 < len(doc)
                  and doc[token.i + 1].pos_ == 'PART'
                  and doc[token.i + 1].text.lower() != 'not'
                  and len([child for child in token.children]) == 0):
                token._.vwp_interactive = True

            # Use of demonstrative articles and pronouns
            elif (token.text.lower() in ['here',
                                         'there',
                                         'this',
                                         'that',
                                         'these',
                                         'those']
                  and token.dep_ != 'mark'
                  and token.tag_ != 'WDT'
                  and token.pos_ in ['PRON', 'DET']):
                token._.vwp_interactive = True

            # Use of indefinite pronouns
            elif (token.text.lower() in ['anyone',
                                         'anybody',
                                         'anything',
                                         'someone',
                                         'somebody',
                                         'something',
                                         'nobody',
                                         'nothing',
                                         'everyone',
                                         'everybody',
                                         'everything']
                  and token.pos_ == 'PRON'):
                token._.vwp_interactive = True

            # Use of a conjunction to start a main clause
            elif token.is_sent_start and token.tag_ == 'CCONJ':
                token._.vwp_interactive = True

            # Use of common verbs of saying with a personal pronoun subject
            elif ((token.tag_ == 'PRP' or token.pos_ == 'PROPN')
                  and token.head.lemma_ in ['say',
                                            'see',
                                            'look',
                                            'tell',
                                            'give',
                                            'ask',
                                            'remind',
                                            'warn',
                                            'promise',
                                            'order',
                                            'command',
                                            'admit',
                                            'confess',
                                            'beg',
                                            'suggest',
                                            'recommend',
                                            'advise',
                                            'command',
                                            'declare',
                                            'forbid',
                                            'refuse',
                                            'thank',
                                            'congratulate',
                                            'praise',
                                            'forgive',
                                            'pardon']):
                token._.vwp_interactive = True

            # Use of common emphatic adverbs
            elif (token.dep_ == 'advmod'
                  and token.head.pos_ in ['ADJ', 'ADV', 'VERB']
                  and token.lemma_ in ['absolutely',
                                       'altogether',
                                       'completely',
                                       'enormously',
                                       'entirely',
                                       'awfully',
                                       'extremely',
                                       'fully',
                                       'greatly',
                                       'highly',
                                       'hugely',
                                       'intensely',
                                       'perfectly',
                                       'strongly',
                                       'thoroughly',
                                       'totally',
                                       'utterly',
                                       'very',
                                       'so',
                                       'too',
                                       'mainly',
                                       'pretty',
                                       'totally',
                                       'even']):
                token._.vwp_interactive = True
                token._.vwp_evaluation = True

            # Use of common emphatic adverbs
            elif (token.dep_ == 'amod'
                  and token.head.pos_ in ['NOUN']
                  and token.lemma_ in ['huge'
                                       'gigantic',
                                       'enormous',
                                       'total',
                                       'complete'
                                       'extreme',
                                       'thorough',
                                       'perfect'
                                       ]):
                token._.vwp_interactive = True
                token._.vwp_evaluation = True

            # Use of verb general evaluation adjectives
            elif (token.pos_ == 'ADJ'
                  and token.lemma_ in ['bad',
                                       'good',
                                       'better',
                                       'best',
                                       'grand',
                                       'happy',
                                       'crazy',
                                       'huge',
                                       'neat',
                                       'nice',
                                       'sick',
                                       'smart',
                                       'strange',
                                       'stupid',
                                       'weird',
                                       'wrong',
                                       'new',
                                       'awful',
                                       'mad',
                                       'funny',
                                       'glad']):
                # Use of these kinds of expressions
                # is also evaluative
                token._.vwp_interactive = True

            # Use of common hedge words and phrases
            elif (token.pos_ == 'ADV'
                  and token.lemma_ in ['just',
                                       'really',
                                       'mostly',
                                       'so',
                                       'actually',
                                       'basically',
                                       'probably',
                                       'awhile',
                                       'almost',
                                       'maybe',
                                       'still',
                                       'kinda',
                                       'kind',
                                       'sorta',
                                       'sort',
                                       'mostly',
                                       'more']):
                token._.vwp_interactive = True
            # Expressions like think so, think not
            elif (token.pos_ == 'VERB'
                  and (token._.vwp_argument
                       or token._.vwp_communication
                       or token._.vwp_cognitive)
                  and token.i + 1 < len(token.doc)
                  and token.nbor().text.lower() in ['so', 'not']):
                token._.vwp_interactive = True
                token.nbor()._.vwp_interactive = True
                # Use of this kind of elliptical construction
                # is also evaluative
                token.nbor()._.vwp_evaluation = True
            elif (token.lemma_ in ['lot', 'bit', 'while', 'ways']
                  and token.dep_ == 'npadvmod'):
                token._.vwp_interactive = True
            elif token.pos_ == 'DET' and token.dep_ == 'advmod':
                # expressions like 'all in all'
                token._.vwp_interactive = True
            elif token.dep_ == 'predet':
                # predeterminers like 'such a', 'what', or 'quite'
                token._.vwp_interactive = True
            elif (token.lemma_ == 'like' and token.dep_ == 'prep'
                  and token.head.lemma_ in ['something',
                                            'stuff',
                                            'thing']):
                # expressions like 'stuff like that'
                token._.vwp_interactive = True
            elif (token.dep_ == 'amod'
                  and token.head.pos_ == 'NOUN'
                  and token.head.dep_ in ['attr', 'ccomp']
                  and token.lemma_ in ['real',
                                       'absolute',
                                       'complete',
                                       'perfect',
                                       'total',
                                       'utter']):
                # expressions like 'a complete idiot'
                token._.vwp_interactive = True
            elif (token.lemma_ == 'old'
                  and self.getFirstChild(token.head) is not None
                  and 'any' == self.getFirstChild(token.head).lemma_):
                token._.vwp_interactive = True
            elif (token.lemma_ in ['bunch', 'couple', 'lot']
                  and self.getFirstChild(token) is not None
                  and 'of' == self.getFirstChild(token).lemma_):
                token._.vwp_interactive = True

            # Use of common private mental state verbs with
            # first or second person pronouns
            elif (token.text.lower() in ['i', 'you']
                  and token.dep_ == 'nsubj'
                  and token.head.lemma_ in ['assume',
                                            'believe',
                                            'beleive',
                                            'bet',
                                            'care',
                                            'consider',
                                            'dislike',
                                            'doubt',
                                            'expect',
                                            'fear',
                                            'feel',
                                            'figure',
                                            'forget',
                                            'gather',
                                            'guess',
                                            'hate',
                                            'hear',
                                            'hope',
                                            'imagine',
                                            'judge',
                                            'know',
                                            'like',
                                            'look',
                                            'love',
                                            'mean',
                                            'notice',
                                            'plan',
                                            'realize',
                                            'recall',
                                            'reckon',
                                            'recognize',
                                            'remember',
                                            'see',
                                            'sense',
                                            'sound',
                                            'suppose',
                                            'suspect',
                                            'think',
                                            'understand',
                                            'want',
                                            'wish',
                                            'wonder']):
                token._.vwp_interactive = True
                token.head._.vwp_interactive = True

            # Use of words with a strong conversational flavor
            elif token.lemma_ in ['guy',
                                  'gal',
                                  'kid',
                                  'plus',
                                  'stuff',
                                  'thing',
                                  'because',
                                  'blah',
                                  'etcetera',
                                  'alright',
                                  'hopefully',
                                  'personally',
                                  'anyhow',
                                  'anyway',
                                  'anyways',
                                  'cuss',
                                  'coulda',
                                  'woulda',
                                  'doncha',
                                  'betcha']:
                token._.vwp_interactive = True
            # Use of interjections
            elif token.pos_ == 'INTJ':
                token._.vwp_interactive = True
                token._.vwp_evaluation = True
            elif (token.text.lower == 'thank'
                  and doc[token.i + 1].text.lower == 'you'
                  and doc[token.i+2].pos_ == 'PUNCT'

                  or token.text.lower == 'pardon'
                  and doc[token.i + 1].text.lower == 'me'
                  and doc[token.i + 2].pos_ == 'PUNCT'

                  or token.text.lower == 'after'
                  and doc[token.i + 1].text.lower == 'you'
                  and doc[token.i + 2].pos_ == 'PUNCT'

                  or token.text.lower == 'never'
                  and doc[token.i + 1].text.lower == 'mind'
                  and doc[token.i + 2].pos_ == 'PUNCT'

                  or token.text.lower == 'speak'
                  and doc[token.i + 1].text.lower == 'soon'
                  and doc[token.i + 2].pos_ == 'PUNCT'

                  or token.text.lower == 'know'
                  and doc[token.i + 1].text.lower == 'better'
                  or token.text.lower == 'shut'
                  and doc[token.i + 1].text.lower == 'up'

                  or token.text.lower == 'you'
                  and doc[token.i + 1].text.lower == 'wish'
                  and doc[token.i + 2].pos_ == 'PUNCT'

                  or token.text.lower == 'take'
                  and doc[token.i + 1].text.lower == 'care'
                  and doc[token.i + 2].pos_ == 'PUNCT'
                  ):
                token._.vwp_interactive = True
                if token.i + 1 < len(doc):
                    doc[token.i+1]._.vwp_interactive = True

            # Use of idiomatic prep + adj combinations like
            # for sure, for certain, for good
            elif token.pos_ == 'ADJ' and token.head.pos_ == 'ADP':
                token._.vwp_interactive = True

    def nominalReferences(self, doc):
        """
        A very simple listing of potential entities. No proper
        resolution of nominal reference. If the two different
        entities are referred to by the same noun, or two different
        nouns used for the same entity, we don't attempt to resolve
        them. TO-DO: a more proper reference resolver that handles
        nouns, instead of just the anaphora resolution handled by
        coreferee.
        """

        characterList = {}
        referenceList = {}
        registered = []
        for token in doc:
            if token._.animate or token.text.lower() in characterList:
                if token.pos_ == 'PROPN' \
                   and token.ent_type_ not in ['ORG', 'DATE']:
                    if token.text.capitalize() in characterList:
                        if token.i not in characterList[
                           token.text.capitalize()]:
                            characterList[
                                token.text.capitalize()].append(token.i)
                            registered.append(token.i)
                    elif token.orth_.capitalize() not in characterList:
                        characterList[token.orth_.capitalize()] = [token.i]
                        registered.append(token.i)
                    else:
                        if token.i not in characterList[
                           token.orth_.capitalize()]:
                            characterList[
                                token.orth_.capitalize()].append(token.i)
                            registered.append(token.i)
                elif token.pos_ == 'NOUN':
                    # Not animate if a noun! Pronoun/noun confusion case
                    if token.text.lower() == 'mine':
                        continue
                    if token._.root is not None \
                       and token._.root.capitalize() not in characterList:
                        characterList[token._.root.capitalize()] = [token.i]
                        registered.append(token.i)
                    elif token.text.capitalize() not in characterList:
                        characterList[token.text.capitalize()] = [token.i]
                        registered.append(token.i)
                    else:
                        if token.i not in characterList[
                           token.text.capitalize()]:
                            characterList[
                                token.text.capitalize()].append(token.i)
                            registered.append(token.i)
                elif token.pos_ == 'PRON':
                    antecedent = [doc[loc]
                                  for loc in ResolveReference(token, doc)]
                    if len(antecedent) > 0 and antecedent[0].i == token.i:
                        if not token._.vwp_quoted \
                           and token.text.lower() in ['i',
                                                      'me',
                                                      'my',
                                                      'mine',
                                                      'myself',
                                                      'we',
                                                      'us',
                                                      'our',
                                                      'ours',
                                                      'ourselves']:
                            if 'SELF' not in characterList:
                                characterList['SELF'] = [token.i]
                                registered.append(token.i)
                            else:
                                if token.i not in characterList['SELF']:
                                    characterList['SELF'].append(token.i)
                                    registered.append(token.i)
                        elif (not token._.vwp_quoted
                              and token.text.lower() in ['you',
                                                         'your',
                                                         'yours',
                                                         'yourselves']):
                            if 'You' not in characterList:
                                characterList['You'] = [token.i]
                                registered.append(token.i)
                            else:
                                if token.i not in characterList['You']:
                                    characterList['You'].append(token.i)
                                    registered.append(token.i)
                        else:
                            if not token._.vwp_quoted:
                                if token.text.lower() in ['they',
                                                          'them',
                                                          'their',
                                                          'theirs']:
                                    antecedents = \
                                        scanForAnimatePotentialAntecedents(
                                            doc, token.i, [])
                                    charname = ''
                                    lems = []
                                    for i, ref in enumerate(antecedents):
                                        if doc[ref].text not in lems:
                                            lems.append(
                                                doc[ref].text.capitalize())
                                        if i == 0:
                                            charname = \
                                                doc[ref].text.capitalize()
                                        elif i+1 == len(antecedents):
                                            charname += ' and ' \
                                                + doc[ref].text.capitalize()
                                        else:
                                            charname += ', ' \
                                                + doc[ref].text.capitalize()
                                    if charname not in characterList:
                                        characterList[charname] = []
                                    characterList[charname].append(token.i)
                                    registered.append(token.i)
                    elif (token.text.lower() in ['they',
                                                 'them',
                                                 'their',
                                                 'theirs',
                                                 'themselves']
                          and len(antecedent) > 1):
                        charname = ''
                        lems = []
                        for i, ref in enumerate(antecedent):
                            if ref.text not in lems:
                                lems.append(ref.text.capitalize())
                                if i == 0:
                                    charname = ref.text.capitalize()
                                elif i+1 == len(antecedent):
                                    charname += ' and ' + ref.text.capitalize()
                                else:
                                    charname += ', ' + ref.text.capitalize()
                        if charname not in characterList:
                            characterList[charname] = []
                        characterList[charname].append(token.i)
                        registered.append(token.i)

                    elif (len(antecedent) > 0
                          and antecedent[0].lemma_.capitalize
                          in characterList):
                        if token.i not in characterList[antecedent[0].lemma_]:
                            characterList[
                                antecedent[0].lemma_.capitalize()
                                ].append(token.i)
                            registered.append(token.i)
                    elif (len(antecedent) > 0
                          and antecedent[0].text.capitalize()
                          in characterList):
                        if token.i not in characterList[
                           antecedent[0].text.capitalize()
                           ]:
                            characterList[
                                antecedent[0].text.capitalize()
                                ].append(token.i)
                            registered.append(token.i)
                    elif len(antecedent) > 0 \
                        and antecedent[0].orth_.capitalize() \
                            in characterList:
                        if token.i not in characterList[
                           antecedent[0].orth_.capitalize()]:
                            characterList[
                                antecedent[0].orth_.capitalize()
                                ].append(token.i)
                            registered.append(token.i)
                    elif len(antecedent) > 0:
                        characterList[
                            antecedent[0].lemma_.capitalize()
                            ] = [token.i]
                        registered.append(token.i)
            else:
                if token.pos_ == 'PROPN' \
                    and token.ent_type_ not in ['ORG',
                                                'DATE']:
                    if token.text.capitalize() in characterList:
                        if token.i not in characterList[
                           token.text.capitalize()]:
                            characterList[
                                token.text.capitalize()].append(token.i)
                            registered.append(token.i)
                    elif token.orth_.capitalize() not in characterList:
                        characterList[token.orth_.capitalize()] = [token.i]
                        registered.append(token.i)
                    else:
                        if token.i not in characterList[
                           token.orth_.capitalize()]:
                            characterList[
                                token.orth_.capitalize()].append(token.i)
                            registered.append(token.i)
                elif token.pos_ == 'NOUN':
                    if token._.root is not None \
                       and token._.root.capitalize() not in referenceList:
                        referenceList[token._.root.capitalize()] = [token.i]
                        registered.append(token.i)
                    elif token.text.capitalize() not in referenceList:
                        referenceList[token.text.capitalize()] = [token.i]
                        registered.append(token.i)
                    else:
                        if token.i not in referenceList[
                           token.text.capitalize()]:
                            referenceList[
                                token.text.capitalize()].append(token.i)
                            registered.append(token.i)
                elif token.pos_ == 'PRON':
                    antecedent = [doc[loc] for loc
                                  in ResolveReference(token, doc)]
                    if antecedent[0].i == token.i:
                        if not token._.vwp_quoted \
                           and token.lemma_ in ['i',
                                                'I',
                                                'me',
                                                'My',
                                                'my',
                                                'mine',
                                                'myself',
                                                'We',
                                                'we',
                                                'us',
                                                'Our',
                                                'our',
                                                'ours',
                                                'ourselves']:
                            if 'SELF' not in characterList:
                                characterList['SELF'] = [token.i]
                                registered.append(token.i)
                            else:
                                if token.i not in characterList['SELF']:
                                    characterList['SELF'].append(token.i)
                                    registered.append(token.i)
                        elif not token._.vwp_quoted \
                            and token.lemma_ in ['You',
                                                 'you',
                                                 'Your',
                                                 'your',
                                                 'yours',
                                                 'yourselves']:
                            if 'You' not in characterList:
                                characterList['You'] = [token.i]
                                registered.append(token.i)
                            else:
                                if token.i not in characterList['You']:
                                    characterList['You'].append(token.i)
                                    registered.append(
                                        token.i)
                        else:
                            if not token._.vwp_quoted:
                                if token.text.lower() in ['they',
                                                          'them',
                                                          'their',
                                                          'theirs']:
                                    antecedents = \
                                        scanForAnimatePotentialAntecedents(
                                            doc, token.i, [])
                                    charname = ''
                                    lems = []
                                    for i, ref in enumerate(antecedents):
                                        if doc[ref].text not in lems:
                                            lems.append(
                                                doc[ref].text.capitalize())
                                        if i == 0:
                                            charname = \
                                                doc[ref].text.capitalize()
                                        elif i+1 == len(antecedents):
                                            charname += ' and ' \
                                                + doc[ref].text.capitalize()
                                        else:
                                            charname += ', ' \
                                                + doc[ref].text.capitalize()
                                    if charname not in characterList:
                                        characterList[charname] = []
                                    characterList[charname].append(token.i)
                                    registered.append(token.i)
                    elif (token.text.lower() in ['they',
                                                 'them',
                                                 'their',
                                                 'theirs',
                                                 'themselves']
                          and len(antecedent) > 1):
                        charname = ''
                        lems = []
                        for i, ref in enumerate(antecedent):
                            if ref.text not in lems:
                                lems.append(ref.text.capitalize())
                                if i == 0:
                                    charname = ref.text.capitalize()
                                elif i+1 == len(antecedent):
                                    charname += ' and ' + ref.text.capitalize()
                                else:
                                    charname += ', ' + ref.text.capitalize()
                        if charname not in referenceList:
                            referenceList[charname] = []
                        referenceList[charname].append(token.i)
                        registered.append(token.i)
                    elif antecedent[0].i not in registered:
                        if antecedent[0].lemma_.capitalize() in referenceList:
                            if token.i not in referenceList[
                               antecedent[0].lemma_.capitalize()]:
                                referenceList[
                                              antecedent[0].lemma_.capitalize()
                                              ].append(token.i)
                                registered.append(token.i)
                        elif antecedent[0].text.capitalize() in referenceList:
                            if token.i not in referenceList[
                               antecedent[0].text.capitalize()]:
                                referenceList[
                                              antecedent[0].text.capitalize()
                                              ].append(token.i)
                                registered.append(token.i)
                        elif antecedent[0].orth_.capitalize() in referenceList:
                            if token.i not in referenceList[
                               antecedent[0].orth_.capitalize()]:
                                referenceList[
                                              antecedent[0].orth_.capitalize()
                                              ].append(token.i)
                                registered.append(token.i)
                        elif len(antecedent) > 0:
                            referenceList[
                                antecedent[0].lemma_.capitalize()] = [token.i]
                            registered.append(token.i)
        directspeech = doc._.vwp_direct_speech_spans
        for character in characterList:
            for speechevent in directspeech:
                speaker = speechevent[0]
                addressee = speechevent[1]
                for item in speaker:
                    if item in characterList[character]:
                        for item2 in speaker:
                            if item2 not in characterList[character]:
                                characterList[character].append(item2)
                for item in addressee:
                    if item in characterList[character]:
                        for item2 in addressee:
                            if item2 not in characterList[character]:
                                characterList[character].append(item2)
        return (characterList, referenceList)

    def tenseSequences(self, document):
        tenseChanges = []
        currentEvent = {}
        i = 0
        past_tense_state = False
        while i < len(document):
            if i > 0 \
               and not in_past_tense_scope(
                   getTensedVerbHead(document[i])) \
               and not document[i].tag_ in ['CC',
                                            '_SP',
                                            'para',
                                            'parapara'] \
               and not document[i-1].tag_ in ['CC',
                                              '_SP',
                                              'para',
                                              'parapara'] \
               and in_past_tense_scope(getTensedVerbHead(document[i-1])):
                if not document[i]._.vwp_in_direct_speech:
                    if past_tense_state:
                        currentEvent['loc'] = i
                        currentEvent['past'] = False
                        tenseChanges.append(currentEvent)
                        currentEvent = {}
                        past_tense_state = False
            elif (i > 1
                  and not in_past_tense_scope(
                      getTensedVerbHead(document[i]))
                  and not document[i].tag_ in ['CC',
                                               '_SP',
                                               'para',
                                               'parapara']
                  and document[i-1].tag_ in ['_SP',
                                             'para',
                                             'parapara']
                  and not document[i-2].tag_ in ['CC',
                                                 '_SP',
                                                 'para',
                                                 'parapara']
                  and in_past_tense_scope(getTensedVerbHead(document[i-2]))):
                if not document[i]._.vwp_in_direct_speech:
                    if past_tense_state:
                        currentEvent['loc'] = i
                        currentEvent['past'] = False
                        tenseChanges.append(currentEvent)
                        currentEvent = {}
                        past_tense_state = False

            elif (i == 0
                  and in_past_tense_scope(getTensedVerbHead(document[i]))):
                if not document[i]._.vwp_in_direct_speech:
                    past_tense_state = True
                    currentEvent['loc'] = i
                    currentEvent['past'] = True
                    tenseChanges.append(currentEvent)
                    currentEvent = {}

            elif (i > 0
                  and not document[i-1].tag_ in ['CC',
                                                 '_SP',
                                                 'para',
                                                 'parapara']
                  and not in_past_tense_scope(
                      getTensedVerbHead(document[i-1]))
                  and not document[i].tag_ in ['CC',
                                               '_SP',
                                               'para',
                                               'parapara']
                  and in_past_tense_scope(
                      getTensedVerbHead(document[i]))):
                if not document[i]._.vwp_in_direct_speech:
                    if not past_tense_state:
                        currentEvent['loc'] = i
                        currentEvent['past'] = True
                        tenseChanges.append(currentEvent)
                        currentEvent = {}
                        past_tense_state = True
            elif (i > 1
                  and not document[i-2].tag_ in ['CC',
                                                 '_SP',
                                                 'para',
                                                 'parapara']
                  and not in_past_tense_scope(
                      getTensedVerbHead(document[i-2]))
                  and document[i-1].tag_ in ['_SP',
                                             'para',
                                             'parapara']
                  and not document[i].tag_ in ['CC',
                                               '_SP',
                                               'para',
                                               'parapara']
                  and in_past_tense_scope(
                      getTensedVerbHead(document[i]))):
                if not document[i]._.vwp_in_direct_speech:
                    if not past_tense_state:
                        currentEvent['loc'] = i
                        currentEvent['past'] = True
                        past_tense_state = True
                        tenseChanges.append(currentEvent)
                        currentEvent = {}
            i += 1
        document._.vwp_tense_changes = tenseChanges

    def concreteDetails(self, doc):
        characterList = None
        characterList = doc._.nominalReferences[0]
        details = []
        for token in doc:

            if token._.vwp_abstract:
                continue

            # Higher frequency words aren't likely to be concrete details
            if token._.max_freq > 5:
                continue

            if token._.is_academic:
                continue

            if token.text.capitalize() in characterList \
               and len(characterList[token.text.capitalize()]) > 2:
                continue
            if token._.vwp_direct_speech \
               or token._.vwp_in_direct_speech:
                continue
            if token._.has_governing_subject \
               and doc[token._.governing_subject]._.animate \
               and (token._.vwp_cognitive
                    or token._.vwp_argument
                    or token._.vwp_communication
                    or token._.vwp_emotion):
                continue
            if getLogicalObject(token) is not None \
               and getLogicalObject(token)._.animate \
               and token._.vwp_emotional_impact:
                continue
            if (token._.vwp_evaluation
                or token._.vwp_hedge
                or (token._.abstract_trait
                    and 'of' in [child.text.lower()
                                 for child in token.children])):
                continue
            if (token.pos_ != 'PROPN'
                and (token.dep_ != 'punct'
                     and not token.is_stop)):
                if (token.pos_ == 'NOUN') and \
                   (token._.max_freq is None
                   or token._.nSenses is None
                   or token._.max_freq < 4.5) \
                   and token._.concreteness is not None \
                   and token._.concreteness >= 4:
                    details.append(token.i)
                elif ((token.pos_ == 'VERB') and
                      token._.max_freq is not None
                      and token._.max_freq < 4.2
                      and token._.concreteness is not None
                      and (token._.concreteness >= 3.5
                           or token._.concreteness >= 2.5
                           and getLogicalObject(token) is not None
                           and getLogicalObject(token)._.concreteness
                           is not None
                           and getLogicalObject(token)._.concreteness
                           >= 4.3)):
                    details.append(token.i)
                elif token.pos_ != 'VERB' and token.pos_ != 'NOUN':
                    if token._.has_governing_subject \
                       and doc[
                               token._.governing_subject
                               ]._.concreteness is not None \
                       and doc[token._.governing_subject]._.concreteness >= 4 \
                       and token._.concreteness is not None \
                       and token._.concreteness > 2.5 \
                       and (token._.max_freq is None
                            or token._.max_freq < 4.3
                            or token._.nSenses is None
                            or token._.nSenses < 4
                            and token._.max_freq < 5):
                        details.append(token.i)
                    elif (token.head._.concreteness is not None
                          and token.head._.concreteness >= 4
                          and token._.concreteness is not None
                          and token._.concreteness > 2.5
                          and (token._.max_freq is None
                               or token._.max_freq < 4.3
                               or token._.nSenses is None
                               or token._.nSenses < 4
                               and token._.max_freq < 5)):
                        details.append(token.i)
        doc._.concrete_details = details
