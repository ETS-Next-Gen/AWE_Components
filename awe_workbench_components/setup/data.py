
import argparse

from distutils.command.build import build
from distutils.command.build import build as _build
from distutils.command.install import install
from distutils.command.install import install as _install
from distutils.dir_util import copy_tree
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.develop import develop as _develop
from spacy.cli.download import download
from pathlib import Path
from itertools import chain
from nltk.corpus import wordnet
import site

import nltk
import os
import sys
import sysconfig
import re
import string
import json
import csv
import srsly
import collections
import operator
import numpy as np
from importlib import resources
from pathlib import Path
import pkgutil

from spacy.tokens import Token
# Spacy parser. We set extensions on these classes
# in order to define our own functions to create text features
# to use as metrics.

import wordfreq
# https://github.com/rspeer/wordfreq
# Large word frequency database. Provides Zipf frequencies
# (log scale of frequency) for most English words, based on a
# variety of corpora.


class data:

    def is_ascii(self, s):
        return all(ord(c) < 128 for c in s)

    def load_text_resource(self, filename):
        data = resources.read_text('awe_workbench_components.data',
                                   filename)
        rowSet = data.split('\n')
        return rowSet

    def load_json_resource(self, filename):
        json_file = pkgutil.get_data('awe_workbench_components.data',
                                     filename).decode('utf-8')
        return json.loads(json_file)

    def load_sylcounts(self, mobyhyphenator='mhyph.txt'):
        """
        Calculate syllable counts for words.

        We use moby hyphenator data from the public domain,
        https://web.archive.org/web/20170930060409/http://icon.shef.ac.uk/Moby/

        Hyphenation marks indicate syllable boundaries, hence give us
        syllable count for a word.

        The number of syllables in a word is a well-known predictor of
        vocabulary difficulty.
        """

        Lines = self.load_text_resource(mobyhyphenator)

        syllables = {}
        for line in Lines:
            line = line.replace('\n', '').replace('\ufeff', '')
            if len(line) == 0:
                continue
            nsyll = 1
            for char in line:  # chars:
                if ord(char) in range(65, 91) \
                   or ord(char) in range(97, 123) or char == ' ':
                    continue
                nsyll += 1
            plainline = ''.join([x for x in line if x in string.printable]
                                ).replace('\n', '').replace(' ', '_')
            word = plainline.lower()
            if not re.match('^[-\'.A-Za-z0-9]+$', word):
                continue
            if not self.is_ascii(word):
                continue
            syllables[word] = nsyll
        syllables['an'] = 1
        syllables['ai'] = 1
        syllables['is'] = 1
        syllables['n\'t'] = 1
        return syllables

    def load_wordfamilies(self, wf='wordfamilies.txt'):
        """
        Published at https://enroots.neocities.org/families.txt
        Based on word families from Paul Nation's publication described in
        https://www.wgtn.ac.nz/__data/assets/pdf_file/0005/1857641/about-bnc-coca-vocabulary-list.pdf

        This gives us a way to identify the root word for any given word,
        and/or to identify words that belong to the same word family or to
        look up the size of the family a word belongs to.

        Size of a word's word family is a predictor of vocabulary difficulty.
        """
        roots = {}
        family_sizes = {}
        family_max_freqs = {}
        family_idxs = {}
        family_lists = {}
        Lines = self.load_text_resource(wf)
        familyidx = 0
        for line in Lines:
            line = line.replace('\n', '').replace(
                                '\ufeff', '').replace(' ', '')
            if len(line) == 0:
                continue
            entries = line.split('\t')
            root = entries[0]
            family_size = len(entries) - 1
            freqs = []
            for i in range(1, len(entries)):
                roots[entries[i]] = entries[0]
                freqs.append(wordfreq.zipf_frequency(entries[i], 'en'))
                family_sizes[entries[i].strip()] = family_size
                family_idxs[entries[i].strip()] = familyidx
                if familyidx not in family_lists:
                    family_lists[familyidx] = [entries[i].strip()]
                else:
                    family_lists[familyidx].append(entries[i].strip())
            family_max_freqs[root] = max(freqs)
            # Maximum frequency for a word family which may be more reliable
            # than individual word frequency to predict difficulty
            familyidx += 1

        return roots, family_sizes, family_max_freqs, family_idxs, family_lists

    def load_morpholex(self, morpho='MorphoLexWords.txt'):
        """
        get morpholex data
        from https://github.com/hugomailhot/MorphoLex-en
        Paper: Sánchez-Gutiérrez, C.H., Mailhot, H., Deacon, S.H. et al. \
        Behav Res (2017). https://doi.org/10.3758/s13428-017-0981-8

        We use morpholex to determine the number of morphemes in a word and to
        determine whether words are of latinate origin (by looking for a list
        of specifically latinate prefixes or suffixes.)

        Morpholex only covers about 60K words, so we'll get no morphological
        complexity information for really rare words. TBD: build an extension
        that covers morpholex style analysis for a fuller vocabulary.
        """
        Lines = self.load_text_resource(morpho)
        lineNo = 0
        word = ''
        morphoLex = {}
        latinate_status = {}
        nMorph_status = {}
        latinate_suf = ['>ist>',
                        '>ale>',
                        '>ogony>',
                        '>atist>',
                        '>ous>',
                        '>age>',
                        '>nox>',
                        '>ina>',
                        '>fice>',
                        '>aire>',
                        '>efac>',
                        '>ast>',
                        '>us>',
                        '>asthenic>'
                        '>at>',
                        '>olent>',
                        '>ifer>',
                        '>ode>',
                        '>and>',
                        '>ment>',
                        '>anda>',
                        '>ifix>',
                        '>acle>',
                        '>acul>',
                        '>ica>',
                        '>astica>',
                        '>ious>',
                        '>ant>',
                        '>lyte>',
                        '>al>',
                        '>ose>',
                        '>ivore>',
                        '>tics>',
                        '>id>',
                        '>ancy>',
                        '>esc>',
                        '>ix>',
                        '>ice>',
                        '>itis>',
                        '>etum>',
                        '>abulary>',
                        '>ics>',
                        '>itary>',
                        '>itate>',
                        '>tograph>',
                        '>ade>',
                        '>ide>',
                        '>esimal>',
                        '>ium>',
                        '>itous>',
                        '>atograph>',
                        '>alia>',
                        '>ulin>',
                        '>oneous>',
                        '>oid>',
                        '>eme>',
                        '>ulum>',
                        '>opath>',
                        '>ally>',
                        '>lys>',
                        '>tive>',
                        '>ard>',
                        '>ison>',
                        '>ae>',
                        '>ify>',
                        '>one>',
                        '>enne>',
                        '>iatric>',
                        '>alg>',
                        '>isit>',
                        '>issimo>',
                        '>ine>',
                        '>iac>',
                        '>ated>',
                        '>iana>',
                        '>ius>',
                        '>ance>',
                        '>ata>',
                        '>ola>',
                        '>iasm>',
                        '>alis>',
                        '>asia>',
                        '>syn>',
                        '>ory>',
                        '>ate>',
                        '>sis>',
                        '>efy>',
                        '>iat>',
                        '>batic>',
                        '>ol>',
                        '>iall>',
                        '>atic>',
                        '>tude>',
                        '>ue>',
                        '>asi>',
                        '>ando>',
                        '>itation>',
                        '>itorium>',
                        '>aci>',
                        '>in>',
                        '>sics>',
                        '>ism>',
                        '>endum>',
                        '>uitous>',
                        '>able>',
                        '>crine>',
                        '>illion>',
                        '>oria>',
                        '>thelial>',
                        '>ile>',
                        '>urn>',
                        '>is>',
                        '>ette>',
                        '>eer>',
                        '>na>',
                        '>uple>',
                        '>enda>',
                        '>erry>',
                        '>ure>',
                        '>le>',
                        '>ite>',
                        '>iste>',
                        '>ole>',
                        '>itize>',
                        '>xeur>',
                        '>orial>',
                        '>et>',
                        '>it>',
                        '>batics>',
                        '>ane>',
                        '>rrhage>',
                        '>ic>',
                        '>ec>',
                        '>astic>',
                        '>ion>',
                        '>orium>',
                        '>ule>',
                        '>ar>',
                        '>ists>',
                        '>atics>',
                        '>iance>',
                        '>ater>',
                        '>eutic>',
                        '>uity>',
                        '>our>',
                        '>omat>',
                        '>acy>',
                        '>es>',
                        '>oli>',
                        '>i>',
                        '>ace>',
                        '>iast>',
                        '>um>',
                        '>ize>',
                        '>atum>',
                        '>ive>',
                        '>ity>',
                        '>atoire>',
                        '>itude>',
                        '>tography>',
                        '>fuge>',
                        '>aria>',
                        '>itan>',
                        '>end>',
                        '>ee>',
                        '>ute>',
                        '>lege>',
                        '>tious>',
                        '>arthr>',
                        '>a>',
                        '>o>',
                        '>ogy>',
                        '>ella>',
                        '>ian>',
                        '>ery>',
                        '>ante>',
                        '>icle>',
                        '>os>',
                        '>ence>']

        latinate_pref = ['<mono<',
                         '<juxta<',
                         '<auto<',
                         '<ex<',
                         '<sym<',
                         '<mis<',
                         '<vas<',
                         '<al<',
                         '<er<',
                         '<pre<',
                         '<de<',
                         '<cine<',
                         '<at<',
                         '<dec<',
                         '<cryo<',
                         '<pur<',
                         '<epi<',
                         '<e<',
                         '<min<',
                         '<syn<',
                         '<poly<',
                         '<quadro<',
                         '<pros<',
                         '<es<',
                         '<ob<',
                         '<ante<',
                         '<inter<',
                         '<re<',
                         '<quad<',
                         '<in<',
                         '<retro<',
                         '<deca<',
                         '<ed<',
                         '<muli<',
                         '<dys<',
                         '<sur<',
                         '<sub<',
                         '<centi<',
                         '<cata<',
                         '<eu<',
                         '<intro<',
                         '<ec<',
                         '<infra<',
                         '<quint<',
                         '<dia<',
                         '<con<',
                         '<com<',
                         '<pro<',
                         '<pos<',
                         '<ambi<',
                         '<trans<',
                         '<di<',
                         '<extra<',
                         '<aes<',
                         '<ant<',
                         '<su<',
                         '<hexa<',
                         '<macro<',
                         '<duo<',
                         '<kilo<',
                         '<milli<',
                         '<pan<',
                         '<semi<',
                         '<omni<',
                         '<post<',
                         '<im<',
                         '<letra<',
                         '<hepta<',
                         '<peri<',
                         '<hemi<',
                         '<per<',
                         '<ad<',
                         '<en<',
                         '<sel<',
                         '<ab<',
                         '<tetra<',
                         '<hypo<',
                         '<melli<',
                         '<ana<',
                         '<uni<',
                         '<super<',
                         '<sy<',
                         '<micro<',
                         '<circum<',
                         '<co<',
                         '<ef<',
                         '<septa<',
                         '<demi<',
                         '<mega<',
                         '<dis<',
                         '<sus<',
                         '<multi<',
                         '<mal<',
                         '<se<',
                         '<intra<',
                         '<pent<',
                         '<tel<',
                         '<anti<',
                         '<penta<',
                         '<octo<',
                         '<hyper<',
                         '<ultra<',
                         '<meta<',
                         '<sexa<',
                         '<bi<',
                         '<tri<',
                         '<pano<',
                         '<pel<']

        for line in Lines:
            line = line.replace('\n', '').replace(
                                '\ufeff', '').replace(' ', '')
            if len(line) == 0:
                continue
            if lineNo == 0:
                headers = line.split('\t')
            else:
                icount = 0
                items = line.split('\t')
                for item in items:
                    if icount == 1:
                        word = item.lower()
                        if not re.match('^[-\'.A-Za-z0-9]+$', word):
                            break
                    elif icount > 0:
                        if icount < len(headers):
                            value = item.replace('\n', '')
                            label = headers[icount]
                            if value is not None and len(value) > 0:
                                if word not in morphoLex:
                                    dat = {}
                                else:
                                    dat = morphoLex[word]
                                dat[label] = value
                                morphoLex[word] = dat
                                latinate = 0
                                nMorph = 1
                                if 'MorphoLexSegm' in dat:
                                    nMorph = dat['Nmorph']
                                    prefixes = \
                                        re.findall('[<][a-z]+[<]',
                                                   dat['MorphoLexSegm'])
                                    suffixes = \
                                        re.findall('[>][a-z]+[>]',
                                                   dat['MorphoLexSegm'])
                                    for pref in prefixes:
                                        if pref in latinate_pref:
                                            latinate = 1
                                            break
                                    for suf in suffixes:
                                        if suf in latinate_suf:
                                            latinate = 1
                                            break
                                    latinate_status[word] = latinate
                                nMorph_status[word] = nMorph
                    icount += 1
            lineNo += 1
        return morphoLex, latinate_status, nMorph_status

    def load_sentiword(self, sentiword='data/SentiWord_1.0.csv'):
        """
         Get SentiWord data (in compressed form) for word positive/
         negative polarity. Original data from https://hlt-nlp.fbk.eu/
         technologies/sentiwords. Gatti, L., Guerini, M., & Turchi, M.
         (2016). SentiWords: Deriving a high precision and high coverage
         lexicon for sentiment analysis. IEEE Transactions on Affective
         Computing, 7(4), 409-421. [Preprint]
        """
        Lines = self.load_text_resource('SentiWord_1.0.csv')
        sentiment = {}
        first = True
        for line in Lines:
            line = \
                line.replace('\n', '').replace('\ufeff', '').replace(' ', '')
            if len(line) == 0:
                continue
            if first:
                first = False
                continue
            word, polarity = line.split(',')
            word = word.replace('"', '')
            polarity = polarity.replace('\n', '')
            polarity = polarity.replace('\r', '')
            sentiment[word] = float(polarity)
        return sentiment

    def compile_academic_wordlists(self, AWL='awlsublists.txt',
                                   NAWL='NAWL.txt',
                                   AKL='AcademicKeywordList.txt',
                                   UWL='UniversityWordList.txt',
                                   EMSWL='EnglishMS_WordList.txt',
                                   HMSWL='Health_MS_WordList.txt',
                                   MMSWL='Math_MS_WordList.txt',
                                   SMSWL='Science_MS_WordList.txt',
                                   SSMSWL='SocialStudies_MS_WordList.txt',
                                   TR_II='Tier-2-vocab-K-12.txt'):

        """
        Load academic words from various word lists -- slightly modified to
        eliminate occasional words that have too many nonacademic uses, like
        'kid' or 'huge'. We end up with a flag for words that humans have
        selected as words people need to know to deal with academic texts.
        This is basically a measure of Tier II vocabulary use.
        """
        academic = []

        # Academic Word List (Coxhead)
        # https://www.wgtn.ac.nz/lals/resources/academicwordlist
        Lines = self.load_text_resource(AWL)
        for line in Lines:
            word = line.replace('\n', '').replace(
                                '\ufeff', '').replace(' ', '')
            if len(word) == 0:
                continue
            if word not in academic:
                academic.append(word)

        # New Academic Word List
        # https://www.eapfoundation.com/vocab/academic/nawl/
        Lines = self.load_text_resource(NAWL)
        for line in Lines:
            word = line.replace('\n', '').replace(
                                '\ufeff', '').replace(' ', '')
            if len(word) == 0:
                continue
            if word not in academic:
                academic.append(word)

        # Academic Keyword List
        # https://uclouvain.be/en/research-institutes/ilc/
        # cecl/academic-keyword-list.html
        Lines = self.load_text_resource(AKL)
        for line in Lines:
            word = line.replace('\n', '').replace(
                                '\ufeff', '').replace(' ', '')
            if len(word) == 0:
                continue
            if word not in academic:
                academic.append(word)

        # University Word List
        # http://webhome.auburn.edu/~nunnath/engl6240/wlistuni.html
        Lines = self.load_text_resource(UWL)
        for line in Lines:
            word = line.replace('\n', '').replace(
                                '\ufeff', '').replace(' ', '')
            if len(word) == 0:
                continue
            if word not in academic:
                academic.append(word)

        # English Middle School Word List
        # Posted at https://lextutor.ca/vp/eng/
        # nat_lists/ms-cat_subs/english.pdf
        Lines = self.load_text_resource(EMSWL)
        for line in Lines:
            word = line.replace('\n', '').replace(
                                '\ufeff', '').replace(' ', '')
            if len(word) == 0:
                continue
            if word not in academic:
                academic.append(word)

        # Health Middle School Word List
        # Posted at https://lextutor.ca/vp/eng/nat_lists/ms-cat_subs/health.pdf
        Lines = self.load_text_resource(HMSWL)
        for line in Lines:
            word = line.replace('\n', '').replace(
                                '\ufeff', '').replace(' ', '')
            if len(word) == 0:
                continue
            if word not in academic:
                academic.append(word)

        # Math Middle School Word List
        # Posted at https://lextutor.ca/vp/eng/nat_lists/ms-cat_subs/math.pdf
        Lines = self.load_text_resource(MMSWL)
        for line in Lines:
            word = line.replace('\n', '').replace(
                                '\ufeff', '').replace(' ', '')
            if len(word) == 0:
                continue
            if word not in academic:
                academic.append(word)

        # Science Middle School Word List
        # Posted at https://lextutor.ca/vp/eng/nat_lists/
        # ms-cat_subs/science.pdf
        Lines = self.load_text_resource(SMSWL)
        for line in Lines:
            word = line.replace('\n', '').replace(
                                '\ufeff', '').replace(' ', '')
            if len(word) == 0:
                continue
            if word not in academic:
                academic.append(word)

        # Social Studies Middle School Word List
        # Posted at https://lextutor.ca/vp/eng/nat_lists/
        # ms-cat_subs/social_and_history.pdf
        Lines = self.load_text_resource(SSMSWL)
        for line in Lines:
            word = line.replace('\n', '').replace(
                                '\ufeff', '').replace(' ', '')
            if len(word) == 0:
                continue
            if word not in academic:
                academic.append(word)

        academic.sort()

        return academic

    def loadTransitions(self, datafile='transition_terms.json'):
        """
         Load a hand-crafted list of transition words and phrases
         along with their categorization into types corresponding
         to different sorts of rhetorical moves.
        """
        temp = {}
        transition_terms = {}
        transition_categories = ['positive',
                                 'conditional',
                                 'consequential',
                                 'contrastive',
                                 'counterpoint',
                                 'comparative',
                                 'crossreferential',
                                 'illustrative',
                                 'negative',
                                 'emphatic',
                                 'evidentiary',
                                 'general',
                                 'ordinal',
                                 'purposive',
                                 'periphrastic',
                                 'hypothetical',
                                 'summative',
                                 'introductory']
        temp = self.load_json_resource(datafile)
        for key in temp:
            for item in temp[key]:
                if key == 'positive':
                    transition_terms[item] = 0
                elif key == 'conditional':
                    transition_terms[item] = 1
                elif key == 'consequential':
                    transition_terms[item] = 2
                elif key == 'contrastive':
                    transition_terms[item] = 3
                elif key == 'counterpoint':
                    transition_terms[item] = 4
                elif key == 'comparative':
                    transition_terms[item] = 5
                elif key == 'crossreferential':
                    transition_terms[item] = 6
                elif key == 'illustrative':
                    transition_terms[item] = 7
                elif key == 'negative':
                    transition_terms[item] = 8
                elif key == 'emphatic':
                    transition_terms[item] = 9
                elif key == 'evidentiary':
                    transition_terms[item] = 10
                elif key == 'general':
                    transition_terms[item] = 11
                elif key == 'ordinal':
                    transition_terms[item] = 12
                elif key == 'purposive':
                    transition_terms[item] = 13
                elif key == 'periphrastic':
                    transition_terms[item] = 14
                elif key == 'hypothetical':
                    transition_terms[item] = 15
                elif key == 'summative':
                    transition_terms[item] = 16
                elif key == 'introductory':
                    transition_terms[item] = 17

        return transition_terms, transition_categories

    def loadViewpointLexicon(self, viewpointLex='argVoc.csv'):
        """
        This functionloads a manually curated database of viewpoint and
        argument vocabulary. The categories associated with the words
        in the database are used to drive a rule-based system that
        identifies viewpoint expressions (i.e., the prototypically
        animate nouns that link to subjective language in the text
        to establish point of view.)

        We add the categories to individual word tokens via extensions
        to the spacy Token object
        """

        stancePerspectiveVoc = {}

        filepath = resources.path('awe_workbench_components.data',
                                  viewpointLex)

        with open(filepath, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0
            for row in csv_reader:
                if row['word'] not in stancePerspectiveVoc:
                    stancePerspectiveVoc[row['word']] = {}
                if row['pos'] not in stancePerspectiveVoc[row['word']]:
                    stancePerspectiveVoc[row['word']][row['pos']] = []
                if row['mode'] is not None \
                   and row['mode'] not in stancePerspectiveVoc[
                       row['word']][row['pos']]:

                    stancePerspectiveVoc[row['word']][
                        row['pos']].append(row['mode'])
                    Token.set_extension('vwp_'
                                        + row['mode'],
                                        default=False,
                                        force=True)

                if row['category'] is not None \
                   and (row['category'] not in
                        stancePerspectiveVoc[row['word']][row['pos']]):

                    stancePerspectiveVoc[row['word']][row['pos']].append(
                        row['category'])
                    Token.set_extension('vwp_'
                                        + row['category'],
                                        default=False,
                                        force=True)
        return stancePerspectiveVoc

    def to_disk(self):

        syllables = self.load_sylcounts()

        roots, family_sizes, family_max_freqs, family_idxs, family_lists = \
            self.load_wordfamilies()

        morpholex, latinate, nMorph_status = \
            self.load_morpholex()

        sentiment = self.load_sentiword()

        academic = self.compile_academic_wordlists()

        transition_terms, transition_categories = \
            self.loadTransitions()

        stancePerspectiveVoc = \
            self.loadViewpointLexicon()

        outputfile = resources.path('awe_workbench_components.json_data',
                                    'syllables.json')
        srsly.write_json(outputfile, syllables)

        outputfile = resources.path('awe_workbench_components.json_data',
                                    'roots.json')
        srsly.write_json(outputfile, roots)

        outputfile = resources.path('awe_workbench_components.json_data',
                                    'family_sizes.json')
        srsly.write_json(outputfile, family_sizes)

        outputfile = resources.path('awe_workbench_components.json_data',
                                    'family_max_freqs.json')
        srsly.write_json(outputfile, family_max_freqs)

        outputfile = resources.path('awe_workbench_components.json_data',
                                    'family_idxs.json')
        srsly.write_json(outputfile, family_idxs)

        outputfile = resources.path('awe_workbench_components.json_data',
                                    'family_lists.json')
        srsly.write_json(outputfile, family_lists)

        outputfile = resources.path('awe_workbench_components.json_data',
                                    'morpholex.json')
        srsly.write_json(outputfile, morpholex)

        outputfile = resources.path('awe_workbench_components.json_data',
                                    'latinate.json')
        srsly.write_json(outputfile, latinate)

        outputfile = resources.path('awe_workbench_components.json_data',
                                    'nMorph_status.json')
        srsly.write_json(outputfile, nMorph_status)

        outputfile = resources.path('awe_workbench_components.json_data',
                                    'sentiment.json')
        srsly.write_json(outputfile, sentiment)

        outputfile = resources.path('awe_workbench_components.json_data',
                                    'academic.json')
        srsly.write_json(outputfile, academic)

        outputfile = resources.path('awe_workbench_components.json_data',
                                    'transition_terms.json')
        srsly.write_json(outputfile, transition_terms)

        outputfile = resources.path('awe_workbench_components.json_data',
                                    'transition_categories.json')
        srsly.write_json(outputfile, transition_categories)

        outputfile = resources.path('awe_workbench_components.json_data',
                                    'stancePerspectiveVoc.json')
        srsly.write_json(outputfile, stancePerspectiveVoc)

    def get_all_hypernyms(self, synset):
        returnSet = []
        if len(synset.instance_hypernyms()) > 0:
            instances = synset.instance_hypernyms()
            for instance in instances:
                hyper = self.get_all_hypernyms(instance)
                for item in hyper:
                    if item not in returnSet:
                        returnSet.append(item)
        moreSet = synset.hypernyms()
        for syn in moreSet:
            returnSet.append(syn)
        hyperSet = returnSet
        for hyper in hyperSet:
            new = self.get_all_hypernyms(hyper)
            for item in new:
                if item not in returnSet:
                    returnSet.append(item)
        return returnSet

    # from https://www.datacamp.com/community/tutorials/fuzzy-string-python
    def levenshtein_ratio_and_distance(self, s, t, ratio_calc=False):
        """ levenshtein_ratio_and_distance:
            Calculates levenshtein distance between two strings.
            If ratio_calc = True, the function computes the
            levenshtein distance ratio of similarity between two strings
            For all i and j, distance[i,j] will contain the Levenshtein
            distance between the first i characters of s and the
            first j characters of t
        """
        # Initialize matrix of zeros
        rows = len(s) + 1
        cols = len(t) + 1
        distance = np.zeros((rows, cols), dtype=int)

        # Populate matrix of zeros with the indeces of each character of
        # both strings
        for i in range(1, rows):
            for k in range(1, cols):
                distance[i][0] = i
                distance[0][k] = k

        # Iterate over the matrix to compute the cost of deletions, insertions
        # and/or substitutions
        for col in range(1, cols):
            for row in range(1, rows):
                if s[row - 1] == t[col - 1]:
                    cost = 0
                    # If the characters are the same in the two strings in
                    # a given position [i,j] then the cost is 0
                else:
                    # In order to align the results with those of the Python
                    # Levenshtein package, if we choose to calculate the ratio
                    # the cost of a substitution is 2. If we calculate just
                    # distance, then the cost of a substitution is 1.
                    if ratio_calc:
                        cost = 2
                    else:
                        cost = 1
                distance[row][col] = min(distance[row-1][col] + 1,
                                         distance[row][col-1] + 1,
                                         distance[row-1][col-1] + cost)
                # Cost of insertions, cost of deletions, cost of substitutions
        if ratio_calc:
            # Computation of the Levenshtein Distance Ratio
            Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
            return Ratio
        else:
            return distance[row][col]

    def processConcretes(self):
        rawconcretes = {}
        filepath = resources.path('awe_workbench_components.data',
                                  'Concr_Base.csv')
        with open(filepath, mode='r', encoding='ISO-8859-1') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0
            synsetCounts = {}
            synsetSums = {}

            # First estimate concreteness using hypernym and meronym
            # relations only
            print('Make initial concreteness estimates for WordNet \
                   synsets using hypernym and meronym relations only.')
            for row in csv_reader:
                word = row['Word']
                concreteness = float(row['CNC_M'])
                rawconcretes[word] = concreteness
                synsets = wordnet.synsets(word)
                if len(synsets) > 0:
                    if synsets[0] not in synsetCounts:
                        synsetCounts[synsets[0]] = 1
                        synsetSums[synsets[0]] = concreteness
                    else:
                        synsetCounts[synsets[0]] += 1
                        synsetSums[synsets[0]] += concreteness
                    for synset in self.get_all_hypernyms(synsets[0]):
                        if synset.name() == 'entity.n.1':
                            continue
                        if synset not in synsetCounts:
                            synsetCounts[synset] = 1
                            synsetSums[synset] = concreteness
                        else:
                            synsetCounts[synset] += 1
                            synsetSums[synset] += concreteness
                    for synset in synsets[0].part_meronyms():
                        if synset not in synsetCounts:
                            synsetCounts[synset] = 1
                            synsetSums[synset] = concreteness
                        else:
                            synsetCounts[synset] += 1
                            synsetSums[synset] += concreteness
                    for synset in synsets[0].substance_meronyms():
                        if synset not in synsetCounts:
                            synsetCounts[synset] = 1
                            synsetSums[synset] = concreteness
                        else:
                            synsetCounts[synset] += 1
                            synsetSums[synset] += concreteness
                    for synset in synsets[0].member_meronyms():
                        if synset not in synsetCounts:
                            synsetCounts[synset] = 1
                            synsetSums[synset] = concreteness
                        else:
                            synsetCounts[synset] += 1
                            synsetSums[synset] += concreteness

            # Then extend it using all other relations
            print('Extend concreteness estimates for WordNet synsets \
                   using all other relations.')
            for row in csv_reader:
                word = row['Word']
                concreteness = float(row['CNC_M'])
                synsets = wordnet.synsets(word)
                if len(synsets) > 0 and synsets[0] not in synsetCounts:
                    for lemma in synsets[0].lemmas():
                        for antonym in lemma.antonyms():
                            if antonym.synset() not in synsetCounts:
                                synsetCounts[antonym.synset()] = 1
                                synsetSums[antonym.synset()] = \
                                    concreteness
                            else:
                                synsetCounts[antonym.synset()] += 1
                                synsetSums[antonym.synset()] += \
                                    concreteness
                        for pertainym in lemma.pertainyms():
                            if pertainym.synset() not in synsetCounts:
                                synsetCounts[pertainym.synset()] = 1
                                if len(pertainym.synset().hypernyms()) > 0:
                                    synsetSums[pertainym.synset()] = \
                                        concreteness * .8
                                else:
                                    synsetSums[pertainym.synset()] = \
                                        concreteness
                            else:
                                synsetCounts[pertainym.synset()] += 1
                                if len(pertainym.synset().hypernyms()) > 0:
                                    synsetSums[pertainym.synset()] += \
                                        concreteness*.8
                                else:
                                    synsetSums[pertainym.synset()] += \
                                        concreteness

                        for related in lemma.derivationally_related_forms():
                            if related.synset() not in synsetCounts:
                                synsetCounts[related.synset()] = 1
                                if len(related.synset().hypernyms()) > 0:
                                    synsetSums[related.synset()] = \
                                        concreteness * .8
                                else:
                                    synsetSums[related.synset()] = \
                                        concreteness
                            else:
                                synsetCounts[related.synset()] += 1
                                if len(related.synset().hypernyms()) > 0:
                                    synsetSums[related.synset()] += \
                                        concreteness*.8
                                else:
                                    synsetSums[related.synset()] += \
                                        concreteness

                    for synset in synsets[0].similar_tos():
                        if synset not in synsetCounts:
                            synsetCounts[synset] = 1
                            synsetSums[synset] = concreteness
                        else:
                            synsetCounts[synset] += 1
                            synsetSums[synset] += concreteness
                        for attr in synset.attributes():
                            if attr in synsetCounts:
                                synsetCounts[synset] += \
                                    synsetCounts[attr]
                                if len(synset.attributes().hypernyms()) > 0:
                                    synsetSums[synset] += \
                                        synsetSums[attr]*.8
                                else:
                                    synsetSums[synset] += \
                                        synsetSums[attr]
                        for lemma in synset.lemmas():
                            for antonym in lemma.antonyms():
                                if antonym.synset() in synsetCounts:
                                    synsetCounts[synset] += \
                                        synsetCounts[antonym.synset()]
                                    synsetSums[synset] += \
                                        synsetSums[antonym.synset()]
                            for pertainym in lemma.pertainyms():
                                if pertainym.synset() in synsetCounts:
                                    synsetCounts[synset] += \
                                        synsetCounts[pertainym.synset()]*.8
                                    if len(pertainym.synset()) > 0:
                                        synsetSums[synset] += \
                                            synsetSums[pertainym.synset()]*.8
                                    else:
                                        synsetSums[synset] += \
                                            synsetSums[pertainym.synset()]
                            for related in lemma.derivationally_related_forms():
                                if related.synset() in synsetCounts:
                                    synsetCounts[synset] += \
                                        synsetCounts[related.synset()]
                                    if len(related.synset().hypernyms()) > 0:
                                        synsetSums[synset] += \
                                            synsetSums[related.synset()]*.8
                                    else:
                                        synsetSums[synset] += \
                                            synsetSums[related.synset()]

                    for synset in synsets[0].attributes():
                        if synset not in synsetCounts:
                            synsetCounts[synset] = 1
                            if len(synset.hypernyms()) > 0:
                                synsetSums[synset] = concreteness*.8
                            else:
                                synsetSums[synset] = concreteness
                        else:
                            synsetCounts[synset] += 1
                            if len(synset.hypernyms()) > 0:
                                synsetSums[synset] += concreteness*.8
                            else:
                                synsetSums[synset] += concreteness
                    for synset in synsets[0].verb_groups():
                        if synset not in synsetCounts:
                            synsetCounts[synset] = 1
                            synsetSums[synset] = concreteness
                        else:
                            synsetCounts[synset] += 1
                            synsetSums[synset] += concreteness
                    for synset in synsets[0].entailments():
                        if synset not in synsetCounts:
                            synsetCounts[synset] = 1
                            synsetSums[synset] = concreteness
                        else:
                            synsetCounts[synset] += 1
                            synsetSums[synset] += concreteness

            print('Extend estimates further by traversing hypernym relations\
                   for synsets not yet assigned a concreteness estimate')
            for synset in wordnet.all_synsets():
                if synset not in synsetCounts:
                    if len(synset.hypernyms()) > 0 \
                       and synset.hypernyms()[0] in synsetCounts:
                        synsetCounts[synset] = \
                            synsetCounts[synset.hypernyms()[0]]
                        synsetSums[synset] = \
                            synsetSums[synset.hypernyms()[0]]
                    elif len(self.get_all_hypernyms(synset)) > 0:
                        for hyper in self.get_all_hypernyms(synset):
                            if synset not in synsetCounts \
                               and hyper in synsetCounts \
                               and hyper.name() != 'entity.n.1':
                                synsetCounts[synset] = \
                                    synsetCounts[hyper]
                                synsetSums[synset] = \
                                    synsetSums[hyper]
                            elif (hyper in synsetCounts
                                  and hyper.name() != 'entity.n.1'):
                                synsetCounts[synset] += \
                                    synsetCounts[hyper]
                                synsetSums[synset] += \
                                    synsetSums[hyper]

            print('Extend estimates further by traversing other relations\
                   for synsets not yet assigned a concreteness estimate')
            for synset in wordnet.all_synsets():
                if synset not in synsetCounts:
                    for synset in synsets[0].also_sees():
                        if synset not in synsetCounts:
                            synsetCounts[synset] = 1
                            synsetSums[synset] = concreteness
                        else:
                            synsetCounts[synset] += 1
                            synsetSums[synset] += concreteness
                    for synset in synsets[0].causes():
                        if synset not in synsetCounts:
                            synsetCounts[synset] = 1
                            synsetSums[synset] = concreteness
                        else:
                            synsetCounts[synset] += 1
                            synsetSums[synset] += concreteness
                    if synset not in synsetCounts:
                        synsetCounts[synset] = 0
                        synsetSums[synset] = 0
                    if len(synset.attributes()) > 0:
                        for attr in synset.attributes():
                            if attr in synsetCounts:
                                synsetCounts[synset] += \
                                    synsetCounts[attr]
                                synsetSums[synset] += \
                                    synsetSums[attr]*.8
                    if len(synset.verb_groups()) > 0:
                        for attr in synset.verb_groups():
                            if attr in synsetCounts:
                                synsetCounts[synset] += \
                                    synsetCounts[attr]
                                synsetSums[synset] += \
                                    synsetSums[attr]
                    if len(synset.similar_tos()) > 0:
                        for attr in synset.similar_tos():
                            if attr in synsetCounts:
                                synsetCounts[synset] += \
                                    synsetCounts[attr]
                                synsetSums[synset] += \
                                    synsetSums[attr]
                    for lemma in synset.lemmas():
                        for antonym in lemma.antonyms():
                            if antonym.synset() in synsetCounts:
                                synsetCounts[synset] += \
                                    synsetCounts[antonym.synset()]
                                synsetSums[synset] += \
                                    synsetSums[antonym.synset()]
                        for pertainym in lemma.pertainyms():
                            if pertainym.synset() in synsetCounts:
                                synsetCounts[synset] += \
                                    synsetCounts[pertainym.synset()]
                                synsetSums[synset] += \
                                    synsetSums[pertainym.synset()]*.8
                        for related in lemma.derivationally_related_forms():
                            if related.synset() in synsetCounts:
                                synsetCounts[synset] += \
                                    synsetCounts[related.synset()]
                                synsetSums[synset] += \
                                    synsetSums[related.synset()]*.8

            concrDict = {}
            print('Handle concreteness estimates for compound words')
            for key in wordnet.all_synsets():
                if key in synsetCounts:
                    if synsetCounts[key] > 0:
                        concrDict[key] = synsetSums[key] / synsetCounts[key]
                    else:
                        # We don't have working instance of relations for
                        # named entities in nltk, so the best we can do
                        # is take the average of the component words
                        for name in key.lemma_names():
                            if '_' in name:
                                subwords = name.split('_')
                                totalCr = 0
                                countCr = 0
                                for wd in subwords:
                                    synsets = wordnet.synsets(wd)
                                    for synset in synsets:
                                        if synset in concrDict:
                                            totalCr += concrDict[synset]
                                            countCr += 1
                                            break
                                if countCr > 0:
                                    concrDict[key] = totalCr / countCr
                            elif '-' in name:
                                subwords = name.split('-')
                                totalCr = 0
                                countCr = 0
                                for wd in subwords:
                                    synsets = wordnet.synsets(wd)
                                    for synset in synsets:
                                        if synset in concrDict:
                                            totalCr += concrDict[synset]
                                            countCr += 1
                                            break
                                if countCr > 0:
                                    concrDict[key] = totalCr / countCr
                            elif name.endswith('ly'):
                                base = name[0: len(name) - 2]
                                synsets = wordnet.synsets(base)
                                for synset in synsets:
                                    if synset in concrDict:
                                        concrDict[key] = concrDict[synset]
                        if key not in concrDict:
                            concrDict[key] = 1
                elif key in ['I',
                             'me',
                             'we',
                             'us',
                             'you',
                             'he',
                             'him',
                             'she',
                             'her',
                             'they',
                             'them']:
                    concrDict[key] = 6.5
                else:
                    # We don't have working instance of relations for named
                    # entities in nltk, so the best we can do is take the
                    # average of the component words
                    for name in key.lemma_names():
                        if '_' in name:
                            subwords = name.split('_')
                            totalCr = 0
                            countCr = 0
                            for wd in subwords:
                                synsets = wordnet.synsets(wd)
                                for synset in synsets:
                                    if synset in concrDict:
                                        totalCr += concrDict[synset]
                                        count += 1
                                        break
                            if countCr > 0:
                                concrDict[key] = totalCr / countCr
                        elif '-' in name:
                            subwords = name.split('-')
                            totalCr = 0
                            countCr = 0
                            for wd in subwords:
                                synsets = wordnet.synsets(wd)
                                for synset in synsets:
                                    if synset in concrDict:
                                        totalCr += concrDict[synset]
                                        countCr += 1
                                        break
                            if countCr > 0:
                                concrDict[key] = totalCr/countCr
                        elif name.endswith('ly'):
                            base = name[0:len(name) - 2]
                            synsets = wordnet.synsets(base)
                            for synset in synsets:
                                if synset in concrDict:
                                    concrDict[key] = concrDict[synset]
                    if key not in concrDict:
                        concrDict[key] = 3
                        # Words we can't link in any way are mostly denominal
                        # adjectives so this is a decent estimate

            finalDict = {}
            print('produce final concreteness dictionary')
            for lemma in wordnet.all_lemma_names():
                if lemma not in finalDict:
                    finalDict[lemma] = {}
                syns = wordnet.synsets(lemma)
                for syn in syns:
                    if '.n.' in syn.name():
                        if 'NOUN' not in finalDict[lemma]:
                            bestLem = lemma
                            bestDist = 50000
                            if lemma not in syn.lemma_names():
                                for lm in syn.lemma_names():
                                    ld = self.levenshtein_ratio_and_distance(
                                        lemma, lm)
                                    if float(ld) < float(bestDist):
                                        bestDict = ld
                                        bestLem = str(lm)
                            if bestLem not in finalDict:
                                finalDict[bestLem] = {}
                            finalDict[bestLem]['NOUN'] = concrDict[syn]
                            if bestLem != lemma and lemma in rawconcretes:
                                if lemma not in finalDict:
                                    finalDict[lemma] = {}
                                finalDict[lemma]['NOUN'] = rawconcretes[lemma]
                    if '.v.' in syn.name():
                        if 'VERB' not in finalDict[lemma]:
                            bestLem = lemma
                            bestDist = 50000
                            if lemma not in syn.lemma_names():
                                for lm in syn.lemma_names():
                                    ld = self.levenshtein_ratio_and_distance(
                                        lemma, lm)
                                    if float(ld) < float(bestDist):
                                        bestDict = ld
                                        bestLem = str(lm)
                            if bestLem not in finalDict:
                                finalDict[bestLem] = {}
                            finalDict[bestLem]['VERB'] = concrDict[syn]
                            if bestLem != lemma and lemma in rawconcretes:
                                if lemma not in finalDict:
                                    finalDict[lemma] = {}
                                finalDict[lemma]['VERB'] = rawconcretes[lemma]
                    if '.r.' in syn.name():
                        if 'ADV' not in finalDict[lemma]:
                            bestLem = lemma
                            bestDist = 50000
                            if lemma not in syn.lemma_names():
                                for lm in syn.lemma_names():
                                    ld = self.levenshtein_ratio_and_distance(
                                        lemma, lm)
                                    if float(ld) < float(bestDist):
                                        bestDict = ld
                                        bestLem = str(lm)
                            if bestLem not in finalDict:
                                finalDict[bestLem] = {}
                            finalDict[bestLem]['ADV'] = concrDict[syn]
                            if bestLem != lemma and lemma in rawconcretes:
                                if lemma not in finalDict:
                                    finalDict[lemma] = {}
                                finalDict[lemma]['ADV'] = rawconcretes[lemma]
                    if '.a.' in syn.name() or '.s.' in syn.name():
                        if 'ADJ' not in finalDict[lemma]:
                            bestLem = lemma
                            bestDist = 50000
                            if lemma not in syn.lemma_names():
                                for lm in syn.lemma_names():
                                    ld = self.levenshtein_ratio_and_distance(
                                        lemma, lm)
                                    if float(ld) < float(bestDist):
                                        bestDict = ld
                                        bestLem = str(lm)
                            if bestLem not in finalDict:
                                finalDict[bestLem] = {}
                            finalDict[bestLem]['ADJ'] = concrDict[syn]
                            if bestLem != lemma and lemma in rawconcretes:
                                if lemma not in finalDict:
                                    finalDict[lemma] = {}
                                finalDict[lemma]['ADJ'] = rawconcretes[lemma]

            extras = {}
            for lemma in finalDict:
                if '_' in lemma and lemma.replace('_', '') not in finalDict:
                    extras[lemma.replace('_', '')] = finalDict[lemma]
                if '-' in lemma and lemma.replace('-', '') not in finalDict:
                    extras[lemma.replace('-', '')] = finalDict[lemma]
            for lemma in extras:
                finalDict[lemma] = extras[lemma]

            outputfile = resources.path('awe_workbench_components.json_data',
                                        'concretes.json')
            srsly.write_json(outputfile, finalDict)

    def extra_build_commands(self):

        if develop:
            print('Preparing json data for AWE Workbench components.')
            self.to_disk()

            print('Preparing json concreteness lexicon.')
            self.processConcretes()

    def extra_install_commands(self):

        print('Downloading Spacy and Coreferee Lexicons')
        download('en_core_web_sm')
        download('en_core_web_lg')
        download('en_core_web_trf')
        os.system("python3 -m coreferee install en")
        nltk.download('wordnet')
        nltk.download('stopwords')
        nltk.download('omw-1.4')
        nltk.download('punkt')

    def __init__(self, args):
        if args.develop:
            self.extra_build_commands()

        if args.install or args.develop:
            self.extra_install_commands()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Run AWE Workbench data download')
    parser.add_argument("-d",
                        "--develop",
                        help="Runs the data downloads to the development \
                              rather than the build location.",
                        action="store_true")
    parser.add_argument("-i",
                        "--install",
                        help="Runs the data downloads to the development \
                              rather than the build location.",
                        action="store_true")

    args = parser.parse_args()

    data(args)
