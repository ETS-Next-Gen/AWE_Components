#!/usr/bin/env python3
# Copyright 2022, Educational Testing Service

import re
import spacy
import srsly
import wordfreq
import numpy as np
import os
from collections import OrderedDict

from scipy.spatial.distance import cosine
# Standard cosine distance metric

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

from spacy.tokens import Token, Doc
from spacy.language import Language

from .utility_functions import *
from ..errors import *

lang = "en"


@Language.component("lexicalclusters")
def assignClusterIDs(hdoc):

    ####################################################################
    # Use agglomerative clustering to group the words inside a document
    # into clusters of related words
    ####################################################################

    filteredVecs = []
    filteredToks = []
    skippedVecs = []
    skippedToks = []
    for i, token in enumerate(hdoc):
        word = token.text.lower().replace('_', ' ')

        #################################################
        # Resolve coreference so we can cluster pronouns
        # with their antecedents
        #################################################
        if token.tag in ['PRP', 'PRP$', 'DT', 'CD', 'WP', 'WP$']:
            referents = ResolveReference(token.i, hdoc)
            if referents is not None \
               and not all_zeros(sum(x.vector for x in referents)):
                # Note: we can't add zero vectors to the output, as
                # that will cause errors in the clustering algorithm
                filteredVecs.append(sum(x.vector for x in referents))
                filteredToks.append(
                    word + '_'
                    + token.lemma_.replace('_', ' ')
                    + '('
                    + ''.join([x.text.lower() for x in referents])
                    + ')_'
                    + str(token.i))

        #######################################################
        # Filter other words so that they're only likely to be
        # true content words
        #######################################################
        if not token.is_stop and word != '.' \
           and not re.match('[-,;:"\'?/><{})(!`~_]+', word) \
           and not wordfreq.zipf_frequency(token.text.lower(), 'en') > 5.7 \
           and token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV']:
            if token.has_vector and not all_zeros(token.vector):
                filteredVecs.append(token.vector)
                filteredToks.append(word
                                    + '_'
                                    + token.lemma_.replace('_', ' ')
                                    + '_'
                                    + str(token.i))

    #################################################
    # if the document has enough data to cluster ...
    #################################################

    try:
        ################################################################
        # Occasionally we can get a vector that is effectively zero but
        # not technically zero, which will crash the clustering by
        # generating an error in the clustering algorithm. So we wrap
        # this piece of the code in an error-handler.
        ################################################################
        if len(filteredVecs) > 1:

            ########################################################
            # Rescale the data so it is centered on the means for
            # this sample. We'll cluster together anything that is
            # clearly above zero similarity after this scaling ...
            ########################################################
            scaler = StandardScaler()
            filteredVecs = scaler.fit_transform(filteredVecs)

            ########################################################
            # Run agglomerative clustering                         #
            ########################################################
            simThreshold = 0.0
            clustering = AgglomerativeClustering(
                affinity='cosine',
                compute_full_tree=True,
                linkage='complete',
                distance_threshold=simThreshold,
                n_clusters=None)
            clustering.fit(filteredVecs)

            ###########################################################
            # Assign cluster labels close to the zero-point for cosine
            # similiarity in the rescaled data. Restructure the output
            # and add in fields for lexical features, then associate
            # that data with the spacy token.
            ############################################################
            tree, vecs, sim = get_all_children(
                clustering,
                len(clustering.children_) - 1,
                filteredToks,
                filteredVecs,
                verbose=False)
            flattened = flatten(tree, [])
            clustered_data = assignClusterLabels(flattened, simThreshold)

            for line in clustered_data:
                word = line[0]
                lemma = line[1]
                locus = line[2]
                clusterID = line[3]
                hdoc[int(locus)]._._tclust = clusterID

            ###########################################################
            # restructure the data into a form that can be associated #
            # with the document as a whole and associate that record  #
            # with the spacy document                                 #
            ###########################################################
            hdoc._._clinfo = calculateClusterProfile(hdoc)

    except Exception as e:
        print('clustering error\n', e)
        hdoc._._clinfo = None

    return hdoc


def get_all_children(agg_cluster, k, tokenized, vectors, verbose=False):
    """
     Restructure the output of AgglomerativeClustering recursively
     into a format that's easier to work with
    """
    node_dict = {}
    leaf_count = agg_cluster.n_leaves_
    n_samples = len(agg_cluster.labels_)
    i, j = agg_cluster.children_[k]
    simL = 0
    simR = 0
    if k in node_dict:
        return node_dict[k]['children']

    sim = 1 - agg_cluster.distances_[k]

    if i < leaf_count:
        left = [tokenized[i]]
        left_vector = vectors[i]
        simL = 1
    else:
        # read the AgglomerativeClustering doc to see why we
        # select i-n_samples
        left, left_vector, simL = get_all_children(
            agg_cluster,
            i - n_samples,
            tokenized,
            vectors)

    if j < leaf_count:
        right = [tokenized[j]]
        right_vector = vectors[j]
        simR = 1
    else:
        right, right_vector, simR = get_all_children(
            agg_cluster,
            j-n_samples,
            tokenized,
            vectors)

    if verbose:
        print(k, i, j, left, right)

    all_children = [sim, left, right]

    sum_vector = np.concatenate([left_vector, right_vector])

    # store the results to speed up any additional or recursive evaluations
    node_dict[k] = {'top_child': [i, j], 'children': all_children}
    return all_children, sum_vector, sim


def flatten(list_of_lists, simList):
    """
     Take the tree output from get_all_children and restructure it
     into a list of lists format.
    """
    if type(list_of_lists) != list:
        word = str(list_of_lists.split('_')[0]).lower()
        return [[word]
                + [list_of_lists.split('_')[1]]
                + [list_of_lists.split('_')[2]]
                + simList]

    if len(list_of_lists) == 1:
        return flatten(list_of_lists[0], simList)
    if len(list_of_lists) == 2:
        return flatten(list_of_lists[0], simList) \
               + flatten(list_of_lists[1:], simList)
    simList = simList + [list_of_lists[0]]
    a = flatten(list_of_lists[1], simList)
    b = flatten(list_of_lists[2], simList)
    return a + b


def assignClusterLabels(flattened, simThreshold):
    """
     Assign tokens to clusters based on the agglomerative clustering,
     using a value just above zero on the normalized cosine scale
     as the cut point.not np.any(a)
    """
    clusterNum = 1
    lastSim = -1.0
    extended = []
    clStart = 3
    clCounts = {}
    for line in flattened:
        firstSimLoc = clStart
        simLoc = firstSimLoc
        for simLoc in range(firstSimLoc, len(line)):
            if line[simLoc] > simThreshold+.05:

                ###################################
                # Grap the cluster number and word
                if line[simLoc] != lastSim:
                    clusterNum += 1
                    lastSim = line[simLoc]
                word = line[0]

                ###################################
                # Track the frequency of words
                # in the cluster so we can check
                # if for some reason as word gets
                # put in multiple clusters.
                ###################################
                if word not in clCounts:
                    clCounts[word] = {}
                    clCounts[word][clusterNum] = 1
                else:
                    if clusterNum not in clCounts[word]:
                        clCounts[word][clusterNum] = 1
                    else:
                        clCounts[word][clusterNum] += 1

                # add the next line to our output data structure
                newline = line[0: clStart] + [clusterNum] + line[clStart:]
                extended.append(newline)
                break

    ################################################################
    # To control for tokens getting put in different clusters
    # add a column where we note for every token what its dominant
    # cluster is. That may be a better way to group for later
    # reporting.
    ################################################################
    extended2 = []
    for line in extended:
        word = line[0]
        lemma = line[1]
        if word in clCounts and wordfreq.zipf_frequency(word, 'en') < 5:
            clusterCounts = clCounts[word]
            dominantCl = max(clusterCounts, key=clusterCounts.get)
        if lemma in clCounts and wordfreq.zipf_frequency(word, 'en') < 5:
            clusterCounts = clCounts[lemma]
            dominantCl = max(clusterCounts, key=clusterCounts.get)
        else:
            dominantCl = line[clStart]
        extended2.append(
            line[0:clStart + 1]
            + [dominantCl]
            + line[clStart + 1:])

    #################################
    # Return the restructured data.
    #################################
    return extended2


def calculateClusterProfile(hdoc):
    """
      Create a cluster profile for the document as a whole.
      We order the clusters so that clusters with more distinct lemmas
      and lower word frequencies are preferred.
    """
    clusterLocs = {}
    sumClusterFreqs = {}
    clusters = {}
    clustersByWord = {}
    clustersByLemma = {}
    for token in hdoc:

        if hdoc[token.i]._.clusterID is None:
            continue

        #######################################################
        # track the word frequency of the words in the cluster
        #######################################################

        if hdoc[token.i]._.clusterID not in sumClusterFreqs:
            sumClusterFreqs[hdoc[token.i]._.clusterID] = \
                wordfreq.zipf_frequency(token.text.lower(), 'en')
        else:
            sumClusterFreqs[hdoc[token.i]._.clusterID] += \
                wordfreq.zipf_frequency(token.text.lower(), 'en')

        ################################################
        # create a list of offsets to cluster tokens   #
        ################################################
        if hdoc[token.i]._.clusterID not in clusterLocs:
            clusterLocs[hdoc[token.i]._.clusterID] = [token.i]
        else:
            clusterLocs[hdoc[token.i]._.clusterID].append(token.i)

        ##################################################################
        # Count the frequencies of clusters and track the distinct lemmas
        # in a cluster
        ##################################################################
        if hdoc[token.i]._.clusterID not in clusters:
            clusters[hdoc[token.i]._.clusterID] = 1
            clustersByLemma[hdoc[token.i]._.clusterID] = {}
            clustersByLemma[hdoc[token.i]._.clusterID][token.lemma_] = 1
        else:
            clusters[hdoc[token.i]._.clusterID] += 1
            if token.lemma_ not in clustersByLemma[hdoc[token.i]._.clusterID]:
                clustersByLemma[hdoc[token.i]._.clusterID][token.lemma_] = 1
            else:
                clustersByLemma[hdoc[token.i]._.clusterID][token.lemma_] += 1

    ######################################################################
    # upweight the clusters by number off tokens assigned to the cluster,
    # downweight by the avarage Zipfian frequency of words in the cluster
    ######################################################################

    ratings = {}
    for clusterID in clusters:
        count = clusters[clusterID]
        if count > 0:
            wfreq = sumClusterFreqs[clusterID]/count
            if wfreq != 0:
                ratings[clusterID] = count / wfreq

    #####################################################################
    # create an ordered list of cluster information (ids, lemmas in the
    # cluster, offsets)
    #####################################################################

    docClusterInfo = []
    for clusterID in OrderedDict(sorted(ratings.items(),
                                 key=lambda kv: kv[1],
                                 reverse=True)):
        if clusters[clusterID] < 3:
            continue
        docClusterInfo.append([clusterID,
                              ratings[clusterID],
                              list(clustersByLemma[clusterID].keys()),
                              clusterLocs[clusterID]])

    ###########################
    # and return that list    #
    ###########################
    return docClusterInfo


def mainClusterSpans(hdoc):
    """
     The strongest cluster in the document should be one that pretty much
     spans the whole document. Thus, the distance between cluster members
     (including the distance from the first or last of them to the left or
     right edge of the document, respectively), gives a sense of the
     extent to which the strongest cluster is consistently referred to
     throughout the whole document. (Note that the clusters include resolved
     pronominal referents clustered using the vectors of their antecedents.)
    """

    if hdoc._.clusterInfo is not None \
       and len(hdoc._.clusterInfo) > 0:
        offsets = hdoc._.clusterInfo[0][3]
        start = 0
        spans = []
        for item in offsets:
            span = item - start
            spans.append(span)
            start = span
        return spans
    else:
        return []


def developmentContentWords(hdoc):
    """
     The degree of development of the essay is reflected in
     content words not associated with the strongest clsuters
     (which reflect the common topic, not development of new
     ideas about that topic.) We approximate this by excluding
     words associated with the four strongest clusters.
    """
    wordlist = []
    if hdoc._.clusterInfo is not None \
       and len(hdoc._.clusterInfo) > 0:
        topClusters = []
        for i, clinfo in enumerate(hdoc._.clusterInfo):
            if i < 5:
                topClusters.append(clinfo[0])
            else:
                break
        for token in hdoc:
            if not token.is_stop \
               and token.text != '.' \
               and not re.match('[-.,;:"\'?/><{})(!`~_]+', token.text) \
               and not wordfreq.zipf_frequency(token.text.lower(),
                                               'en') > 5.7 \
               and token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV'] \
               and token._.clusterID not in topClusters:
                wordlist.append(token)
    return wordlist


###################################
# Register the token and document
# extensions we need to make this
# information accessible.
###################################

Token.set_extension("_tclust", default=None, force=True)
Doc.set_extension("_clinfo", default=None, force=True)


def get_tclust(token):
    return token._._tclust


def set_tclust(token, value):
    token._._tclust = value


def get_clinfo(doc):
    return doc._._clinfo


def set_clinfo(token, value):
    doc._._clinfo = value


Token.set_extension("clusterID", getter=get_tclust, force=True)
Doc.set_extension("clusterInfo", getter=get_clinfo, force=True)
Span.set_extension("clusterInfo", getter=get_clinfo, force=True)

##############################################################
# Mean number of words between instances of the primary topic
# cluster within the document (or between a cluster word and
# the document edge)
##############################################################


def mnMainClusterSpan(tokens):
    return summarize(mainClusterSpans(tokens),
                     summaryType=FType.MEAN)


Doc.set_extension("mean_main_cluster_span",
                  getter=mnMainClusterSpan,
                  force=True)


################################################################
# Median number of words between instances of the primary topic
# cluster within the document (or between a cluster word and the
# document edge)
#################################################################


def mdMainClusterSpan(tokens):
    return summarize(mainClusterSpans(tokens),
                     summaryType=FType.MEDIAN)


Doc.set_extension("median_main_cluster_span",
                  getter=mdMainClusterSpan,
                  force=True)


##################################################################
# Min number of words between instances of the primary topic
# cluster within the document (or between a cluster word and the
# document edge)
##################################################################


def minMainClusterSpan(tokens):
    return summarize(mainClusterSpans(tokens),
                     summaryType=FType.MIN)


Doc.set_extension("min_main_cluster_span",
                  getter=minMainClusterSpan,
                  force=True)


#############################################################
# Max number of words between instances of the primary topic
# cluster within the document (or between a cluster word and
# the document edge)
#############################################################

def maxMainClusterSpan(tokens):
    return summarize(mainClusterSpans(tokens),
                     summaryType=FType.MAX)


Doc.set_extension("max_main_cluster_span",
                  getter=maxMainClusterSpan,
                  force=True)


###################################################################
# St Dev of number of words between instances of the primary topic
# cluster within the document (or between a cluster word and the
# document edge)
###################################################################


def stdMainClusterSpan(tokens):
    return summarize(mainClusterSpans(tokens),
                     summaryType=FType.STDEV)


Doc.set_extension("stdev_main_cluster_span",
                  getter=stdMainClusterSpan,
                  force=True)

######################################################################
# Return list of content words that do not appear in the 4 strongest
# in-document content clusters
######################################################################


def devlist(tokens):
    devlist = [token for token in developmentContentWords(tokens)]
    return [1 if token in devlist else 0 for token in tokens]


def dcwrd(tokens):
    return devlist(tokens)


Doc.set_extension("devwords",
                  getter=dcwrd,
                  force=True)


#####################################################################
# Return proportion of words in the document that are content words
# that do not appear in the 4 strongest in-document content clusters
#####################################################################


def dcwrdlen(tokens):
    return len([x.text.lower()
                for x in developmentContentWords(tokens)
                ]) / (len(tokens) + .01)
    # +.01 to avoid errors if empty


Doc.set_extension("propn_devwords",
                  getter=dcwrdlen,
                  force=True)


###############################################################
# Mean number of syllables in content words that do not appear
# in the 4 strongest in-document content clusters
###############################################################

def mnDwrdNSyll(tokens):
    return summarize(
                     [int(x._.nSyll)
                      for x in developmentContentWords(tokens)
                      if x._.nSyll is not None],
                     summaryType=FType.MEAN)


Doc.set_extension("mean_devword_nsyll",
                  getter=mnDwrdNSyll,
                  force=True)


#################################################################
# Median number of syllables in content words that do not appear
# in the 4 strongest in-document content clusters
#################################################################


def mdDwrdNSyll(tokens):
    return summarize([int(x._.nSyll)
                      for x in developmentContentWords(tokens)
                      if x._.nSyll is not None
                      ], summaryType=FType.MEDIAN)


Doc.set_extension("median_devword_nsyll",
                  getter=mdDwrdNSyll,
                  force=True)

##############################################################
# Min number of syllables in content words that do not appear
# in the 4 strongest in-document content clusters
##############################################################


def minDwrdNSyll(tokens):
    return summarize([int(x._.nSyll)
                      for x in developmentContentWords(tokens)
                      if x._.nSyll is not None
                      ], summaryType=FType.MIN)


Doc.set_extension("min_devword_nsyll",
                  getter=minDwrdNSyll,
                  force=True)


##############################################################
# Max number of syllables in content words that do not appear
# in the 4 strongest in-document content clusters
##############################################################


def mxDwrdNSyll(tokens):
    return summarize([int(x._.nSyll)
                      for x in developmentContentWords(tokens)
                      if x._.nSyll is not None
                      ], summaryType=FType.MAX)


Doc.set_extension("max_devword_nsyll",
                  getter=mxDwrdNSyll,
                  force=True)


######################################################################
# Std dev. of number of syllables in content words that do not appear
# in the 4 strongest in-document content clusters
######################################################################


def stdDwrdNSyll(tokens):
    return summarize([int(x._.nSyll)
                      for x in developmentContentWords(tokens)
                      if x._.nSyll is not None
                      ], summaryType=FType.STDEV)


Doc.set_extension("stdev_devword_nsyll",
                  getter=stdDwrdNSyll,
                  force=True)


###############################################################
# Mean number of morphemes in content words that do not appear
# in the 4 strongest in-document content clusters
###############################################################


def mnDwrdNMorph(tokens):
    return summarize([int(x._.nMorph)
                      for x in developmentContentWords(tokens)
                      if x._.nMorph is not None
                      ], summaryType=FType.MEAN)


Doc.set_extension("mean_devword_nmorph",
                  getter=mnDwrdNMorph,
                  force=True)


#################################################################
# Median number of morphemes in content words that do not appear
# in the 4 strongest in-document content clusters
#################################################################


def mdDwrdNMorph(tokens):
    return summarize([int(x._.nMorph)
                      for x in developmentContentWords(tokens)
                      if x._.nMorph is not None
                      ], summaryType=FType.MEDIAN)


Doc.set_extension("median_devword_nmorph",
                  getter=mdDwrdNMorph,
                  force=True)


###############################################################
# Min number of morphemes in content words that do not appear
# in the 4 strongest in-document content clusters
###############################################################


def minDwrdNMorph(tokens):
    return summarize([int(x._.nMorph)
                      for x in developmentContentWords(tokens)
                      if x._.nMorph is not None
                      ], summaryType=FType.MIN)


Doc.set_extension("min_devword_nmorph",
                  getter=minDwrdNMorph,
                  force=True)


##############################################################
# Max number of morphemes in content words that do not appear
# in the 4 strongest in-document content clusters
##############################################################


def mxDwrdNMorph(tokens):
    return summarize([int(x._.nMorph)
                      for x in developmentContentWords(tokens)
                      if x._.nMorph is not None
                      ], summaryType=FType.MAX)


Doc.set_extension("max_devword_nmorph",
                  getter=mxDwrdNMorph,
                  force=True)


#####################################################################
# Std dev of number of morphemes in content words that do not appear
# in the 4 strongest in-document content clusters
#####################################################################


def stdDwrdNMorph(tokens):
    return summarize([int(x._.nMorph)
                      for x in developmentContentWords(tokens)
                      if x._.nMorph is not None
                      ], summaryType=FType.STDEV)


Doc.set_extension("stdev_devword_nmorph",
                  getter=stdDwrdNMorph,
                  force=True)


############################################################
# Mean number of senses in content words that do not appear
# in the 4 strongest in-document content clusters
############################################################


def mnDwrdNSenses(tokens):
    return summarize([int(x._.nSenses)
                      for x in developmentContentWords(tokens)
                      if x._.nSenses is not None
                      ], summaryType=FType.MEAN)


Doc.set_extension("mean_devword_nsenses",
                  getter=mnDwrdNSenses,
                  force=True)


##############################################################
# Median number of senses in content words that do not appear
# in the 4 strongest in-document content clusters
##############################################################


def mdDwrdNSenses(tokens):
    return summarize([int(x._.nSenses)
                      for x in developmentContentWords(tokens)
                      if x._.nSenses is not None
                      ], summaryType=FType.MEDIAN)


Doc.set_extension("median_devword_nsenses",
                  getter=mdDwrdNSenses,
                  force=True)


###########################################################
# Min number of senses in content words that do not appear
# in the 4 strongest in-document content clusters
###########################################################


def minDwrdNSenses(tokens):
    return summarize([int(x._.nSenses)
                      for x in developmentContentWords(tokens)
                      if x._.nSenses is not None
                      ], summaryType=FType.MIN)


Doc.set_extension("min_devword_nsenses",
                  getter=minDwrdNSenses,
                  force=True)


###########################################################
# Max number of senses in content words that do not appear
# in the 4 strongest in-document content clusters
###########################################################


def mxDwrdNSenses(tokens):
    return summarize([int(x._.nSenses)
                      for x in developmentContentWords(tokens)
                      if x._.nSenses is not None
                      ], summaryType=FType.MAX)


Doc.set_extension("max_devword_nsenses",
                  getter=mxDwrdNSenses,
                  force=True)


##################################################################
# Std dev of number of senses in content words that do not appear
# in the 4 strongest in-document content clusters
##################################################################


def stdDwrdNSenses(tokens):
    return summarize([int(x._.nSenses)
                      for x in developmentContentWords(tokens)
                      if x._.nSenses is not None
                      ], summaryType=FType.STDEV)


Doc.set_extension("stdev_devword_nsenses",
                  getter=stdDwrdNSenses,
                  force=True)


#####################################################
# Mean frequency of content words that do not appear
# in the 4 strongest in-document content clusters
#####################################################


def mnDwrdTokFreq(tokens):
    return summarize([float(x._.token_freq)
                      for x in developmentContentWords(tokens)
                      if x._.token_freq is not None
                      ], summaryType=FType.MEAN)


Doc.set_extension("mean_devword_token_freq",
                  getter=mnDwrdTokFreq,
                  force=True)


#######################################################
# Median frequency of content words that do not appear
# in the 4 strongest in-document content clusters
#######################################################


def mdDwrdTokFreq(tokens):
    return summarize([float(x._.token_freq)
                      for x in developmentContentWords(tokens)
                      if x._.token_freq is not None
                      ], summaryType=FType.MEDIAN)


Doc.set_extension("median_devword_token_freq",
                  getter=mdDwrdTokFreq,
                  force=True)


####################################################
# Min frequency of content words that do not appear
# in the 4 strongest in-document content clusters
####################################################


def minDwrdTokFreq(tokens):
    return summarize([float(x._.token_freq)
                      for x in developmentContentWords(tokens)
                      if x._.token_freq is not None
                      ], summaryType=FType.MIN)


Doc.set_extension("min_devword_token_freq",
                  getter=minDwrdTokFreq,
                  force=True)


####################################################
# Max frequency of content words that do not appear
# in the 4 strongest in-document content clusters
####################################################


def mxDwrdTokFreq(tokens):
    return summarize([float(x._.token_freq)
                      for x in developmentContentWords(tokens)
                      if x._.token_freq is not None
                      ], summaryType=FType.MAX)


Doc.set_extension("max_devword_token_freq",
                  getter=mxDwrdTokFreq,
                  force=True)


###########################################################
# Std dev of frequency of content words that do not appear
# in the 4 strongest in-document content clusters
###########################################################


def stdDwrdTokFreq(tokens):
    return summarize([float(x._.token_freq)
                      for x in developmentContentWords(tokens)
                      if x._.token_freq is not None
                      ], summaryType=FType.STDEV)


Doc.set_extension("stdev_devword_token_freq",
                  getter=stdDwrdTokFreq,
                  force=True)


########################################################
# Mean concreteness of content words that do not appear
# in the 4 strongest in-document content clusters
########################################################


def mnDwrdConcr(tokens):
    return summarize([float(x._.concreteness)
                      for x in developmentContentWords(tokens)
                      if x._.concreteness is not None
                      ], summaryType=FType.MEAN)


Doc.set_extension("mean_devword_concreteness",
                  getter=mnDwrdConcr,
                  force=True)


##########################################################
# Median concreteness of content words that do not appear
# in the 4 strongest in-document content clusters
##########################################################


def mdDwrdConcr(tokens):
    return summarize([float(x._.concreteness)
                      for x in developmentContentWords(tokens)
                      if x._.concreteness is not None
                      ], summaryType=FType.MEDIAN)


Doc.set_extension("median_devword_concreteness",
                  getter=mdDwrdConcr,
                  force=True)


#######################################################
# Min concreteness of content words that do not appear
# in the 4 strongest in-document content clusters
#######################################################


def minDwrdConcr(tokens):
    return summarize([float(x._.concreteness)
                      for x in developmentContentWords(tokens)
                      if x._.concreteness is not None
                      ], summaryType=FType.MEDIAN)


Doc.set_extension("min_devword_concreteness",
                  getter=mdDwrdConcr,
                  force=True)


#######################################################
# Max concreteness of content words that do not appear
# in the 4 strongest in-document content clusters
#######################################################

def mxDwrdConcr(tokens):
    return summarize([float(x._.concreteness)
                      for x in developmentContentWords(tokens)
                      if x._.concreteness is not None
                      ], summaryType=FType.MAX)


Doc.set_extension("max_devword_concreteness",
                  getter=mxDwrdConcr,
                  force=True)


##############################################################
# Std dev of concreteness of content words that do not appear
# in the 4 strongest in-document content clusters
##############################################################


def stdDwrdConcr(tokens):
    return summarize([float(x._.concreteness)
                      for x in developmentContentWords(tokens)
                      if x._.concreteness is not None
                      ], summaryType=FType.STDEV)


Doc.set_extension("stdev_devword_concreteness",
                  getter=stdDwrdConcr,
                  force=True)
