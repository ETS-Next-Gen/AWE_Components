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

def allClusterInfo(hdoc):
    """
    """
    # check for clusterInfo, not clusterInfo_,
    # to force clustering to run if it hasn't yet
    if hdoc._.clusterInfo is not None \
       and len(hdoc._.clusterInfo_) > 0:
        clusterSpans = []
        for cluster in hdoc._.clusterInfo_:
            offsets = cluster[3]
            spans = []
            for item in offsets:
                entry={}
                entry['name'] = 'clusterinfo'
                entry['offset'] = hdoc[item].idx
                entry['startToken'] = item
                entry['endToken'] = item
                entry['text'] = json.dumps(cluster[2])
                entry['value'] = (cluster[0], cluster[1])
                entry['length'] = len(hdoc[item].text_with_ws)
                clusterSpans.append(entry)
        return clusterSpans
    else:
        return []

@Language.component("lexicalclusters")
def assignClusterIDs(hdoc):
    '''
        Use agglomerative clustering to group the words inside a document
        into clusters of related words
    '''
    # By default we don't assign cluster IDs
    # so it will just pass the document through
    # unless we have made clusterInfo_ non None first.
    if hdoc._.clusterInfo_ is None:
        return hdoc

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
                hdoc[int(locus)]._.clusterID_ = clusterID

            ###########################################################
            # restructure the data into a form that can be associated #
            # with the document as a whole and associate that record  #
            # with the spacy document                                 #
            ###########################################################
            hdoc._.clusterInfo_ = calculateClusterProfile(hdoc)
            mainClusterSpans(hdoc)

    except Exception as e:
        print('clustering error\n', e)
        hdoc._.clusterInfo_ = None

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

        if hdoc[token.i]._.clusterID_ is None:
            continue

        #######################################################
        # track the word frequency of the words in the cluster
        #######################################################

        if hdoc[token.i]._.clusterID_ not in sumClusterFreqs:
            sumClusterFreqs[hdoc[token.i]._.clusterID_] = \
                wordfreq.zipf_frequency(token.text.lower(), 'en')
        else:
            sumClusterFreqs[hdoc[token.i]._.clusterID_] += \
                wordfreq.zipf_frequency(token.text.lower(), 'en')

        ################################################
        # create a list of offsets to cluster tokens   #
        ################################################
        if hdoc[token.i]._.clusterID_ not in clusterLocs:
            clusterLocs[hdoc[token.i]._.clusterID_] = [token.i]
        else:
            clusterLocs[hdoc[token.i]._.clusterID_].append(token.i)

        ##################################################################
        # Count the frequencies of clusters and track the distinct lemmas
        # in a cluster
        ##################################################################
        if hdoc[token.i]._.clusterID_ not in clusters:
            clusters[hdoc[token.i]._.clusterID_] = 1
            clustersByLemma[hdoc[token.i]._.clusterID_] = {}
            clustersByLemma[hdoc[token.i]._.clusterID_][token.lemma_] = 1
        else:
            clusters[hdoc[token.i]._.clusterID_] += 1
            if token.lemma_ not in clustersByLemma[hdoc[token.i]._.clusterID_]:
                clustersByLemma[hdoc[token.i]._.clusterID_][token.lemma_] = 1
            else:
                clustersByLemma[hdoc[token.i]._.clusterID_][token.lemma_] += 1

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

    if hdoc._.clusterInfo_ is not None \
       and len(hdoc._.clusterInfo_) > 0:
        offsets = hdoc._.clusterInfo_[0][3]
        start = 0
        spans = []
        hdoc._.main_cluster_spans_ = []
        for item in offsets:
            entry={}
            entry['name'] = 'clusterspan'
            entry['offset'] = hdoc[start].idx
            entry['startToken'] = start
            entry['endToken'] = item
            entry['text'] = None
            span = item - start
            entry['value'] = span
            entry['length'] = hdoc[item].idx \
                              + len(hdoc[item].text_with_ws) \
                              - hdoc[start].idx
            spans.append(span)
            start = span
        hdoc._.main_cluster_spans_.append(entry)
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
    if hdoc._.clusterInfo_ is not None \
       and len(hdoc._.clusterInfo_) > 0:
        topClusters = []
        for i, clinfo in enumerate(hdoc._.clusterInfo_):
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
               and token.pos_ \
                   in ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV'] \
               and token._.clusterID_ not in topClusters:
                wordlist.append(token)
    return wordlist

def devword(token):
    '''
       Return list of content words that do not appear in the 4 strongest
       in-document content clusters
    '''
    if token.doc.clusterInfo_ is None:
        # flag assignClusterIDs to run
        # by setting it to a non None value
        token.doc._.clusterInfo_ = []
        self.assignClusterIDs(token.doc)
    devlist = [token.text \
               for token \
               in developmentContentWords(token.doc)]
    if token.text in devlist:
        return True
    else:
        return False    

###################################
# Register the token and document
# extensions we need to make cluster
# information accessible.
###################################
Token.set_extension("devword", getter=devword, force=True)

# Storage locations for precalculated data
Token.set_extension("clusterID_",
                    default=None,
                    force=True)

Doc.set_extension("clusterInfo_",
                  default=None,
                  force=True)

Doc.set_extension("main_cluster_spans_",
                  default=None,
                  force=True)

# Getter functions for attributes used externally

# set AWE_Info if this is the first
# component to call for it
if not Doc.has_extension('AWE_Info'):
    Doc.set_extension('AWE_Info',
                      method=AWE_Info)

def clusterID(token):
    if token.doc.clusterInfo_ is None:
        # flag assignClusterIDs to run
        # by setting it to a non None value
        token.doc._.clusterInfo_ = []
        assignClusterIDs(token.doc)
    return token._.clusterID_

Token.set_extension('clusterID',
                    getter=clusterID,
                    force=True)

def clusterInfo(doc):
    if doc._.clusterInfo_ is None:
        # flag assignClusterIDs to run
        # by setting it to a non None value
        doc._.clusterInfo_ = []
        assignClusterIDs(doc)
    return doc._.clusterInfo_

Doc.set_extension('clusterInfo',
                  getter=clusterInfo,
                  force=True)

def main_cluster_spans(doc):
    if doc._.clusterInfo_ is None:
        # flag assignClusterIDs to run
        # by setting it to a non None value
        doc._.clusterInfo_ = []
        assignClusterIDs(doc)
    return doc._.main_cluster_spans_

Doc.set_extension('main_cluster_spans',
                  getter=main_cluster_spans,
                  force=True)

Doc.set_extension("all_cluster_info",
                  getter=allClusterInfo,
                  force=True)
