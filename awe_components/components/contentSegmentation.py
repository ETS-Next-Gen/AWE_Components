#!/usr/bin/env python3
# Copyright 2022, Educational Testing Service

from .utility_functions import *
from operator import itemgetter
from spacy.tokens import Doc
from spacy.language import Language
import wordfreq

@Language.component("contentsegmentation")
def contentsegmentation(doc):

    # If we haven't run it yet, don't.
    # We'll reset main_ideas_ to a non None
    # value when we want to process the doc
    # for content segments
    if doc._.main_ideas_ is None:
        return doc

    if doc._.clusterInfo is not None:
        core_sentences, \
            extended_core_sentences, \
            elaboration_sentences, \
            pclusters, \
            plemmas = extract_content_segments(None, doc)

        doc._.main_ideas_ = core_sentences
        doc._.supporting_ideas_ = extended_core_sentences
        doc._.supporting_details_ = elaboration_sentences
        doc._.prompt_related_ = pclusters
        doc._.prompt_language_ = plemmas
    return doc


def countDetails(start, end, doc, plemmas):
    segStart = start
    segEnd = end
    segLoc = segStart
    detailCount = 0
    covered = []
    while segLoc < segEnd and segLoc < end:
        if doc[segLoc].lemma_ not in plemmas \
          and doc[segLoc].lower_ not in covered \
          and doc[segLoc]._.max_freq is not None \
          and doc[segLoc]._.max_freq > 1.75 \
          and doc[segLoc]._.max_freq < 5 \
          and not doc[segLoc]._.is_academic \
          and doc[segLoc].pos_ in ['NOUN', 'PROPN', 'ADJ']:
            covered.append(doc[segLoc].lower_)
            detailCount += 1
        elif doc[segLoc]._.transition \
            and doc[segLoc]._.transition_category in ['temporal',
                                                      'illustrative',
                                                      'comparative',
                                                      'periphrastic']:
            detailCount += 1
        elif doc[segLoc]._.vwp_quoted:
            detailCount += 1
        elif doc[segLoc]._.vwp_source:
            detailCount += 1
        elif doc[segLoc]._.vwp_attribution:
            detailCount += 1
        elif doc[segLoc]._.vwp_cite:
            detailCount += 1
        elif doc[segLoc].ent_type_ \
            in ['WORK_OF_ART',
                'PERSON',
                'GPE',
                'PRODUCT',
                'NORP',
                'LAW',
                'LOC',
                'ORG'] \
                and doc[segLoc]._.max_freq is not None \
                and doc[segLoc]._.max_freq < 5:
            detailCount += 1
        segLoc += 1
    return detailCount


def extract_content_segments(prompt, doc):

    # If no prompt has been provided,
    # assume that the dominant clusters with
    # many repeated words spanning the text
    # define the core topic.
    pclusters = []
    rclusters = []
    embargoed = []
    embargoed_lemmas = []
    if prompt is None:
        hcount = 0

        # pick the top three clusters for this purpose
        for cluster in doc._.clusterInfo:
            if len(cluster) > 2 and len(cluster[3]) > 3:
                start = cluster[3][0]
                end = cluster[3][len(cluster[3])-1]
                if (end-start)*1.0 / len(doc) > 0.6:
                    pclusters.append(cluster)
                    rclusters.append(cluster)
                    hcount += 1
                    if hcount > 2:
                        break

        # and put the words in these clusters into our list of
        # prompt words
        pwrds = []
        plemmas = []
        proots = []
        for item in rclusters:
            for wrd in item[2]:
                pwrds.append(wrd)

        # define word frequencies in the doc
        freqs = {}
        for token in doc:
            if token._.root not in freqs:
                freqs[token._.root] = 1
            else:
                freqs[token._.root] += 1

        # record the words from the selected topics plus argument words
        # as key content
        for token in doc:
            if token._.root is not None \
               and not token.is_stop \
               and not token._.vwp_evaluation \
               and not token._.vwp_hedge \
               and token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV'] \
               and token._.root not in plemmas \
               and (token._.root in pwrds
                    or token.lemma_ in pwrds
                    or token.lower_ in pwrds
                    ) \
               and freqs[token._.root] > 1:
                if token._.root not in plemmas:
                    plemmas.append(token._.root)
                if token.lemma_ not in plemmas:
                    plemmas.append(token.lemma_)
                if token.lower_ not in plemmas:
                    plemmas.append(token.lower_)
            if token._.transition \
               and token._.transition_category not in ['illustrative',
                                                       'temporal',
                                                       'comparative',
                                                       'crossreferential',
                                                       'general'] \
               and token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']:
                if token._.root not in plemmas:
                    plemmas.append(token._.root)
                if token.lemma_ not in plemmas:
                    plemmas.append(token.lemma_)
                if token.lower_ not in plemmas:
                    plemmas.append(token.lower_)
            elif (token._.vwp_argument
                  or token._.vwp_argue
                  or token._.vwp_evaluation
                  ) \
                and not token.is_stop \
                and not token._.transition \
                and not token._.vwp_hedge \
                and (token.dep_ == 'ROOT' or token.head.dep_ == 'ROOT') \
                    and token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']:
                if token._.root not in plemmas:
                    plemmas.append(token._.root)
                if token.lemma_ not in plemmas:
                    plemmas.append(token.lemma_)
                if token.lower_ not in plemmas:
                    plemmas.append(token.lower_)
    else:

        # If we have a prompt text, we sort out which words in the
        # prompt we can use to define key points in the text

        # first we get the roots and lemmas
        plemmas = [tok.lemma_ for tok in prompt
                   if tok.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV']
                   and not tok.is_stop and not tok._.transition]
        proots = [tok._.root for tok in prompt
                  if tok.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV']
                  and not tok.is_stop and not tok._.transition]
        for token in doc:
            if token._.root in plemmas \
               and token.lemma_ not in plemmas \
               and token._.root != token.lemma_:
                plemmas.append(token.lemma_)
        # then embargo prompt language and argument language from
        # being used as content later
        docargs = [tok.lemma_ for tok in doc if tok._.vwp_argumentation]
        for lemma in plemmas:
            if lemma not in embargoed_lemmas:
                embargoed_lemmas.append(lemma)

        for root in proots:
            if root not in embargoed_lemmas:
                embargoed_lemmas.append(root)

        for lemma in docargs:
            if lemma not in embargoed_lemmas:
                embargoed_lemmas.append(lemma)

        # identify clusters drawing on prompt-related vocabulary
        for item in doc._.clusterInfo:
            for wrd in item[2]:
                if (wrd in plemmas or wrd in proots) and item not in pclusters:
                    pclusters.append(item)
                    if item not in embargoed:
                        embargoed.append(item)

    ####################################
    # Common processing for both cases #
    ####################################

    # Identify sentences with significant content overlap with prompt language
    core_sentences = []
    core_offsets = []
    extended_core_offsets = []
    flat_list = []
    cw = []
    for chain in pclusters:
        flat_list += chain[3]
    for item in flat_list:
        cw.append(item)

    for sent in doc.sents:
        pcount = 0
        prcount = 0
        prloose = 0
        attrcount = 0
        refcount = 0
        for tok in sent:
            # cross-sentence anaphora means that this sentence is part of
            # a local content chain and is subordinate to a previous sentence
            if tok._.coref_chains is not None \
               and tok.lower_ in ['he',
                                  'him',
                                  'she',
                                  'her',
                                  'it',
                                  'they',
                                  'them']:
                offset = None
                for chain in tok._.coref_chains:
                    for mentionIndex, item in enumerate(chain):
                        offset = item[0]
                        if offset < tok.sent.start:
                            refcount += 1
                        break
            # Immediate back reference using this/that + N in main clause
            if tok.lower_ in ['this', 'that'] \
               and tok.dep_ != 'mark' \
               and (tok.head.dep_ in ['nsubj', 'dobj']
                    or tok.dep_ in ['nsubj', 'dobj']) \
               and tok.i - tok.sent.start < 4.5:
                refcount += 1
            if tok.is_stop or tok._.vwp_evaluation or tok._.vwp_hedge \
               or tok.pos_ not in ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV']:
                continue
            if tok.lemma_ in plemmas:
                pcount += 1
            elif match_related_form(tok, plemmas):
                prcount += 1
            if tok._.vwp_attribution:
                attrcount += 1
        if (pcount > 0 and prcount > 1) or pcount > 1:
            root = getRoot(doc[sent.start])
            if not in_past_tense_scope(root) \
               and attrcount == 0 \
               and refcount == 0:
                i = sent.start
                while i <= sent.end:
                    core_offsets.append(i)
                    i += 1
                    
                entry = \
                    newSpanEntry("main_ideas",
                                 sent.start,
                                 sent.end-1,
                                 doc,
                                 sent.end - sent.start)
                core_sentences.append(entry)
    # Identify sequences of overlapping content chains that suggest
    # supporting points linked to the core content
    chains = []
    newcl = []
    covered = []
    # include coreferentiality chains in the analysis
    for ch in doc._.coref_chains:
        newchain = []
        for mention_index, mention in enumerate(ch):
            for offset in mention:
                newchain.append(offset)
        chains.append(newchain)
    for cluster in doc._.clusterInfo:
        if len(cluster) > 2 and len(cluster[3]) > 3:
            start = cluster[3][0]
            end = cluster[3][len(cluster[3]) - 1]
            offsets = cluster[3]
            newcl = []
            lastItem = 0
            for item in offsets:
                if lastItem is None:
                    newcl.append(item)
                if (item-lastItem < 30
                   or (lastItem is not None
                       and (doc[item].sent.start == doc[lastItem].sent.start
                            or doc[item].sent.start <= doc[
                                lastItem].sent.end + 1))) \
                   and (item-lastItem < 15
                        or doc[item].sent.start == doc[lastItem].sent.start
                        or doc[item].sent.start <= doc[lastItem].sent.end+1) \
                   and lastItem != 0 \
                   and (item-lastItem < 12
                        or '\n' not in doc[lastItem:item].text):
                    if lastItem not in newcl \
                       and item not in embargoed \
                       and not doc[item]._.transition:
                        newcl.append(lastItem)
                        covered.append(lastItem)
                    if item not in newcl \
                       and item not in embargoed \
                       and not doc[item]._.transition:
                        newcl.append(item)
                        covered.append(item)
                else:
                    firstSentenceStart = 0
                    lastSentenceStart = 0
                    if len(newcl) > 0:
                        firstSentenceStart = doc[newcl[0]].sent.start
                        lastSentenceStart = \
                            doc[newcl[len(newcl) - 1]].sent.start
                    if len(newcl) > 0 \
                       and firstSentenceStart != lastSentenceStart:
                        chains.append(newcl)
                    newcl = []
                lastItem = item
            if len(newcl) > 0 and firstSentenceStart != lastSentenceStart:
                chains.append(newcl)
    # sort the chains so that they are in ascending order
    # by position of the first word in the chain in the text
    chains = sorted(chains, key=itemgetter(0))

    # figure out which chains overlap enough to be treated as
    # part of the same segment. Break chains if the gap between
    # members is too large.
    last = None
    segments = []
    currentseg = []
    for item in chains:
        if last is None:
            currentseg.append(item)
        elif last is not None \
            and (doc[item[0]].sent.start <= doc[last].sent.start
                 or item[len(item) - 1] - 12 < doc[last].sent.end):
            currentseg.append(item)
        else:
            # We need support from more than one cluster to
            # add segments to the extended core.
            if len(currentseg) > 1:
                segments.append(currentseg)
            currentseg = [item]
        if last is None or last < item[len(item) - 1]:
            last = item[len(item) - 1]
    if currentseg not in segments and len(currentseg) > 1:
        segments.append(currentseg)

    # Process supporting point segments to extract the start of the
    # first sentence in the segment and the end of the last
    # sentence in the segment
    prefinalSegmentData = []
    segmentChains = []
    for segment in segments:
        minSentStart = None
        maxSentEnd = None
        noncore_chain = []
        # record the content words that were in the chains
        for chain in segment:
            for item in chain:
                if minSentStart is None \
                   or doc[item].sent.start <= minSentStart:
                    minSentStart = doc[item].sent.start+1
                    if doc[minSentStart].text \
                       in ['.', '!', '?', ';', '\n', '\n\n']:
                        minSentStart += 1
                if maxSentEnd is None or doc[item].sent.end >= maxSentEnd:
                    maxSentEnd = doc[item].sent.end
                    if maxSentEnd == len(doc):
                        maxSentEnd -= 1
                    if doc[maxSentEnd].text in ['\n', '\n\n']:
                        maxSentEnd -= 1
        prefinalSegmentData.append([minSentStart, maxSentEnd])
        segmentChains = []

    # Clean up so that the secondary segments are disjoint from
    # the core sentences.
    elaboration_sentences = []
    extended_core_sentences = []
    extended_core_offsets = core_offsets
    lastSegEnd = 0
    for segment in prefinalSegmentData:
        newSegment = []
        # Get beginning and end of segments
        start = segment[0] - 1
        end = segment[1]
        if start < 0:
            start = 0
        while end >= len(doc):
            end -= 1

        # Adjust to eliminate overlaps with core sentences
        loc = start
        while loc <= end:
            while loc in core_offsets and loc < end:
                loc += 1
            if len(newSegment) == 0:
                newSegment.append(loc)
            while loc not in core_offsets and loc < end:
                loc += 1
                if len(newSegment) > 1:
                    newSegment = newSegment[:-1]
                newSegment.append(loc)

                # divide segments at transitions
                if loc == doc[loc].sent.start \
                   and doc[loc - 1].text in ['.',
                                             ';',
                                             '!',
                                             '?',
                                             '\n',
                                             '\n\n'] \
                   and (doc[loc]._.transition
                        or loc > 0 and doc[loc - 1]._.transition):
                    prons = ['it', 'he', 'she', 'they', 'him', 'her', 'them']
                    if doc[loc + 1].lower_ not in prons \
                       and doc[loc + 2].lower_ not in prons \
                       and doc[loc + 3].lower_ not in prons \
                       and doc[loc + 4].lower_ not in prons \
                       and doc[loc + 5].lower_ not in prons:
                        break

            if len(newSegment) > 1:
                # Check whether this segment contains any non prompt word
                # rare enough to provide anything anyone would call a detail
                detailCount = countDetails(newSegment[0],
                                           newSegment[1],
                                           doc,
                                           plemmas)
                if detailCount > 1 \
                   and detailCount * 1.0 / \
                   (newSegment[1] - newSegment[0]) > .03 \
                   and (doc[newSegment[0]]._.transition is None
                        or not doc[newSegment[0]]._.transition
                        or not doc[newSegment[0]]._.transition_category
                        in ['ordinal', 'emphatic']) \
                   and (doc[newSegment[0] - 1]._.transition is None
                        or not doc[newSegment[0] - 1]._.transition
                        or not doc[newSegment[0] - 1]._.transition_category
                        in ['ordinal', 'emphatic']):
                    startLoc = newSegment[0]-1
                    if startLoc < 0:
                        startLoc += 1
                    if doc[startLoc].text in ['.', '?', '!']:
                        startLoc += 1
                    endLoc = newSegment[1]
                    if endLoc == len(doc) - 1:
                        endLoc += 1
                    elaboration_sentences.append([startLoc, endLoc])
                    detailLoc = newSegment[0]
                    if detailLoc < lastSegEnd:
                        detailLoc = lastSegEnd + 1
                    detailEnd = newSegment[1]
                    lastSegEnd = detailEnd
                    while detailLoc < detailEnd and detailLoc <= end:
                        extended_core_offsets.append(detailLoc)
                        detailLoc += 1
            newSegment = []
            loc += 1

    extras = []
    conFlag = False
    for sent in doc.sents:
        if sent.start + 1 not in extended_core_offsets:
            detailCount = countDetails(sent.start, sent.end, doc, plemmas)
            current = [sent.start, sent.end]
            if detailCount > 1 \
               and detailCount * 1.0 / (sent.end - sent.start) > .03 \
               or (conFlag
                   or (doc[sent.start]._.transition is not None
                       and doc[sent.start]._.transition
                       and doc[sent.start]._.transition_category in [
                           'illustrative',
                           'temporal',
                           'contrastive',
                           'periphrastic'])):
                conFlag = True
                extras.append(current)
            else:
                extended_core_sentences.append(current)
        else:
            conFlag = False
    final_elaboration = []
    for i, item in enumerate(elaboration_sentences):
        while len(extras) > 0 and item[0] > extras[0][0]:
            current = extras[0]
            if i < len(elaboration_sentences):
                next = elaboration_sentences[i]
                if next[0] < current[1]:
                    current = [current[0], next[0]]
            final_elaboration.append(current)
            extras = extras[1:]
        final_elaboration.append(item)
    if len(extras) > 0:
        for item in extras:
            final_elaboration.append(item)

    # restructure extended_core_sentences and final_elaboration
    # using the span format
    newEC = []
    for item in extended_core_sentences:
        newEntry =  newSpanEntry('supporting_ideas',
                                 item[0],
                                 item[1]-1,
                                 doc,
                                 item[1] - item[0])
        newEC.append(newEntry)
    extended_core_sentences = newEC
            
    new_final = []
    for item in final_elaboration:
        newEntry =  newSpanEntry('supporting_details',
                                 item[0],
                                 item[1]-1,
                                 doc,
                                 item[1] - item[0])
        new_final.append(newEntry)
    final_elaboration = new_final
        
    # We removed rare words appearing very few times from the
    # plemmas list for the purpose of establishing chains of
    # linked content. Add those words now, to report as prompt-
    # related vocabulary.
    for cluster in pclusters:
        for wrd in cluster[2]:
            if wordfreq.zipf_frequency(wrd, "en") < 5 \
               and wrd not in plemmas:
                   plemmas.append(wrd)
        
    # clean up to eliminate words that are not true content-focused
    # words from the plemmas display
    for token in doc:
        if token.lower_ in plemmas \
           and (token.is_stop
                or token._.vwp_evaluation
                or token._.vwp_hedge
                or token._.transition
                or token._.vwp_argue
                or token._.vwp_argument):
            plemmas.remove(token.lower_)
        if token.lemma_ in plemmas \
           and (token.is_stop
                or token._.vwp_evaluation
                or token._.vwp_hedge
                or token._.transition
                or token._.vwp_argue
                or token._.vwp_argument):
            plemmas.remove(token.lemma_)
        if token._.root in plemmas \
           and (token.is_stop
                or token._.vwp_evaluation
                or token._.vwp_hedge
                or token._.transition
                or token._.vwp_argue
                or token._.vwp_argument):
            plemmas.remove(token._.root)

    return core_sentences, extended_core_sentences, \
        final_elaboration, pclusters, plemmas


def prompt(self, doc):
     if doc._.prompt_ is None:
         # Flag that we need to process the doc
         # for content segments
         doc._.main_ideas_ = []
         contentsegmentation(doc)     
     return doc._.prompt_


def prompt_language(doc):
     if doc._.prompt_language_ is None:
         # Flag that we need to process the doc
         # for content segments
         doc._.main_ideas_ = []
         contentsegmentation(doc)     
     return doc._.prompt_language_


def prompt_related(doc):
     if doc._.prompt_related_ is None:
         # Flag that we need to process the doc
         # for content segments
         doc._.main_ideas_ = []
         contentsegmentation(doc)     
     return doc._.prompt_related_


def main_ideas(doc):
     if doc._.main_ideas_ is None:
         # Flag that we need to process the doc
         # for content segments
         doc._.main_ideas_ = []
         contentsegmentation(doc)     
     return doc._.main_ideas_


def supporting_ideas(doc):
     if doc._.supporting_ideas_ is None:
         # Flag that we need to process the doc
         # for content segments
         doc._.main_ideas_ = []
         contentsegmentation(doc)     
     return doc._.supporting_ideas_


def supporting_details(doc):
     if doc._.supporting_details_ is None:
         # Flag that we need to process the doc
         # for content segments
         doc._.main_ideas_ = []
         contentsegmentation(doc)     
     return doc._.supporting_details_

# set AWE_Info if this is the first
# component to call for it
if not Doc.has_extension('AWE_Info'):
    Doc.set_extension('AWE_Info',
                      method=AWE_Info)

Doc.set_extension("prompt_",
                  default=None,
                  force=True)

Doc.set_extension("prompt_language_",
                  default=None,
                  force=True)

Doc.set_extension("prompt_related_",
                  default=None,
                  force=True)

Doc.set_extension("main_ideas_",
                  default=None,
                  force=True)

Doc.set_extension("supporting_ideas_",
                  default=None,
                  force=True)

Doc.set_extension("supporting_details_",
                  default=None,
                  force=True)

Doc.set_extension("prompt",
                  getter=prompt,
                  force=True)

Doc.set_extension("prompt_language",
                  getter=prompt_language,
                  force=True)

Doc.set_extension("prompt_related",
                  getter=prompt_related,
                  force=True)

Doc.set_extension("main_ideas",
                  getter=main_ideas,
                  force=True)

Doc.set_extension("supporting_ideas",
                  getter=supporting_ideas,
                  force=True)

Doc.set_extension("supporting_details",
                  getter=supporting_details,
                  force=True)
