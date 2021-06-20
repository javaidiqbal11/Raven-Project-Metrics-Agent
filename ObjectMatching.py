from PIL import Image
import numpy as np
import copy
# import ipdb

from AttributeTypes import vocab, weights
from SemanticNetwork import SemEdge


def matchObjectFeatures(fMatA, fMatB):
    if (len(fMatA[1]) == 0 and len(fMatB[1]) == 0):
        return {}
    elif (len(fMatA[1]) == 0):
        return {-1: fMatB[1]}
    elif (len(fMatB[1]) == 0):
        return {obj: -1 for obj in fMatA[1]}

    objsA = fMatA[0]
    objsB = fMatB[0]
    featuresFigA = fMatA[1]
    featuresFigB = fMatB[1]
    # compute pairwise square differences between each feature vec

    sqDiffs = None
    try:
        sqDiffs = (featuresFigA[:, np.newaxis, :] - featuresFigB[np.newaxis, :, :]) ** 2
    except:
        ipdb.set_trace()

    #  apply weights to square differences and sqrt to get weighted pairwise dist mat
    wPDist = (sqDiffs * weights).sum(axis=2) ** 0.5

    confSortInd = None
    pairings = np.zeros((0, 2))
    aliasPair = {}

    if (len(fMatB[0]) == 1):
        # ipdb.set_trace()
        # matching several objs/features to 1 obj/feature
        confSortInd = np.argsort(wPDist, axis=0)
        # optPair = np.asarray([confSortInd[0,0], 0])
        aliasPair[objsB[0]] = objsA[confSortInd[0, 0]]

        # pairings = np.vstack((pairings, optPair))
        for i in range(1, len(confSortInd)):
            # pairings = np.vstack((pairings, np.asarray([confSortInd[i,0], -1])))
            if not -1 in aliasPair:
                aliasPair[-1] = []
            # ipdb.set_trace()
            aliasPair[-1].append(objsA[confSortInd[i, 0]])

    else:
        # ipdb.set_trace()
        # sort by relative confidence to get optimal matching
        # sort rows to find object matches
        wPDist = np.sort(wPDist, axis=1)
        wPDistInd = np.argsort(wPDist, axis=1)

        # Ratio Test for confidence (|1st closest neighbor| / |2nd closest neighbor|)
        ratioTest = wPDist[:, 0] / wPDist[:, 1]

        # the larger the difference between the closest object pairing and teh 2nd closest object pairing indicates
        # confidence
        confSortInd = np.argsort(ratioTest)

        # ipdb.set_trace()
        exhaustedObjB = set()

        for indA in confSortInd:
            if len(exhaustedObjB) < len(fMatB[0]):
                for indB in wPDistInd[indA]:
                    if (indB not in exhaustedObjB):
                        aliasPair[objsB[indB]] = objsA[indA]
                        # pairings = np.vstack((pairings, np.asarray([indA, indB])))
                        exhaustedObjB.add(indB)
                        break;
            else:
                # pairings = np.vstack((pairings, np.asarray([confSortInd[indA], -1])))
                if (-1 not in aliasPair): aliasPair[-1] = []
                aliasPair[-1].append(objsA[confSortInd[indA]])

        if len(exhaustedObjB) < len(fMatB[0]):
            for i in range(0, len(fMatB[0])):
                if i not in exhaustedObjB:
                    # pairings = np.vstack((pairings, np.asarray([-1, i])))
                    aliasPair[objsB[i]] = -1

    # ipdb.set_trace()
    # alias pair not contains all objects in B mapped to their siblings in A
    return aliasPair


def constructFigureFeatureMatrices(problem):
    # do pairwise distance for each entry in two figures respective matrices

    problemFeatureMatrices = {}

    for fName, fig in problem.figures.items():

        # for each figure, np concat into an objects x features matrix
        objNameList = []
        figFeatMatrix = np.empty((0, len(weights)))

        for oName, oAttrs in fig.objects.items():
            oAttrs.attributes = cleanFormatting(oAttrs.attributes)

            featureVec = vectorize(oAttrs.attributes)

            # add on each object vec to feature mat
            figFeatMatrix = np.vstack((figFeatMatrix, featureVec))

            objNameList.append(oName)

        problemFeatureMatrices[fName] = (objNameList, figFeatMatrix)

    # ipdb.set_trace()
    return problemFeatureMatrices


def vectorize(attr):
    vec = []

    # go through each word in my vocab
    for word, numRepr in vocab.items():

        if (not word in attr):
            # null case
            if isinstance(numRepr, dict):
                vec += numRepr['null']
            else:
                vec += [-1]


        elif (len(numRepr) == 0):
            # attribute is already a num (ie: angle)
            vec.append(attr[word])
        else:
            # attribute needs to be represented numerically
            # ipdb.set_trace()
            vec += numRepr if isinstance(numRepr, list) else numRepr[attr[word]]

    # ipdb.set_trace()
    return np.asarray(vec, dtype=float)


def matchObjects(problem, rel, fMats):
    # debug = problem.name == 'Basic Problem B-12'
    debug = False

    aliasPairRef = {}

    maxObjs = -1
    for node0, neighbors in rel.items():
        # get maximum number of objects in each node
        maxObjs = max(len(problem.figures[node0].objects), maxObjs)
        for node1 in neighbors:
            objs1 = len(problem.figures[node1].objects)
            maxObjs = max(objs1, maxObjs)

            key = node0 + node1

            # aliasPair = matchObjectFeatures(fMats[node0], fMats[node1]) # better for vector descriptors of visual image
            aliasPair = extractAliasPairing((problem.figures[node0], problem.figures[node1]),
                                            debug)  # better for attribute dict descriptors of verbal image

            aliasPairRef[key] = aliasPair
    return (aliasPairRef, maxObjs)


# matching shapes
def extractAliasPairing(figures, debug):
    # print('extracted Object Map')
    aliasPair = {}
    fig0 = None
    fig1 = None
    invert = False

    if (len(figures[0].objects) < len(figures[1].objects)):
        fig0 = copy.deepcopy(figures[0].objects)
        fig1 = copy.deepcopy(figures[1].objects)
    else:
        fig0 = copy.deepcopy(figures[1].objects)
        fig1 = copy.deepcopy(figures[0].objects)
        invert = True

    for name0, obj0 in fig0.items():
        if not fig1:
            # if we've exhausted the result figure's objects and there's still something here
            if -1 not in aliasPair:
                aliasPair[-1] = []
            aliasPair[-1].append(name0)
            continue

        minDiffScore = None
        bestPairedName = None
        o0attrs = obj0.attributes

        for name1, obj1 in fig1.items():
            diffScore = 0

            o1attrs = obj1.attributes

            for attr0 in o0attrs:
                if attr0 in o1attrs:
                    diffScore -= 1.25
                    if o0attrs[attr0] == o1attrs[attr0]:
                        diffScore -= 2
                    # elif attr0 in SemEdge.edgeTerms:
                    #     diffScore -= 2 if len(o0attrs[attr0]) == len(o1attrs[attr0]) else 0
                else:
                    diffScore += 0.5
            for attr1 in o1attrs:
                if not attr1 in o0attrs:
                    diffScore += 0.5

            # update best diff score if applicable
            if minDiffScore == None or diffScore < minDiffScore:  # if we find an object pairing with few diffs update our objectMap
                minDiffScore = diffScore
                bestPairedName = name1

        aliasPair[bestPairedName] = name0
        del fig1[bestPairedName]
    # gone through all of the initial figure's objects and there's still items in fig1
    if fig1:
        # have items in fig1

        for k in fig1.keys():
            aliasPair[k] = -1

    if invert:
        invertedAliasPair = {}
        for key, val in aliasPair.items():
            if key == -1:
                for v in val:
                    invertedAliasPair[v] = -1
            elif val == -1:
                if not -1 in invertedAliasPair:
                    invertedAliasPair[-1] = []
                invertedAliasPair[-1].append(key)
            else:
                invertedAliasPair[val] = key

        aliasPair = invertedAliasPair

    # if debug:
    #     ipdb.set_trace()

    return aliasPair


def cleanFormatting(attrs):
    if ('size' in attrs):
        attrs['width'] = attrs['size']
        attrs['height'] = attrs['size']
        del attrs['size']

    return attrs;
