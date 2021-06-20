from PIL import Image
import numpy as np
import copy
# import ipdb

from AttributeTypes import attrGen


class SemNet:

    @staticmethod
    def generate(problem, netName, dim, allNodes, globalIDs, aliasPair):
        ravenFigure = problem.figures[netName]
        adjMat = np.zeros((dim, dim), dtype=object)

        # print('got omap')
        edges = []
        locallyUsedIDs = set()

        matchPart1 = []
        matchPart2 = []
        for objName, objVal in ravenFigure.objects.items():
            if aliasPair and aliasPair[objName] != -1:
                matchPart1.append(objName)
            else:
                matchPart2.append(objName)
        objList = matchPart1 + matchPart2

        for objName in objList:
            # save manual -1 matches for later
            objVal = ravenFigure.objects[objName]

            node = None
            # if we can alias this node automatically
            if aliasPair and aliasPair[objName] != -1:
                aliasName = aliasPair[objName]
                # print('using alias: {0}'.format(aliasName))
                node = SemNode.convert(objName, objVal, edges, allNodes[aliasName])

            # if generating another id category still leaves room for one more
            elif SemNode.objectIDs < dim:
                node = SemNode.convert(objName, objVal, edges)

            # manually choosing alias
            else:
                aliasName = manualCandidateMatch(problem, locallyUsedIDs, globalIDs, objVal)
                node = SemNode.convert(objName, objVal, edges, allNodes[aliasName])

            # add the node into the Semantic network, allNodes libary, locallyUsedIDS, globalIDs libary
            adjMat[node.id][node.id] = node
            allNodes[objName] = node
            locallyUsedIDs.add(node.id)

            if not node.id in globalIDs:
                globalIDs[node.id] = []
            globalIDs[node.id].append(objName)

        # have all nodes in Sent now
        # have all internalLinks, so place edges now accordingly
        if edges:
            # print('edges with internal links!')
            for node0, edge, nodes1 in edges:
                nodes1 = nodes1.split(',')
                for node1 in nodes1:
                    row = allNodes[node0].id
                    col = allNodes[node1].id

                    if adjMat[row][col] and isinstance(adjMat[row][col], SemEdge):
                        # some edge already exists, so append
                        adjMat[row][col].addEdge(edge)
                    else:
                        # new edge
                        adjMat[row][col] = SemEdge.generate(edge)

                # print(node0, edge, node1)

        return SemNet(len(ravenFigure.objects), adjMat)

    def __init__(self, status, adjMat=0):
        # print('creating semnet', type(adjMat))
        self.status = status
        self.adjMat = adjMat

    def __sub__(self, other):
        if other and isinstance(other, SemNet):
            return SemNet(self.status - other.status, self.adjMat - other.adjMat)
        else:
            return None

        # print('subtacting two sem nets', type(self.adjMat), type(other.adjMat))

    def __rsub__(self, other):
        if other and isinstance(other, SemNet):
            return other - self
        else:
            return None

    def __str__(self):
        return '\n\t{0}\n'.format(str(self.adjMat))

    def __repr__(self):
        return self.__str__()


class SemNode:
    objectIDs = 0

    @staticmethod
    def convert(objName, objVal, edges, alias=None):
        nID = None
        if alias:
            nID = alias.id
        else:
            # use the current id and then increment it for the next node to use if necessary
            nID = SemNode.objectIDs
            # print('{0} is a new object: {1}'.format(objName, SemNode.objectIDs))
            SemNode.objectIDs += 1

        nodeAttrs = {}
        for attrName, attrVal in objVal.attributes.items():
            if attrName in SemEdge.edgeTerms:
                edges.append((objName, attrName, attrVal))
            else:
                attrGen(nodeAttrs, attrName, attrVal)

        return SemNode(nID, 1, nodeAttrs)

    def __init__(self, id, status, attributes):
        self.id = id
        self.status = status
        self.attributes = attributes

    def __sub__(self, other):
        if other == 0:
            return copy.deepcopy(self)

        if self.status == other.status:
            # 0 status change
            if self.id != other.id:
                raise ValueError('Node Subtraction: Incorrect Index!', self, other)

            sAttr = self.attributes
            oAttr = other.attributes
            newAttr = {}

            for attrName in sAttr:
                if attrName in oAttr:
                    newAttr[attrName] = sAttr[attrName] - oAttr[attrName]
                else:
                    newAttr[attrName] = sAttr[attrName]

            for attrName in oAttr:
                if not attrName in sAttr:
                    newAttr[attrName] = 0 - oAttr[attrName]

            return SemNode(self.id, 0, newAttr)

        raise ValueError('Node Subtraction: Invalid Status!', self, other)

    def __rsub__(self, other):
        if other == 0:
            return SemNode(self.id, -1, self.attributes)
        else:
            raise ValueError('Node Subtraction: Inconsistent Types!', self, other)

    def __str__(self):
        return '\n\tID: {0}, ST: {1}, ATRS: {2}\n'.format(self.id, self.status, self.attributes)
        # return 'id:', str(self.id),'status:',str(self.status), 'attr:',str(self.attributes)

    def __repr__(self):
        return self.__str__()


class SemEdge:
    edgeTerms = {'inside', 'left-of', 'above', 'overlaps'}

    @staticmethod
    def generate(edgeType):
        return SemEdge(1, set([edgeType]))

    def __init__(self, status, attributes):
        self.status = status
        self.attributes = attributes

    def __sub__(self, other):
        if other == 0:
            return SemEdge(self.status, self.attributes.copy())

        newStatus = self.status - other.status

        if newStatus:
            return SemEdge(newStatus, self.attributes.copy())
        elif self.attributes == other.attributes:
            return 0
        elif isinstance(other, SemEdge):
            return SemEdge(0, self.attributes - other.attributes)
        else:
            return None

    def __rsub__(self, other):
        if other == 0:
            return SemEdge(-1, self.attributes.copy())
        else:
            raise ValueError('Edge Subtraction: Inconsistent Types!', self, other)

    def addEdge(self, edgeType):
        self.attributes.add(edgeType)

    def __str__(self):
        return '{0}~{1}'.format(self.status, self.attributes)

    def __repr__(self):
        return self.__str__()


def manualCandidateMatch(problem, locallyUsedIDs, globalIDs, objVal):
    # go through all current objectIDs and choose best among unused categories
    candidates = {}
    for i in range(0, SemNode.objectIDs):
        if (not i in locallyUsedIDs):
            # candidates += globalIDs[i]
            for c in globalIDs[i]:
                candidates[c] = None

    # get all the object attributes from original problem for easy comparison

    for figName2, figObj2 in problem.figures.items():
        # print('Scanning {0}'.format(figName2))
        for objName2, objVal2 in figObj2.objects.items():
            if objName2 in candidates:
                candidates[objName2] = objVal2.attributes

    # print('could use {0}'.format(candidates))
    o0attrs = objVal.attributes

    minDiffScore = None
    bestPairedName = None
    for name1, o1attrs in candidates.items():
        diffScore = 0

        for attr0 in o0attrs:
            if attr0 in o1attrs:
                diffScore -= 1.25
                if o0attrs[attr0] == o1attrs[attr0]:
                    diffScore -= 2
            else:
                diffScore += 0.5
        for attr1 in o1attrs:
            if not attr1 in o0attrs:
                diffScore += 0.5

        # update best diff score if applicable
        if minDiffScore == None or diffScore < minDiffScore:  # if we find an object pairing with few diffs update
            # our objectMap
            minDiffScore = diffScore
            bestPairedName = name1

    return bestPairedName
