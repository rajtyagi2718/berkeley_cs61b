## Data Structures: Stack, Node, BinaryTrieST, Heap ##

###########
## Stack ##
###########

class Stack:

    def __init__(self):
        self.array = []

    def isEmpty(self):
        return not self.array

    def push(self, item):
        self.array.append(item)

    def pop(self):
        return self.array.pop()

#############
## BTrieST ##
#############

class Node:

    def __init__(self, value=None, isFinal=False):
        self.value = value
        self.children = [None, None]
        self.isFinal = isFinal

    def __repr__(self):
        child0, child1 = self.children
        reprLst = []
        if self.value:
            reprLst.append(self.value.__repr__())
        if child1:
            if child0:
                reprLst.append(child0.__repr__())
            else:
                reprLst.append(' ')
            reprLst.append(child1.__repr__())
        return '[' + ', '.join(reprLst) + ']'

class BTrieST:

    def __init__(self):
        self.root = None
        self.size = 0

    def checkKey(key):
        if key == None:
            raise ValueError('key argument cannot be None')

    def contains(self, key):
        BTrieST.checkKey(key)
        node = self._getNode(self.root, key)
        if node != None:
            return node.isFinal
        return False

    def get(self, key):
        BTrieST.checkKey(key)
        node = self._getNode(self.root, key)
        if node != None and node.isFinal:
            return node.value
        return None

    def _getNode(self, node, key):
        return self._getNodeDigit(node, key, 0)

    def _getNodeDigit(self, node, key, i):
        if node == None:
            return None
        if i == len(key):
            return node
        # more digits to process
        k = int(key[i])
        return self._getNodeDigit(node.children[k], key, i+1)

    def insert(self, key, value=None):
        BTrieST.checkKey(key)
        self.root = self._insertDigit(self.root, key, value, 0)

    def _insertDigit(self, node, key, value, i):
        if node == None:
            node = Node()
        if i == len(key):
            if not node.isFinal:
                node.isFinal = True
                self.size += 1
            node.value = value
            return node
        # find child for key[i]
        k = int(key[i])
        node.children[k] = self._insertDigit(node.children[k],
                                             key, value, i+1)
        return node

    def getLongestPrefix(self, key):
        node = self.root
        j, value = self._getLongestPrefixNode(node, key, 0, -1,
                                              node.value)
        if j == -1:
            return None
        return key[0:j], value

    def _getLongestPrefixNode(self, node, key, i, j, value):
        if node == None:
            return j, value
        if node.isFinal:
            j = i
            value = node.value
        if i == len(key):
            return j, value
        k = int(key[i])
        return self._getLongestPrefixNode(node.children[k], key,
                                          i+1, j, value)

    def getRange(self, start, stop):
        pass

    def __iter__(self):

        class NodeIterator:

            def __init__(self, root, size):
                self.size = size
                self.stack = Stack()
                self.stack.push((root, ''))

            def pushChildren(self, node, prefix):
                if node == None:
                    return
                for i in range(2):
                    self.stack.push((node.children[i],
                                     prefix + str(i)))

            def __iter__(self):
                return self

            def __next__(self):
                returnItem = ()
                while self.size and not returnItem:
                    item = self.stack.pop()
                    node, prefix = item
                    if node and node.isFinal:
                        returnItem = (prefix, node.value)
                    self.pushChildren(node, prefix)
                if returnItem:
                    self.size -= 1
                    return returnItem
                raise StopIteration

        return NodeIterator(self.root, self.size)


##########
## Heap ##
##########
import heapq

class Heap:

    def __init__(self, data=None, key=lambda x:x):
        self.key = key
        self.data = [(key(data[i]), i, data[i])
                     for i in range(len(data))]
        self.id = len(data)
        heapq.heapify(self.data)

    def push(self, item):
        heapItem = (self.key(item), self.id, item)
        heapq.heappush(self.data, heapItem)
        self.id += 1

    def pop(self):
        return heapq.heappop(self.data)[2]

    def peek(self):
        return self.data[0][2]

    def replaceMin(self, item):
        heapItem = (self.key(item), self.id, item)
        heapq.heapreplace(self.data, heapItem)
        self.id += 1

    def __iter__(self):
        return iter(item[2] for item in self.data)

    def __len__(self):
        return len(self.data)

#############
## Outline ##
#############

# f txtFile to str
# g str to alphabet list, frequency dicts
# h alphabetLst, freqDict to charTrie, charMap
# i charMap, str to compressedStr
# j charTrie, compressedStr to str

## txtFile path to str ##
# pathStr = input()
def getTxtStr(pathStr=None):
    if not pathStr:
        pathStr = input()
    with open(pathStr, 'r') as myfile:
        txtStr = myfile.read()
    return txtStr

## str to aLst, freqDict ##
def getAFDict(txtStr): #alphabetFrequencyDict
    afDict = {}
    for char in txtStr:
        f = afDict.get(char, 0)
        afDict[char] = f + 1
    return afDict


## alphabetFreqDict to charTrie, charMap ##

def getCharTrie(afDict):
    #frequenceNodeHeap
    data = [(freq, Node(char, True)) for char, freq in afDict.items()]
    fnHeap = Heap(data, key=lambda x: x[0])
    while len(fnHeap) > 1:
        freq0, Node0 = fnHeap.pop()
        freq1, Node1 = fnHeap.peek()
        superFreq = freq0 + freq1
        superNode = Node()
        superNode.children = [Node0, Node1]
        fnHeap.replaceMin((superFreq, superNode))
    charTrie = BTrieST()
    charTrie.root = fnHeap.peek()[1]
    charTrie.size = len(data)
    return charTrie

def getCharMap(charTrie):
    return {a : c for c, a in iter(charTrie)}

## charMap to compressedTxtStr ##

def getCTxtStr(charMap, txtStr):
    result = ''
    for char in txtStr:
        result += charMap[char]
    return result

## compressedTxtStr, charTrie to decompressedTxtStr ##

def getDTxtStr(charTrie, cStr):
    result = ''
    while cStr:
        pref, char = charTrie.getLongestPrefix(cStr)
        result += char
        cStr = cStr[len(pref) : ]
    return result


###########
## Tests ##
###########

import unittest, random, string

class someTest(unittest.TestCase):

    def testOpen(self):
        pathStr = '/home/roger/Downloads/python/cs61b/Temp3.txt'
        txtStr = getTxtStr(pathStr)
        tstStr = 'This is a txt file.\n\nProgram terminated.'
        self.assertEqual(txtStr, tstStr)

    def testAFDict(self):
        txtStr = 'abcabaacba'
        afDict = getAFDict(txtStr)
        afItems = list(afDict.items())
        afItems.sort(key=lambda x: x[1])
        tstItems = [('c', 2), ('b', 3), ('a', 5)]
        self.assertEqual(afItems, tstItems)

    def testCharTrie(self):
        txtStr = 'abcabaacba'
        afDict = getAFDict(txtStr)
        charTrie = getCharTrie(afDict)
        tstStr = "[['a'], [['c'], ['b']]]"
        self.assertEqual(charTrie.root.__repr__(), tstStr)
        charMap = getCharMap(charTrie)
        charLst = list(charMap.items())
        charLst.sort(key=lambda x: x[1])
        tstLst = [('a', '0'), ('c', '10'), ('b', '11')]
        self.assertEqual(charLst, tstLst)

    def testCStr(self):
        txtStr = 'abcabaacba'
        afDict = getAFDict(txtStr)
        charTrie = getCharTrie(afDict)
        charMap = getCharMap(charTrie)
        cTxtStr = getCTxtStr(charMap, txtStr)
        tstStr = '011100110010110'
        self.assertEqual(cTxtStr, tstStr)

    def testCStr(self):
        lower = list(string.printable)
        txtStr = ''.join([random.choice(lower) for _ in range(100)])
        print(txtStr)
        afDict = getAFDict(txtStr)
        charTrie = getCharTrie(afDict)
        charMap = getCharMap(charTrie)
        cTxtStr = getCTxtStr(charMap, txtStr)
        print(cTxtStr)
        dTxtStr = getDTxtStr(charTrie, cTxtStr)
        self.assertEqual(txtStr, dTxtStr)



if __name__ == '__main__':
    unittest.main()
