
##############
## Preamble ##
##############

import unittest




#####################
## Data Structures ##
#####################

class Node:

    def __init__(self, R=0):
        self.value = None
        self.children = [None for _ in range(R)]
        self.isFinal = False

    def __repr__(self):
        childrenRepr = ','. join([C.__repr__()  if C else '' for C in self.children])
        if self.value:
            valueRepr = self.value.__repr__()
        else:
            valueRepr = ''
        return '[' + valueRepr + ',' + childrenRepr + ']'

class TrieST:

    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.aMap = {k:v for v,k in enumerate(alphabet)}
        self.R = len(alphabet)
        self.root = None
        self.size = 0

    def checkKey(key):
        if key == None:
            raise ValueError('key argument cannot be None')

    def contains(self, key):
        TrieST.checkKey(key)
        node = self._getNode(self.root, key)
        if node != None:
            return node.isFinal
        return False

    def get(self, key):
        TrieST.checkKey(key)
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
        j = self.aMap[key[i]]
        return self._getNodeDigit(node.children[j], key, i+1)

    def insert(self, key, value=None):
        TrieST.checkKey(key)
        self.root = self._insertDigit(self.root, key, value, 0)

    def _insertDigit(self, node, key, value, i):
        if node == None:
            node = Node(self.R)
        if i == len(key):
            if not node.isFinal:
                node.isFinal = True
                self.size += 1
            node.value = value
            return node
        # find child for key[i]
        j = self.aMap[key[i]]
        node.children[j] = self._insertDigit(node.children[j],
                                             key, value, i+1)
        return node

    def getPrefix(self, key):
        node = self._getNode(self.root, key)
        returnLst = []
        self._getPrefixNode(node, key, returnLst)
        return returnLst

    def _getPrefixNode(self, node, prefix, lst):
        if node == None:
            return
        if node.isFinal:
            lst.append((prefix, node.value))
        for i in range(len(node.children)):
            childNode = node.children[i]
            childPrefix = prefix + self.alphabet[i]
            self._getPrefixNode(childNode, childPrefix, lst)

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
        k = self.aMap[key[i]]
        return self._getLongestPrefixNode(node.children[k], key,
                                          i+1, j, value)

    def getRange(self, start, stop):
        pass

    def __iter__(self):

        class Stack:

            def __init__(self):
                self.array = []

            def isEmpty(self):
                return not self.array

            def push(self, item):
                self.array.append(item)

            def pop(self):
                return self.array.pop()

        class NodeIterator:

            def __init__(self, R, alphabet, root, size):
                self.R = R
                self.alphabet = alphabet
                self.size = size
                self.stack = Stack()
                self.stack.push((root, ''))

            # add successor branch
            def pushChildren(self, node, prefix):
                if node == None:
                    return
                for i in range(self.R):
                    self.stack.push((node.children[i],
                                    prefix + self.alphabet[i]))

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

        return NodeIterator(self.R, self.alphabet, self.root, self.size)

###########
## Tests ##
###########
import random

class TrieSTTest(unittest.TestCase):

    def testEmptyNode(self):
        N = Node()
        # self.assertEqual(N.R, 0)
        self.assertFalse(N.isFinal)
        self.assertEqual(N.children, [])

    def testNode(self):
        N = Node(5)
        # self.assertEqual(N.R, 5)
        self.assertFalse(N.isFinal)
        self.assertEqual(len(N.children), 5)
        for L in N.children:
            # self.assertEqual(L.R, 0)
            self.assertEqual(L, None)

    def testEmpty(self):
        A = list(map(str, range(10)))
        T = TrieST(A)
        self.assertFalse(T.contains('2'))
        self.assertFalse(T.contains(''))
        self.assertEqual(T.get('61'), None)
        self.assertEqual(T.get(''), None)
        self.assertEqual(T.size, 0)

    def testInsertEmpty(self):
        A = list(map(str, range(10)))
        T = TrieST(A)
        T.insert('')
        self.assertTrue(T.contains(''))
        self.assertEqual(T.get(''), None)
        self.assertFalse(T.contains('123'))
        self.assertEqual(T.size, 1)
        T = TrieST(A)
        T.insert('', 'A')
        self.assertTrue(T.contains(''))
        self.assertEqual(T.get(''), 'A')
        self.assertFalse(T.contains('9'))
        self.assertEqual(T.size, 1)

    def testInsert(self):
        A = list(map(str, range(10)))
        T = TrieST(A)
        T.insert('123', 'ab')
        self.assertTrue(T.contains('123'))
        self.assertEqual(T.get('123'), 'ab')
        self.assertFalse(T.contains('523'))
        self.assertEqual(T.size, 1)
        T.insert('123', 'abc')
        self.assertTrue(T.contains('123'))
        self.assertEqual(T.get('123'), 'abc')
        self.assertFalse(T.contains('124'))
        self.assertEqual(T.size, 1)
        T.insert('124', 'abd')
        self.assertEqual(T.get('123'), 'abc')
        self.assertEqual(T.get('124'), 'abd')
        self.assertEqual(T.size, 2)

    def testGetPrefix(self):
        A = list(map(str, range(10)))
        T = TrieST(A)
        toInsert = ['123', '124', '1245', '12222',
                    '23142', '56334', '43', '9']
        for i in range(len(toInsert)):
            T.insert(toInsert[i], i)
        self.assertEqual(T.size, len(toInsert))
        P = T.getPrefix('12')
        Q = [('123', 0), ('124', 1), ('1245', 2),
             ('12222', 3)]
        self.assertEqual(set(P), set(Q))
        P1 = T.getPrefix('23144')
        self.assertEqual(P1, [])
        leafStrs = toInsert
        leafStrs.remove('124')
        leafNodes = [T._getNode(T.root, s) for s in leafStrs]
        for l in leafNodes:
            for c in l.children:
                self.assertEqual(c, None)

    def testGetLongestPrefix(self):
        A = list(map(str, range(10)))
        T = TrieST(A)
        toInsert = ['123', '124', '1245', '12222',
                    '23142', '56334', '43', '9']
        for i in range(len(toInsert)):
            T.insert(toInsert[i], i)
        p, v = T.getLongestPrefix('1247891')
        self.assertEqual(p, '124')
        self.assertEqual(v, 1)
        p, v = T.getLongestPrefix('124512423')
        self.assertEqual(p, '1245')
        self.assertEqual(v, 2)
        p, v = T.getLongestPrefix('12222')
        self.assertEqual(p, '12222')
        self.assertEqual(v, 3)

    def testItr(self):
        A = list(map(str, range(10)))
        T = TrieST(A)
        toInsert = [str(random.randint(0, 100000)) for _ in range(10)]
        for i in range(len(toInsert)):
            T.insert(toInsert[i], i)
        print('toInsert', toInsert)
        print(list(T))


if __name__ == '__main__':
    unittest.main()
