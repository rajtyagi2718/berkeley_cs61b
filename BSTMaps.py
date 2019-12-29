class Node:

    def __init__(self, key, value, left=None, right=None, size=1):
        assert ((left is None or isinstance(left, Node)) and
                (right is None or isinstance(right,Node)))
        self.key = key
        self.value = value
        self.left = left
        self.right = right

class BST:

    def __init__(self, root=None):
        self.root = root
        self.size = 0

    def get(self, key):
        node = self.root
        return BST.getNode(node, key)

    def getNode(node, key):
        if not node:
            return None
        if key == node.key:
            return node.value
        if key < node.key:
            return BST.getNode(node.left, key)
        if key > node.key:
            return BST.getNode(node.right, key)

    def insert(self, key, value):
        self.root, size = BST.insertNode(self.root, key, value)
        self.size += size

    def insertNode(node, key, value):
        if not node:
            return Node(key, value), 1
        if key == node.key:
            node.value = value
            size = 0
        elif key < node.key:
            node.left, size = BST.insertNode(node.left, key, value)
        else: # elif key > node.key:
            node.right, size = BST.insertNode(node.right, key, value)
        return node, size

    def __iter__(self):

        class Stack:

            def __init__(self):
                self.array = []

            def isEmpty(self):
                return not self.array

            def push(self, item):
                self.array.append(item)

            def pop(self):
                return self.array.pop() #Stack(SLList) maybe faster

        class NodeIterator:

            def __init__(self, root):
                self.stack = Stack()
                self.pushLeftBranch(root)

            # add successor branch
            def pushLeftBranch(self, node):
                if not node:
                    return None
                self.stack.push(node)
                self.pushLeftBranch(node.left)

            def __iter__(self):
                return self

            def __next__(self):
                if self.stack.isEmpty():
                    raise StopIteration
                else:
                    node = self.stack.pop()
                    returnVal = node.value
                    node = node.right
                    self.pushLeftBranch(node)
                    return returnVal

        return NodeIterator(self.root)

    def remove(self, key):
        self.root, returnVal = BST.removeNode(self.root, key)
        self.size -= 1
        return returnVal

    def removeNode(node, key):
        if key == node.key:
            returnVal = node.value
            node = BST.removeRoot(node)
        elif key < node.key:
            node, returnVal = BST.removeNode(node.left, key)
        else: # key > node.key:
            node, returnVal = BST.removeNode(node.right, key)
        return node, returnVal

    def removeRoot(root):
        node = root
        if not node.left and not node.right:
            return None
        if node.left and not node.right:
            return node.left
        if node.right and not node.left:
            return node.right
        else: # node.right and node.left
            return BST.removeRootSuccessor(root)

    def removeRootSuccessor(root):
        node = root
        successor = node.right
        while successor.left:
            node, successor = successor, successor.left
        root.key = successor.key
        root.value = successor.value
        if successor == root.right:
            root.right = None
        else:
            node.left = None
        return root

        # def removeRootPredecessor(root):
        #     node = root
        #     predecessor = node.left
        #     while predecessor.right:
        #         node, predecessor = predecessor, predecessor.right
        #     root.key = predecessor.key
        #     root.value = predecessor.value
        #     if predecessor == root.left:
        #         root.left = None
        #     else:
        #         node.right = None
        #     return root

    def listInOrderTraversal(self):
        return BST.listInOrderTraversalNode(self.root)

    def listInOrderTraversalNode(node):
        result = []
        if not node:
            return result
        result.extend(BST.listInOrderTraversalNode(node.left))
        result.append(node.value)
        result.extend(BST.listInOrderTraversalNode(node.right))
        return result

    def listPreOrderTraversal(self):
        return BST.listPreOrderTraversalNode(self.root)

    def listPreOrderTraversalNode(node):
        result = []
        if not node:
            return result
        result.append(node.value)
        result.extend(BST.listPreOrderTraversalNode(node.left))
        result.extend(BST.listPreOrderTraversalNode(node.right))
        return result

    def listPostOrderTraversal(self):
        return BST.listPostOrderTraversalNode(self.root)

    def listPostOrderTraversalNode(node):
        result = []
        if not node:
            return result
        result.extend(BST.listPostOrderTraversalNode(node.left))
        result.extend(BST.listPostOrderTraversalNode(node.right))
        result.append(node.value)
        return result




    # @property
    # def key(self):
    #     return self.root.key
    #
    # @property
    # def value(self):
    #     return self.root.value
    #
    # @property
    # def left(self):
    #     return self.root.left
    #
    # @property
    # def right(self):
    #     return self.root.right


###########
## OTHER ##
###########

def isBST(T):
    return helper(T.root)

def helper(root, min=None, max=None): # no repeats
    if not root:
        return True
    if min and min >= root.key:
        return False
    if max and max <= root.key:
        return False
    else:
        return ((not root.left or helper(root.left, min, root.key)) and
                (not root.right or helper(root.right, root.key, max)))



###########
## TESTS ##
###########

import unittest
class InsertionsTest(unittest.TestCase):

    def testInstantiation(self):
        B = BST()
        self.assertEqual(B.size, 0)
        self.assertEqual(B.root, None)
        self.assertEqual(B.get(1), None)

    def testOneInsertion(self):
        B = BST()
        B.insert(3, 'd')
        self.assertEqual(B.size, 1)
        self.assertEqual(B.root.key, 3)
        self.assertEqual(B.root.value, 'd')
        self.assertEqual(B.get(3), 'd')
        self.assertEqual(B.root.left, None)
        self.assertEqual(B.root.right, None)

    def testSameInsertion(self):
        B = BST()
        B.insert(3, 'd')
        B.insert(3, 'c')
        self.assertEqual(B.size, 1)
        self.assertEqual(B.root.key, 3)
        self.assertEqual(B.root.value, 'c')
        self.assertEqual(B.get(3), 'c')
        self.assertEqual(B.root.left, None)
        self.assertEqual(B.root.right, None)

    def testRemoveToNone(self):
        B = BST()
        B.insert(3, 'd')
        B.remove(3)
        self.assertEqual(B.size, 0)
        self.assertEqual(B.root, None)
        self.assertEqual(B.get(3), None)

    def testFiveInsertions(self):
        B = BST()
        B.insert(3, 'c')
        B.insert(2, 'b')
        B.insert(1, 'a')
        B.insert(5, 'e')
        B.insert(4, 'd')
        self.assertEqual(B.size, 5)
        self.assertEqual(B.root.key, 3)
        self.assertEqual(B.get(5), 'e')
        self.assertEqual(B.root.left.left.key, 1)
        self.assertEqual(B.root.left.key, 2)
        self.assertEqual(B.root.right.key, 5)
        self.assertEqual(B.root.right.left.key, 4)

    def testInOrderTraversalOrderedInsertions(self):
        import string, random
        B = BST()
        self.assertEqual(B.listInOrderTraversal(), [])
        for i in range(10):
            B.insert(i, string.ascii_lowercase[i])
        self.assertEqual(B.listInOrderTraversal(),
                         list(string.ascii_lowercase[0 : 10]))

    def testInOrderTraversalRandomInsertions(self):
        import string, random
        B = BST()
        enum = list(enumerate(string.ascii_lowercase))
        random.shuffle(enum)
        for key, value in enum:
            B.insert(key, value)
        self.assertEqual(B.listInOrderTraversal(),
                         list(string.ascii_lowercase))

    def testIterator(self):
        import string, random
        B = BST()
        self.assertEqual([x for x in B], [])
        enum = list(enumerate(string.ascii_lowercase))
        random.shuffle(enum)
        for key, value in enum:
            B.insert(key, value)
        self.assertEqual([x for x in B],
                         list(string.ascii_lowercase))


if __name__ == '__main__':
    #import doctest
    #doctest.testmod()

    unittest.main()
