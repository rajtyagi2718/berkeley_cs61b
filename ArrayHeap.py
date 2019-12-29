class ArrayHeap():

    def __init__(self):
        self.contents = [None] * 16
        self.size = 0
        # self.contents[0] = ArrayHeap.Node()

    def leftIndex(self, i):
        return i * 2

    def rightIndex(self, i):
        return i * 2 + 1

    def parentIndex(self, i):
        return i // 2

    def getNode(self, index):
        if not self.inBounds(index):
            return None
        return self.contents[index]

    def inBounds(self, index):
        if index > self.size or index < 1:
            return False
        return True

    def swap(self, index1, index2):
        node1 = self.getNode(index1)
        node2 = self.getNode(index2)
        self.contents[index1] = node2
        self.contents[index2] = node1

    def min(self, index1, index2):
        node1 = self.getNode(index1)
        node2 = self.getNode(index2)
        if node1 is None:
            return index2
        if node2 is None:
            return index1
        if node1.myPriority < node2.myPriority:
            return index1
        else:
            return index2

    def swim(self, index):
        self.validateSinkSwimArg(index)
        pIndex = self.parentIndex(index)
        parent = self.getNode(pIndex)
        child = self.getNode(index)
        if parent and parent.myPriority > child.myPriority:
            self.swap(pIndex, index)
            self.swim(pIndex)

    def sink(self, index):
        self.validateSinkSwimArg(index)
        if self.leftIndex(index) > self.size:
            return None
        parent = self.getNode(index)
        childIndex = self.min(self.leftIndex(index),
                            self.rightIndex(index))
        child = self.getNode(childIndex)
        if parent.myPriority > child.myPriority:
            self.swap(index, childIndex)
            self.sink(childIndex)

    def insert(self, item, priority):
        if (self.size + 1 == len(self.contents)):
            self.resize(len(self.contents) * 2)
        self.size += 1
        self.contents[self.size] = ArrayHeap.Node(item, priority)
        self.swim(self.size)

    def peek(self):
        return self.getNode(1)

    def removeMin(self):
        returnNode = self.peek()
        self.swap(1, self.size)
        self.contents[self.size] = None
        self.size -= 1
        if self.size > 0:
            self.sink(1)
        return returnNode

    def changePriority(self, item, priority):
        i = 1
        while i <= self.size:
            nodeI = self.getNode(i)
            if nodeI.item == item:
                prevPriority = nodeI.priority
                nodeI.priority = priority
                if prevPriority < priority:
                    self.swim(i)
                elif prevPriority > priority:
                    self.sink(i)
                return None
            i += 1

    def remove(self, index):
        returnNode = self.getNode(index)
        self.swap(index, self.size)
        self.contents[self.size] = None
        self.size -= 1
        if self.size > 0:
            self.swim(index)
            self.sink(index)
        return returnNode


    def __str__(self):
        return self.toStringHelper(1, '')

    def toStringHelper(self, index, soFar):
        if self.getNode(index) is None:
            return ''
        else:
            toReturn = ''
            rightChild = self.rightIndex(index)
            toReturn += self.toStringHelper(rightChild, '        ' + soFar)
            if self.getNode(rightChild) is not None:
                toReturn += soFar + '    /'
            toReturn += '\n' + soFar + str(self.getNode(index)) + '\n'
            leftChild = self.leftIndex(index)
            if self.getNode(leftChild) is not None:
                toReturn += soFar + '    \\'
            toReturn += self.toStringHelper(leftChild, '        ' + soFar)
            return toReturn

    def validateSinkSwimArg(self, index):
        if index < 1:
            raise ValueError('Cannot sink or swim nodes with index 0 or less.')
        if index > self.size:
            raise ValueError('Cannot sink or swim nodes with index greater than current size.')
        if self.contents[index] is None:
            raise ValueError('Cannot sink or swim None.')

    class Node:

        def __init__(self, item=None, priority=None):
            self.myItem = item
            self.myPriority = priority

        def __str__(self):
            return self.myItem.__str__() + ', ' + str(self.myPriority)

    def resize(self, capacity):
        newContents = [None] * capacity
        newContents[1 : self.size + 1] = self.contents[1: self.size + 1]
        self.contents = newContents



import unittest
class InsertionsTest(unittest.TestCase):

    def testIndexing(self):
        pq = ArrayHeap()
        self.assertEqual(6, pq.leftIndex(3));
        self.assertEqual(10, pq.leftIndex(5));
        self.assertEqual(7, pq.rightIndex(3));
        self.assertEqual(11, pq.rightIndex(5));

        self.assertEqual(3, pq.parentIndex(6));
        self.assertEqual(5, pq.parentIndex(10));
        self.assertEqual(3, pq.parentIndex(7));
        self.assertEqual(5, pq.parentIndex(11));

    def testSwim(self):
        pq = ArrayHeap()
        pq.size = 7;
        for i in range(1, 8):
            pq.contents[i] = ArrayHeap.Node("x" + str(i), i)
        # Change item x6's priority to a low value.

        pq.contents[6].myPriority = 0;
        print("PQ before swimming:");
        print(pq);

        # Swim x6 upwards. It should reach the root.

        pq.swim(6);
        print("PQ after swimming:");
        print(pq);
        self.assertEqual("x6", pq.contents[1].myItem);
        self.assertEqual("x2", pq.contents[2].myItem);
        self.assertEqual("x1", pq.contents[3].myItem);
        self.assertEqual("x4", pq.contents[4].myItem);
        self.assertEqual("x5", pq.contents[5].myItem);
        self.assertEqual("x3", pq.contents[6].myItem);
        self.assertEqual("x7", pq.contents[7].myItem);

    def testSink(self):
        pq = ArrayHeap()
        pq.size = 7;
        for i in range(1, 8):
            pq.contents[i] = ArrayHeap.Node("x" + str(i), i);

        # Change root's priority to a large value.
        pq.contents[1].myPriority = 10;
        print("PQ before sinking:");
        print(pq);

        # Sink the root.
        pq.sink(1);
        print("PQ after sinking:");
        print(pq);
        self.assertEqual("x2", pq.contents[1].myItem);
        self.assertEqual("x4", pq.contents[2].myItem);
        self.assertEqual("x3", pq.contents[3].myItem);
        self.assertEqual("x1", pq.contents[4].myItem);
        self.assertEqual("x5", pq.contents[5].myItem);
        self.assertEqual("x6", pq.contents[6].myItem);
        self.assertEqual("x7", pq.contents[7].myItem);

    def testInsert(self):
        pq = ArrayHeap();
        pq.insert("c", 3);
        self.assertEqual("c", pq.contents[1].myItem);

        pq.insert("i", 9);
        self.assertEqual("i", pq.contents[2].myItem);

        pq.insert("g", 7);
        pq.insert("d", 4);
        self.assertEqual("d", pq.contents[2].myItem);

        pq.insert("a", 1);
        self.assertEqual("a", pq.contents[1].myItem);

        pq.insert("h", 8);
        pq.insert("e", 5);
        pq.insert("b", 2);
        pq.insert("c", 3);
        pq.insert("d", 4);
        print("pq after inserting 10 items: ");
        print(pq);
        self.assertEqual(10, pq.size);
        self.assertEqual("a", pq.contents[1].myItem);
        self.assertEqual("b", pq.contents[2].myItem);
        self.assertEqual("e", pq.contents[3].myItem);
        self.assertEqual("c", pq.contents[4].myItem);
        self.assertEqual("d", pq.contents[5].myItem);
        self.assertEqual("h", pq.contents[6].myItem);
        self.assertEqual("g", pq.contents[7].myItem);
        self.assertEqual("i", pq.contents[8].myItem);
        self.assertEqual("c", pq.contents[9].myItem);
        self.assertEqual("d", pq.contents[10].myItem);

    def testInsertAndRemoveOnce(self):
        pq = ArrayHeap();
        pq.insert("c", 3);
        pq.insert("i", 9);
        pq.insert("g", 7);
        pq.insert("d", 4);
        pq.insert("a", 1);
        pq.insert("h", 8);
        pq.insert("e", 5);
        pq.insert("b", 2);
        pq.insert("c", 3);
        pq.insert("d", 4);
        removed = pq.removeMin();
        self.assertEqual("a", removed.myItem);
        self.assertEqual(9, pq.size);
        self.assertEqual("b", pq.contents[1].myItem);
        self.assertEqual("c", pq.contents[2].myItem);
        self.assertEqual("e", pq.contents[3].myItem);
        self.assertEqual("c", pq.contents[4].myItem);
        self.assertEqual("d", pq.contents[5].myItem);
        self.assertEqual("h", pq.contents[6].myItem);
        self.assertEqual("g", pq.contents[7].myItem);
        self.assertEqual("i", pq.contents[8].myItem);
        self.assertEqual("d", pq.contents[9].myItem);

    def testInsertAndRemoveAllButLast(self):
        pq = ArrayHeap();
        pq.insert("c", 3);
        pq.insert("i", 9);
        pq.insert("g", 7);
        pq.insert("d", 4);
        pq.insert("a", 1);
        pq.insert("h", 8);
        pq.insert("e", 5);
        pq.insert("b", 2);
        pq.insert("c", 3);
        pq.insert("d", 4);
        i = 0;
        expected = ["a", "b", "c", "c", "d", "d", "e", "g", "h", "i"]
        while (pq.size > 1):
            self.assertEqual(expected[i], pq.removeMin().myItem);
            i += 1;


if __name__ == '__main__':
    #import doctest
    #doctest.testmod()

    unittest.main()
