

import numpy as np

class Board:

    def __init__(self, numRows, numCols, blocks):
        self.numRows = numRows
        self.numCols = numCols
        self.array = np.zeros((numRows, numCols), int)
        self.initBlocks(blocks)

    def initBlocks(self, blocks):
        for b in blocks:
            self.array[b] = 1

    def listEdges(self, position):
        returnLst = []
        if position[0] > 0:
            returnLst.append((position[0] - 1, position[1]))
        if position[0] < self.numRows - 1:
            returnLst.append((position[0] + 1, position[1]))
        if position[1] > 0:
            returnLst.append((position[0], position[1] - 1))
        if position[1] < self.numCols - 1:
            returnLst.append((position[0], position[1] + 1))
        return returnLst



class Stack:

    def __init__(self):
        self.array = []

    def isEmpty(self):
        return not self.array

    def push(self, item):
        self.array.append(item)

    def pop(self):
        return self.array.pop()

    def __iter__(self):
        return reversed(self.array)

class Queue:

    def __init__(self):
        self.array = []
        self.first = 0
        self.size = 0

    def isEmpty(self):
        return self.size == 0

    def enqueue(self, item):
        self.array.append(item)
        self.size += 1

    def dequeue(self):
        if self.isEmpty():
            raise AssertionError('Cannot dequeue from empty list')
        returnItem = self.array[self.first]
        self.array[self.first] = None
        self.size -= 1
        self.first += 1
        self.resize()
        return returnItem

    def resize(self):
        if len(self.array) > 16 and (len(self.array) + 1) * .75 < self.first + 1:
            self.array = self.array[self.first : ]
            self.first = 0

    def __iter__(self):
        return self.array[self.first : ].__iter__()


class BreadthFirstPaths:

    def __init__(self, board, source):
        self.board = board
        self.source = source
        self.marked = np.array(board.array)
        self.edgeTo = np.zeros((self.board.numRows,
                                self.board.numCols),
                               dtype='int, int')
        self.order = []
        self.search(self.source)

    def mark(self, position):
        self.marked[position] = 2

    def isMarked(self, position):
        return bool(self.marked[position])

    def fillEdgeTo(self, curr, prev):
        self.edgeTo[curr] = prev

    def search(self, position):
        callQueue = Queue()
        callQueue.enqueue(position)
        self.mark(position)
        while not callQueue.isEmpty():
            position = callQueue.dequeue()
            self.order.append(position)
            self.enqueueUnmarkedAdj(position, callQueue)

    def enqueueUnmarkedAdj(self, position, callQueue):
        for p in self.board.listEdges(position):
            if not self.isMarked(p):
                callQueue.enqueue(p)
                self.mark(p)
                self.fillEdgeTo(p, position)

    def hasPathTo(self, target):
        return self.marked[target] == 2

    def pathTo(self, target):
        if not self.hasPathTo(target):
            return None
        path = Stack()
        while target != self.source:
            path.push(target)
            target = tuple(self.edgeTo[target])
        path.push(self.source)
        return list(path)

    def printPath(self, path):
        pathBoard = self.printPathBoard(path)
        printBorder = '-' * (self.board.numCols + 2)
        printBoard = printBorder + '\n'
        for row in range(self.board.numRows):
            printRow = '-'
            for col in range(self.board.numCols):
                val = pathBoard[row, col]
                printRow += self.printFcn(val)
            printBoard += printRow + '-' + '\n'
        printBoard += printBorder
        print(printBoard)

    def printPathBoard(self, path):
        pathBoard = np.array(self.board.array)
        start = path[0]
        pathBoard[start] = 2
        for i in range(1, len(path) - 1):
            middle = path[i]
            pathBoard[middle] = 3
        end = path[-1]
        pathBoard[end] = 4
        return pathBoard

    def printFcn(self, k):
        returnStr = ''
        if k == 0:
            returnStr += ' '
        elif k == 1:
            returnStr += '#'
        elif k == 2:
            returnStr += 'o'
        elif k == 3:
            returnStr += '.'
        else: # k == 4:
            returnStr += 'O'
        return returnStr


import unittest
import random

class BFPTest(unittest.TestCase):

    def test(self):
        B = Board(3, 3, [(1,1), (2,1)])
        BFP = BreadthFirstPaths(B, (2,0))
        path = BFP.pathTo((2,2))
        self.assertEqual(path,
            [(2,0), (1,0), (0,0), (0,1), (0,2), (1,2), (2,2)])
        BFP.printPath(path)


    def test2(self):
        for _ in range(5):
            hasPath = False
            while not hasPath:
                m, n = random.randint(3, 50), random.randint(3, 70)
                blocks = []
                inBlocks = {0:1}
                for _ in range(int(m*n*.4)):
                    b = random.randint(0, m-1), random.randint(0, n-1)
                    while inBlocks.get(b):
                        b = random.randint(0, m-1), random.randint(0, n-1)
                    blocks.append(b)
                    inBlocks[b] = 1
                B = Board(m, n, blocks)
                source = random.randint(0, m-1), random.randint(0, n-1)
                while inBlocks.get(source):
                    source = random.randint(0, m-1), random.randint(0, n-1)
                inBlocks[source] = 1
                target = random.randint(0, m-1), random.randint(0, n-1)
                while inBlocks.get(target) or (abs(target[0] - source[0]) +
                    abs(target[1] - source[1]) < .5*(m+n)):
                    target = random.randint(0, m-1), random.randint(0, n-1)
                inBlocks[target] = 1
                BFP = BreadthFirstPaths(B, source)
                hasPath = BFP.hasPathTo(target)
            path = BFP.pathTo(target)
            BFP.printPath(path)

if __name__ == '__main__':

    unittest.main()
