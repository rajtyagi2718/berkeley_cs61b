
# To Do:
# Improve search function: heuristic, number of moves before measuring,  repeats
# Try vector storage with print display as is
# Or maybe tuple storage since not mutating


#############################
## General Data Structures ##
#############################

import numpy as np, random

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

class PriorityQueue:

    def __init__(self):
        self.array = [None]

    @property
    def size(self):
        return len(self.array) - 1

    def isEmpty(self):
        return self.size == 0

    class PObj:

        def __init__(self, item, priority):
            self.item = item
            self.priority = priority

        def changePriority(self, newPriority):
            self.priority = newPriority

    def add(self, item, priority):
        """ Adds the item to the priority queue."""
        pObj = PriorityQueue.PObj(item, priority)
        self.array.append(pObj)
        self.swim(self.size)

    def swim(self, index):
        parent = index // 2
        if parent and self.array[parent].priority > self.array[index].priority:
            self.array[parent], self.array[index] = (
                self.array[index], self.array[parent])
            self.swim(parent)

    def getSmallest(self):
        """ Returns the smallest item in the priority queue."""
        return self.array[1].item

    def removeSmallest(self):
        """ Removes the smallest item in the priority queue."""
        # place last item at root
        # demote as needed, pick strongest child
        returnItem = self.array[1].item
        last = self.array.pop()
        if not self.isEmpty():
            self.array[1] = last
            self.sink(1)
        return returnItem

    def sink(self, index):
        child = self.smallestChild(index)
        if child and self.array[child].priority < self.array[index].priority:
            self.array[child], self.array[index] = (
                self.array[index], self.array[child])
            self.sink(child)

    def smallestChild(self, index):
        child1 = index * 2
        if child1 == self.size:
            return child1
        if child1 > self.size:
            return None
        child2 = child1 + 1
        smallerChild = min([child1, child2],
                           key=lambda x: self.array[x].priority)
        return smallerChild

    def changePriority(self, item, newPriority):
        #if item not in queue, nothing done
        for i in range(1, self.size):
            if self.array[i].item == item:
                self.array[i].changePriority(newPriority)
                self.swim(i)
                self.sink(i)
                break

    def max(self):
        return max(self.array[self.size // 2 + 1 : self.size + 1],
                   key=lambda x: x.priority)

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

class Face:

    def __init__(self, array):
        self.array = array
        self.center = array[1,1]

    @property
    def top(self):
        return self.array[0,:]

    @property
    def left(self):
        return self.array[:,0]

    @property
    def btm(self):
        return self.array[2,:]

    @property
    def right(self):
        return self.array[:,2]

    # def changeTop(self, newArr):
    #     self.array[0,:] = newArr
    #
    # def changeLeft(self, newArr):
    #     self.array[:,0] = newArr
    #
    # def changeBtm(self, newArr):
    #     self.array[2,:] = newArr
    #
    # def changeRight(self, newArr):
    #     self.array[2,:] = newArr

    def completeness(self):
        count = 0
        for i in [0, 2]:
            for j in range(3):
                if self.array[i, j] == self.center:
                    count += 1
        i = 1
        for j in [0, 2]:
            if self.array[i, j] == self.center:
                count += 1
        return count

    def __str__(self):
        return self.array.__str__()

############################
## Rubiks Data Structures ##
############################

class Cube:

    def __init__(self, Top, Left, Front, Right, Back, Btm):
        self.top = Top
        self.left = Left
        self.front = Front
        self.right = Right
        self.back = Back
        self.btm = Btm
        self.faces = ['top', 'left', 'front', 'right', 'back', 'btm']

    def completeness(self):
        return sum([getattr(getattr(self, f), 'completeness')()
                    for f in self.faces])

    def __repr__(self):
        result = ''
        for f in self.faces:
            lSpace = ' ' * ((7-len(f)) // 2)
            rSpace = ' ' * (8 - len(f) - len(lSpace))
            result += lSpace + f + rSpace
        for i in range(3):
            rowi = ''
            for f in self.faces:
                rowi += getattr(self, f).array[i].__str__() + ' '
            result += '\n' + rowi
        return result

    def __key(self):
        return tuple(tuple(getattr(self, f).array.flatten())
                     for f in self.faces)

    def __eq__(x, y):
        return x.__key() == y.__key()

    def __hash__(self):
        return hash(self.__key())

class NewFace:
    # return new Face with desired rotations
    # does not alter input

    def fill(i):
        arr = np.empty((3,3), int)
        arr.fill(i)
        return Face(arr)

    def top(F, newTop):
        return Face(np.array([newTop, F.array[1], F.array[2]]))

    def left(F, newLeft):
        return Face(np.array([[newLeft[0], F.array[0,1], F.array[0,2]],
                              [newLeft[1], F.array[1,1], F.array[1,2]],
                              [newLeft[2], F.array[2,1], F.array[2,2]]]))

    # def left2(F, newLeft):
    #     newArr = np.array(F.array)
    #     newArr[:,0] = newLeft
    #     return Face(newArr)

    def btm(F, newBtm):
        return Face(np.array([F.array[0], F.array[1], newBtm]))

    def right(F, newRight):
        return Face(np.array([[F.array[0,0], F.array[0,1], newRight[0]],
                              [F.array[1,0], F.array[1,1], newRight[1]],
                              [F.array[2,0], F.array[2,1], newRight[2]]]))

    # def right2(F, newRight):
    #     newArr = np.array(F.array)
    #     newArr[:,2] = newRight
    #     return Face(newArr)

    def rotate90(F):
        return Face(np.array(np.rot90(F.array)))

    def rotate90CW(F):
        return Face(np.array(np.rot90(F.array, -1)))

    # def rotate180(F):
    #     return Face(np.array(np.rot90(F.array, 2)))


    # def rotate90(self):
    #     (self.array[0,0], self.array[2,0], self.array[2,2], self.array[0,2]) = (
    #         self.array[0,2], self.array[0,0], self.array[2,0], self.array[2,2])
    #     (self.array[1,0], self.array[2,1], self.array[1,2], self.array[0,1]) = (
    #         self.array[0,1], self.array[1,0], self.array[2,1], self.array[1,2])
    #
    # def rotate180(self):
    #     self.array[0,0], self.array[2,2] = self.array[2,2], self.array[0,0]
    #     self.array[0,2], self.array[2,0] = self.array[2,0], self.array[0,2]
    #     self.array[0,1], self.array[2,1] = self.array[2,1], self.array[0,1]
    #     self.array[1,0], self.array[1,2] = self.array[1,2], self.array[1,0]
    #
    # def rotate90CW(self):
    #     (self.array[0,0], self.array[2,0], self.array[2,2], self.array[0,2]) = (
    #         self.array[2,0], self.array[2,2], self.array[0,2], self.array[0,0])
    #     (self.array[1,0], self.array[2,1], self.array[1,2], self.array[0,1]) = (
    #         self.array[2,1], self.array[1,2], self.array[0,1], self.array[1,0])

    def arangeFill(i):
        return Face(np.arange(9*i, 9*(i+1)).reshape(3,3))

class NewCube: #returns new cube, does not affect input Cube

    FaceStrs = ['Top', 'Left', 'Front', 'Right', 'Back', 'Btm']
    rotType = (['rotate' + s + '90' for s in FaceStrs] +
               ['rotate' + s + '90CW' for s in FaceStrs])

    def fill():
        Faces = [NewFace.fill(i) for i in range(6)]
        return Cube(*Faces)

    def rotateTop90(C): #CCW
        newTop = NewFace.rotate90(C.top)
        newLeft = NewFace.top(C.left, C.back.top)
        newFront = NewFace.top(C.front, C.left.top)
        newRight = NewFace.top(C.right, C.front.top)
        newBack = NewFace.top(C.back, C.right.top)
        newBtm = Face(np.array(C.btm.array))
        return Cube(newTop, newLeft, newFront, newRight,
                    newBack, newBtm)

    def rotateTop90CW(C):
        newTop = NewFace.rotate90CW(C.top)
        newLeft = NewFace.top(C.left, C.front.top)
        newFront = NewFace.top(C.front, C.right.top)
        newRight = NewFace.top(C.right, C.back.top)
        newBack = NewFace.top(C.back, C.left.top)
        newBtm = Face(np.array(C.btm.array))
        return Cube(newTop, newLeft, newFront, newRight,
                    newBack, newBtm)

    # def rotateTop180(C):
    #     newTop = NewFace.rotate180(C.top)
    #     newLeft = NewFace.top(C.left, C.right.top)
    #     newFront = NewFace.top(C.front, C.back.top)
    #     newRight = NewFace.top(C.right, C.left.top)
    #     newBack = NewFace.top(C.back, C.front.top)
    #     newBtm = Face(np.array(C.btm.array))
    #     return Cube(newTop, newLeft, newFront, newRight,
    #                 newBack, newBtm)
    #
    # def rotateTop90CW(C):
    #     newTop = NewFace.rotate90CW(C.top)
    #     newLeft = NewFace.top(C.left, C.front.top)
    #     newFront = NewFace.top(C.front, C.right.top)
    #     newRight = NewFace.top(C.right, C.back.top)
    #     newBack = NewFace.top(C.back, C.left.top)
    #     newBtm = Face(np.array(C.btm.array))
    #     return Cube(newTop, newLeft, newFront, newRight,
    #                 newBack, newBtm)
    #
    # def rotateTop902(C): #CCW
    #     FaceLst = [None for _ in range(6)]
    #     # FaceLst = [newTop, newLeft, newFront,
    #     #            newRight, newBack, newBtm]
    #     FaceLst[0] = NewFace.rotate90(C.top)
    #     FaceLst[1] = NewFace.top(C.left, C.back.top)
    #     FaceLst[2] = NewFace.top(C.front, C.left.top)
    #     FaceLst[3] = NewFace.top(C.right, C.front.top)
    #     FaceLst[4] = NewFace.top(C.back, C.right.top)
    #     FaceLst[5] = Face(np.array(C.btm.array))
    #     return Cube(*FaceLst)

    def rotateLeft90(C): #CCW
        newLeft = NewFace.rotate90(C.left)
        newBack = NewFace.right(C.back, C.top.left)
        newBtm = NewFace.left(C.btm, C.back.right)
        newFront = NewFace.left(C.front, C.btm.left)
        newTop = NewFace.left(C.top, C.front.left)
        newRight = Face(np.array(C.right.array))
        return Cube(newTop, newLeft, newFront, newRight,
                    newBack, newBtm)

    def rotateLeft90CW(C):
        newLeft = NewFace.rotate90CW(C.left)
        newBack = NewFace.right(C.back, C.btm.left)
        newBtm = NewFace.left(C.btm, C.front.left)
        newFront = NewFace.left(C.front, C.top.left)
        newTop = NewFace.left(C.top, C.back.right)
        newRight = Face(np.array(C.right.array))
        return Cube(newTop, newLeft, newFront, newRight,
                    newBack, newBtm)

    def rotateFront90(C):
        newFront = NewFace.rotate90(C.front)
        newLeft = NewFace.right(C.left, C.top.btm)
        newBtm = NewFace.top(C.btm, C.left.right)
        newRight = NewFace.left(C.right, C.btm.top)
        newTop = NewFace.btm(C.top, C.right.left)
        newBack = Face(np.array(C.back.array))
        return Cube(newTop, newLeft, newFront, newRight,
                    newBack, newBtm)

    def rotateFront90CW(C):
        newFront = NewFace.rotate90CW(C.front)
        newLeft = NewFace.right(C.left, C.btm.top)
        newBtm = NewFace.top(C.btm, C.right.left)
        newRight = NewFace.left(C.right, C.top.btm)
        newTop = NewFace.btm(C.top, C.left.right)
        newBack = Face(np.array(C.back.array))
        return Cube(newTop, newLeft, newFront, newRight,
                    newBack, newBtm)

    def rotateRight90(C):
        newRight = NewFace.rotate90(C.right)
        newFront = NewFace.right(C.front, C.top.right)
        newBtm = NewFace.right(C.btm, C.front.right)
        newBack = NewFace.left(C.back, C.btm.right)
        newTop = NewFace.right(C.top, C.back.left)
        newLeft = Face(np.array(C.left.array))
        return Cube(newTop, newLeft, newFront, newRight,
                    newBack, newBtm)

    def rotateRight90CW(C):
        newRight = NewFace.rotate90CW(C.right)
        newFront = NewFace.right(C.front, C.btm.right)
        newBtm = NewFace.right(C.btm, C.back.left)
        newBack = NewFace.left(C.back, C.top.right)
        newTop = NewFace.right(C.top, C.front.right)
        newLeft = Face(np.array(C.left.array))
        return Cube(newTop, newLeft, newFront, newRight,
                    newBack, newBtm)

    def rotateBack90(C):
        newBack = NewFace.rotate90(C.back)
        newRight = NewFace.right(C.right, C.top.top)
        newBtm = NewFace.btm(C.btm, C.right.right)
        newLeft = NewFace.left(C.left, C.btm.btm)
        newTop = NewFace.top(C.top, C.left.left)
        newFront = Face(np.array(C.front.array))
        return Cube(newTop, newLeft, newFront, newRight,
                    newBack, newBtm)

    def rotateBack90CW(C):
        newBack = NewFace.rotate90CW(C.back)
        newRight = NewFace.right(C.right, C.btm.btm)
        newBtm = NewFace.btm(C.btm, C.left.left)
        newLeft = NewFace.left(C.left, C.top.top)
        newTop = NewFace.top(C.top, C.right.right)
        newFront = Face(np.array(C.front.array))
        return Cube(newTop, newLeft, newFront, newRight,
                    newBack, newBtm)

    def rotateBtm90(C):
        newBtm = NewFace.rotate90(C.btm)
        newLeft = NewFace.btm(C.left, C.front.btm)
        newBack = NewFace.btm(C.back, C.left.btm)
        newRight = NewFace.btm(C.right, C.back.btm)
        newFront = NewFace.btm(C.front, C.right.btm)
        newTop = Face(np.array(C.top.array))
        return Cube(newTop, newLeft, newFront, newRight,
                    newBack, newBtm)

    def rotateBtm90CW(C):
        newBtm = NewFace.rotate90CW(C.btm)
        newLeft = NewFace.btm(C.left, C.back.btm)
        newBack = NewFace.btm(C.back, C.right.btm)
        newRight = NewFace.btm(C.right, C.front.btm)
        newFront = NewFace.btm(C.front, C.left.btm)
        newTop = Face(np.array(C.top.array))
        return Cube(newTop, newLeft, newFront, newRight,
                    newBack, newBtm)

    def randomCube(n=5):
        returnCube = NewCube.fill()
        numRot = [random.randint(0, 11) for _ in range(n)]
        for i in numRot:
            returnCube = getattr(NewCube, NewCube.rotType[i])(returnCube)
        return returnCube

    def neighbors(C):
        return [getattr(NewCube, NewCube.rotType[i])(C)
                for i in range(len(NewCube.rotType))]

    def neighborsFrom(C, k):
        inv = 100
        if k:
            inv = (k+6) % 12
        return [getattr(NewCube, NewCube.rotType[i])(C)
                for i in range(len(NewCube.rotType))
                if i != inv]

    def arangeFill():
        return Cube(*[NewFace.arangeFill(i) for i in range(6)])



class Heuristic:

    def oneByOne(C):
        pass

class ShortestPaths:
    # with source, target, heuristic

    from math import inf

    def __init__(self, source, heuristic=None, terminate=12**4+1):
        if not heuristic:
            heuristic = lambda C: 0
            # heuristic = lambda C: 1 - (C.completeness() / 48)
        self.heuristic = heuristic
        self.currNum = 0
        self.target = None
        self.vertices = [source]
        self.verticesHashMap = {source: 0}
        self.edgeTo = [None]
        self.rotTo = [None]
        # self.fringe = PriorityQueue()
        self.fringe = Queue()
        self.search(terminate)

    @property
    def curr(self):
        return self.vertices[self.currNum]

    def addNeighbors(self, i):
        C = self.vertices[i]
        neighborsLst = NewCube.neighborsFrom(C, self.rotTo[i])
        for j in range(len(neighborsLst)):
            N = neighborsLst[j]
            if self.verticesHashMap.get(N, -1) == -1:
                vertexNum = len(self.vertices)
                self.vertices.append(N)
                self.verticesHashMap[N] = len(self.vertices) - 1
                self.edgeTo.append(i)
                self.rotTo.append(j)
                vertexPriority = self.heuristic(N)
                # self.fringe.add(vertexNum, vertexPriority)
                self.fringe.enqueue(vertexNum)

    def search(self, n):
        while self.curr.completeness() != 48 and n > 0:
            self.addNeighbors(self.currNum)
            # self.currNum = self.fringe.removeSmallest()
            self.currNum = self.fringe.dequeue()
            if n > 10**4-1 and n % 10**4 == 0:
                print(n)
            n -= 1
        if self.curr.completeness() == 48:
            self.target = self.currNum

    def path(self):
        if self.target == None:
            return None
        path = Stack()
        target = self.target
        while target != 0:
            rotToStr = NewCube.rotType[self.rotTo[target]]
            C = self.vertices[target]
            path.push((rotToStr, C))
            target = self.edgeTo[target]
        path.push((None, self.vertices[0]))
        return list(path)


def SPTF(n, m): # n=number of scrambles, m=order of 12 search vertices
    C = NewCube.fill()
    scramble = []
    for i in range(n):
        r = random.randint(0, 11)
        scramble.append(NewCube.rotType[r])
        C = getattr(NewCube, NewCube.rotType[r])(C)
    SP = ShortestPaths(C, None, 12**m)
    path = SP.path()
    if path:
        print(path[0][1])
        print([p[0] for p in path])
    else:
        print('Did not find. vertices=6**{}'.format(m))
    print(scramble)
    return path

def SPTL(permTuple):
    C = NewCube.fill()
    for i in permTuple:
        C = getattr(NewCube, NewCube.rotType[i])(C)
    print(C)
    SP = ShortestPaths(C, None, 12**6)
    path = SP.path()
    if not path:
        return None
    return [p[0] for p in path]

def test010():
    C = NewCube.fill()
    p = (0,1,0)
    for i in p:
        C = getattr(NewCube, NewCube.rotType[i])(C)
    return C

import unittest

# class FaceTest(unittest.TestCase):
#
#     def test0(self):
#         Top = NewFace.fill(0)
#         self.assertEqual(Top.completeness(), 8)
#         Top.array = np.arange(9).reshape(3,3)
#         Top.array[1,1] = Top.center
#         self.assertTrue(np.array_equal(Top.top,
#             np.array([0, 1, 2])))
#         self.assertEqual(Top.completeness(), 1)
#         # print(Top.array)
#
#     def testRotate(self):
#         a = np.arange(9).reshape(3,3)
#         F = Face(a)
#         rot90F = NewFace.rotate90(F)
#         # print('90', F.array, rot90F.array, sep='\n')
#         self.assertTrue(np.array_equal(np.rot90(a), rot90F.array))
#
#         # rot180F = NewFace.rotate180(F)
#         # # print('180', F.array, rot180F.array, sep='\n')
#         # self.assertTrue(np.array_equal(np.rot90(a, 2), rot180F.array))
#         #
#         # rot90CWF = NewFace.rotate90CW(F)
#         # # print('90CW', F.array, rot90CWF.array, sep='\n')
#         # self.assertTrue(np.array_equal(np.rot90(a, -1), rot90CWF.array))
#
# class CubeTest(unittest.TestCase):
#
#     def test0(self):
#         C = NewCube.fill()
#         self.assertEqual(C.completeness(), 48)
#         # print('fill', C)
#
# class CubeRotationsTest(unittest.TestCase):
#
#     def testTop90(self):
#         C = NewCube.fill()
#         C.top.array[0,0] = 6
#         # print('C', C)
#         D = NewCube.rotateTop90(C)
#         # print('top90', D)
#
#         newTop = np.zeros((3,3), int)
#         newTop[2,0] = 6
#         self.assertTrue(np.array_equal(D.top.array, newTop))
#
#         newLeft = np.empty((3,3), int)
#         newLeft.fill(1)
#         newLeft[0,:] = np.array([4,4,4], int)
#         self.assertTrue(np.array_equal(D.left.array, newLeft))
#
#         newBack = np.empty((3,3), int)
#         newBack.fill(4)
#         newBack[0,:] = np.array([3,3,3], int)
#         self.assertTrue(np.array_equal(D.back.array, newBack))
#
#     # def testTop180(self):
#     #     C = NewCube.fill()
#     #     C.top.array[0,0] = 6
#     #     # print('C', C)
#     #     D = NewCube.rotateTop180(C)
#     #     # print('top180', D)
#     #
#     #     newTop = np.zeros((3,3), int)
#     #     newTop[2,2] = 6
#     #     self.assertTrue(np.array_equal(D.top.array, newTop))
#     #
#     #     newLeft = np.empty((3,3), int)
#     #     newLeft.fill(1)
#     #     newLeft[0,:] = np.array([3,3,3], int)
#     #     self.assertTrue(np.array_equal(D.left.array, newLeft))
#     #
#     #     newBack = np.empty((3,3), int)
#     #     newBack.fill(4)
#     #     newBack[0,:] = np.array([2,2,2], int)
#     #     self.assertTrue(np.array_equal(D.back.array, newBack))
#     #
#     # def testTop90CW(self):
#     #     C = NewCube.fill()
#     #     C.top.array[0,0] = 6
#     #     # print('C', C)
#     #     D = NewCube.rotateTop90CW(C)
#     #     # print('top90CW', D)
#     #
#     #     newTop = np.zeros((3,3), int)
#     #     newTop[0,2] = 6
#     #     self.assertTrue(np.array_equal(D.top.array, newTop))
#     #
#     #     newLeft = np.empty((3,3), int)
#     #     newLeft.fill(1)
#     #     newLeft[0,:] = np.array([2,2,2], int)
#     #     self.assertTrue(np.array_equal(D.left.array, newLeft))
#     #
#     #     newBack = np.empty((3,3), int)
#     #     newBack.fill(4)
#     #     newBack[0,:] = np.array([1,1,1], int)
#     #     self.assertTrue(np.array_equal(D.back.array, newBack))
#
#     def testByEyeLeft90(self):
#         C = NewCube.fill()
#         C.left.array[0,0] = 6
#         # print('C', C)
#         D = NewCube.rotateLeft90(C)
#         # print('left90', D)
#
#     def testByEyeFront90(self):
#         C = NewCube.fill()
#         C.front.array[0,0] = 6
#         # print('C', C)
#         D = NewCube.rotateFront90(C)
#         # print('front90', D)
#         # C = NewCube.arangeFill()
#         # print('C', C)
#         # D = NewCube.rotateFront90(C)
#         # print('D', D)
#
#     def testByEyeRight90(self):
#         C = NewCube.fill()
#         C.right.array[0,0] = 6
#         # print('C', C)
#         D = NewCube.rotateRight90(C)
#         # print('right90', D)
#
#     def testByEyeBack90(self):
#         C = NewCube.fill()
#         C.back.array[0,0] = 6
#         # print('C', C)
#         D = NewCube.rotateBack90(C)
#         # print('back90', D)
#
#     def testByEyeBack90(self):
#         C = NewCube.fill()
#         C.btm.array[0,0] = 6
#         # print('C', C)
#         D = NewCube.rotateBtm90(C)
#         # print('btm90', D)
#
# class CubeCWRotationsTest(unittest.TestCase):
#
#     def testTop(self):
#         C = NewCube.arangeFill()
#         D = NewCube.rotateTop90(C)
#         D = NewCube.rotateTop90CW(D)
#         for s in NewCube.FaceStrs:
#             CFace = getattr(C, s.lower())
#             DFace = getattr(D, s.lower())
#             # print(CFace.array, DFace.array)
#             self.assertTrue(np.array_equal(CFace.array, DFace.array))
#
#     def testLeft(self):
#         C = NewCube.arangeFill()
#         D = NewCube.rotateLeft90(C)
#         D = NewCube.rotateLeft90CW(D)
#         for s in NewCube.FaceStrs:
#             CFace = getattr(C, s.lower())
#             DFace = getattr(D, s.lower())
#             # print(CFace.array, DFace.array)
#             self.assertTrue(np.array_equal(CFace.array, DFace.array))
#
#     def testFront(self):
#         C = NewCube.arangeFill()
#         D = NewCube.rotateFront90(C)
#         D = NewCube.rotateFront90CW(D)
#         for s in NewCube.FaceStrs:
#             CFace = getattr(C, s.lower())
#             DFace = getattr(D, s.lower())
#             # print(CFace.array, DFace.array)
#             self.assertTrue(np.array_equal(CFace.array, DFace.array))
#
#     def testRight(self):
#         C = NewCube.arangeFill()
#         D = NewCube.rotateRight90(C)
#         D = NewCube.rotateRight90CW(D)
#         for s in NewCube.FaceStrs:
#             CFace = getattr(C, s.lower())
#             DFace = getattr(D, s.lower())
#             # print(CFace.array, DFace.array)
#             self.assertTrue(np.array_equal(CFace.array, DFace.array))
#
#     def testBack(self):
#         C = NewCube.arangeFill()
#         D = NewCube.rotateBack90(C)
#         D = NewCube.rotateBack90CW(D)
#         for s in NewCube.FaceStrs:
#             CFace = getattr(C, s.lower())
#             DFace = getattr(D, s.lower())
#             # print(CFace.array, DFace.array)
#             self.assertTrue(np.array_equal(CFace.array, DFace.array))
#
#     def testBtm(self):
#         C = NewCube.arangeFill()
#         D = NewCube.rotateBtm90(C)
#         D = NewCube.rotateBtm90CW(D)
#         for s in NewCube.FaceStrs:
#             CFace = getattr(C, s.lower())
#             DFace = getattr(D, s.lower())
#             # print(CFace.array, DFace.array)
#             self.assertTrue(np.array_equal(CFace.array, DFace.array))
#
# class DynamicGraphTest(unittest.TestCase):
#
#     def testPrintRandomCube(self):
#         R = NewCube.randomCube(n=20)
#         # print('random', R)
#
#     def testPrintNeighbors(self):
#         C = NewCube.fill()
#         NLst = NewCube.neighbors(C)
#         # print('NLst')
#         # for N in NLst:
#             # print(N)

class ShortestPathsTest(unittest.TestCase):

    # def test1(self):
    #     C1 = NewCube.fill()
    #     rti = random.randint(0, 11)
    #     C1 = getattr(NewCube, NewCube.rotType[rti])(C1)
    #     SP1 = ShortestPaths(C1)
    #     path = SP1.path()
    #     # for p in path:
    #     #     print(p[0])
    #     #     print(p[1])
    #
    # def test2(self):
    #     C2 = NewCube.fill()
    #     for i in range(2):
    #         C2 = getattr(NewCube, NewCube.rotType[random.randint(0, 11)])(C2)
    #     SP1 = ShortestPaths(C2)
    #     path = SP1.path()
    #     # print([p[0] for p in path])
    #     # print(path[0][1])
    #     # print(path[-1][1])
    #
    # def test3(self):
    #     rotPerms = [(i,j) for i in range(12)
    #                       for j in range(12)]
    #     noPath = []
    #     for p in rotPerms:
    #         C = NewCube.fill()
    #         for i in p:
    #             C = getattr(NewCube, NewCube.rotType[i])(C)
    #         SP = ShortestPaths(C, None, 12**4+1)
    #         path = SP.path()
    #         if not path:
    #             noPath.append(p)
    #     # print(noPath)
    #     self.assertEqual(noPath, [])

    def test4(self):
        from itertools import product
        rotPerms = product(*[range(12) for _ in range(3)])
        noPath = []
        for p in rotPerms:
            print(p)
            C = NewCube.fill()
            for i in p:
                C = getattr(NewCube, NewCube.rotType[i])(C)
            SP = ShortestPaths(C, None, 12**6+1)
            path = SP.path()
            if not path:
                noPath.append(p)
        print('noPath for ', noPath)
        self.assertEqual(noPath, [])

    # def test5(self):
    #     C = NewCube.fill()
    #     p = (0,1,0)
    #     for i in p:
    #         C = getattr(NewCube, NewCube.rotType[i])(C)
    #     SP = ShortestPaths(C, None, 12**4+1)
    #     path = SP.path()
    #     if not path:
    #         print(p)


# 010 011 012 016   01 10   020 022 026 027 030 033 036 038 040 041 044



if __name__ == '__main__':

    unittest.main()

# def testSpeedRotateMe(arr):
#     for i in range(len(arr)):
#         Rotations.rotate90(arr[i])
#
# from numpy import array, rot90
# def testSpeedRotateNp(arr):
#     for i in range(len(arr)):
#         arr[i] = array(rot90(arr[i]))
#
# import timeit
# def timeMe():
#     return timeit.timeit(
#         setup = 'from __main__ import lstMe; ' +
#                 'from __main__ import Rotations; ' +
#                 'from __main__ import testSpeedRotateMe',
#         stmt = 'testSpeedRotateMe(lstMe)',
#         number = 100)
#
# def timeNp():
#     return timeit.timeit(
#             setup = 'from __main__ import lstNp; ' +
#                     'from numpy import array; ' +
#                     'from numpy import rot90; ' +
#                     'from __main__ import testSpeedRotateNp',
#             stmt = 'testSpeedRotateNp(lstNp)',
#             number = 100)
#
# if __name__ == '__main__':
    # import random
    # from numpy import array, reshape
    # n = 10000
    # lstMe = []
    # lstNp = []
    # for _ in range(n):
    #     aMe = array([random.randint(0,5) for _ in range(9)]).reshape(3,3)
    #     aNp = array(aMe)
    #     lstMe.append(aMe)
    #     lstNp.append(aNp)
    #
    # tMe = timeMe()
    # tNp = timeNp()
    #
    # m = min(tMe, tNp)
    # print('Me= ', tMe, tMe/m)
    # print('Np= ', tNp, tNp/m)
