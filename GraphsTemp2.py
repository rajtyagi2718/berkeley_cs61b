

import numpy as np

class Face:

    def __init__(self, center):
        self.array = np.empty((3,3), int)
        self.array.fill(center)
        self.center = self.array[1,1]

    def changeArray(self, newArr):
        self.array = newArr

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

    def changeTop(self, newArr):
        self.array[0,:] = newArr

    def changeLeft(self, newArr):
        self.array[:,0] = newArr

    def changeBtm(self, newArr):
        self.array[2,:] = newArr

    def changeRight(self, newArr):
        self.array[2,:] = newArr

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


class Cube:

    def __init__(self, Top=Face(0), Left=Face(1), Front=Face(2),
                 Right=Face(3), Back=Face(4), Btm=Face(5)):
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

    def __str__(self):
        result = ''
        for f in self.faces:
            result += f + '\n' + getattr(getattr(
                self, f), '__str__')() + '\n'
        return result

class Rotations:

    faces = ['top', 'left', 'front', 'right', 'back', 'btm']
    changes = ['changeTop', 'changeLeft', 'changeBtm', 'changeRight']

    def top(C): #CCW
        C.top.changeArray(np.rot90(C.top.array))
        leftTop = np.array(C.left.top)
        # faces = ['left', 'back', 'right', 'front']
        # for i in range(len(faces)-1):
        #     getattr(C, faces[i]).changeTop(getattr(C, faces[i+1]).top)
        C.left.changeTop(C.back.top)
        C.back.changeTop(C.right.top)
        C.right.changeTop(C.front.top)
        C.front.changeTop(leftTop)



def rotate90(arr):
    (arr[0,0], arr[2,0], arr[2,2], arr[0,2]) = (
        arr[0,2], arr[0,0], arr[2,0], arr[2,2])
    (arr[1,0], arr[2,1], arr[1,2], arr[0,1]) = (
        arr[0,1], arr[1,0], arr[2,1], arr[1,2])



# import unittest
#
# class FaceTest(unittest.TestCase):
#
#     def test0(self):
#         Top = Face(0)
#         self.assertEqual(Top.completeness(), 8)
#         Top.array = np.arange(9).reshape(3,3)
#         Top.array[1,1] = Top.center
#         self.assertTrue(np.array_equal(Top.top,
#             np.array([0, 1, 2])))
#         Top.changeBtm(np.array([-6,-7,-8]))
#         self.assertTrue(np.array_equal(Top.right,
#             np.array([2,5,-8])))
#         self.assertEqual(Top.completeness(), 1)
#         print(Top.array)
#
# class CubeTest(unittest.TestCase):
#
#     def test0(self):
#         C = Cube()
#         self.assertEqual(C.completeness(), 48)
#         print(Cube())
#
# class RotationsTest(unittest.TestCase):
#
#     def testRotate(self):
#         a = np.arange(9).reshape(3,3)
#         b = np.array(a)
#         c = np.rot90(b)
#         rotate90(a)
#         print(c, a)
#         self.assertTrue(np.array_equal(c, a))
#
#
#     def testTop(self):
#         C = Cube()
#         C.top.array[0,0] = 6
#         print(C)
#         Rotations.top(C)
#         print(C)
#         newTop = np.zeros((3,3), int)
#         newTop[2,0] = 6
#         self.assertTrue(np.array_equal(C.top.array, newTop))
#         newLeft = np.empty((3,3), int)
#         newLeft.fill(1)
#         newLeft[0,:] = np.array([4,4,4], int)
#         self.assertTrue(np.array_equal(C.left.array, newLeft))
#         newBack = np.empty((3,3), int)
#         newBack.fill(4)
#         newBack[0,:] = np.array([3,3,3], int)
#         self.assertTrue(np.array_equal(C.back.array, newBack))
#
#
# # if __name__ == '__main__':
# #
# #     unittest.main()

def testSpeedRotateMe(arr):
    for i in range(len(arr)):
        rotate90(arr[i])

from numpy import array, rot90
def testSpeedRotateNp(arr):
    for i in range(len(arr)):
        arr[i] = array(rot90(arr[i]))

import timeit
def timeMe(arr):
    return timeit.timeit(
        setup = 'from __main__ import lstMe; ' +
                'from __main__ import rotate90; ' +
                'from __main__ import testSpeedRotateMe',
        stmt = 'testSpeedRotateMe(lstMe)',
        number = 100)

def timeNp(arr):
    return timeit.timeit(
            setup = 'from __main__ import lstNp; ' +
                    'from numpy import array; ' +
                    'from numpy import rot90; ' +
                    'from __main__ import testSpeedRotateNp',
            stmt = 'testSpeedRotateNp(lstNp)',
            number = 100)

if __name__ == '__main__':
    import random
    from numpy import array, reshape
    n = 10000
    lstMe = []
    lstNp = []
    for _ in range(n):
        aMe = array([random.randint(0,5) for _ in range(9)]).reshape(3,3)
        aNp = array(aMe)
        lstMe.append(aMe)
        lstNp.append(aNp)

    tMe = timeMe(lstMe)
    tNp = timeNp(lstNp)

    m = min(tMe, tNp)
    print('Me= ', tMe, tMe/m)
    print('Np= ', tNp, tNp/m)
