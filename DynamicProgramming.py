


#####################
## Data Structures ##
#####################

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

from math import inf


####################################
## Longest Increasing Subsequence ##
####################################

def LLIS(Q):
    if not Q:
        return 0
    LLSEALengths = [1] # Longest Length Sequence Ending At
    for i in range (1, len(Q)):
        prevLesserSeqValue = [LLSEALengths[j] for j in range(i)
                                    if Q[j] <= Q[i]]
        prevLesserSeqValue.append(0)
        lengthI = 1 + max(prevLesserSeqValue)
        LLSEALengths.append(lengthI)
    return max(LLSEALengths)

def LIS(Q):
    if not Q:
        return 0
    LLSEAPrevIndices = [None] # Previous Index of LLSEA
    LLSEALengths = [1] # Longest Length Sequence Ending At
    for i in range (1, len(Q)):
        prevLesserSeqIndices = [j for j in range(i)
                                    if Q[j] <= Q[i]]
        if not prevLesserSeqIndices:
            LLSEAPrevIndices.append(None)
            LLSEALengths.append(1)
        else:
            prevIndex = max(prevLesserSeqIndices,
                            key=lambda i: LLSEALengths[i])
            prevVal = LLSEALengths[prevIndex]
            LLSEAPrevIndices.append(prevIndex)
            LLSEALengths.append(1 + prevVal)
    index =  max(list(range(len(LLSEALengths))),
                 key=lambda i: LLSEALengths[i])
    R = Stack()
    while index != None:
        R.push(Q[index])
        index = LLSEAPrevIndices[index]
    return list(R)

def MVCS(Q):
    # slice of Q[i : j] that maximizes sum
    # max value contiguous subsequence
    MSEA, MSSI = [], []
    # Max sum ending at, Max sum starting index
    # Max sum ending index[i] = i+1
    prevMSEA, prevMSSI = 0, 0
    for i in range(len(Q)):
        if prevMSEA + Q[i] >= 0:
            MSEA.append(prevMSEA + Q[i])
            MSSI.append(prevMSSI)
        else:
            MSEA.append(0)
            MSSI.append(i+1)
        prevMSEA, prevMSSI = MSEA[i], MSSI[i]
    mseai = max(range(len(Q)), key=lambda i: MSEA[i])
    mssi = MSSI[mseai]
    ms = MSEA[mseai]
    msei = mseai + 1
    return ms, Q[mssi : msei]

def makeChange(Q, v):
    # Q[i] are positive coin denominations, inc order
    # return min number of coins and their denominations for value
    if v < 0:
        return inf, []
    if v == 0:
        return 0, []
    if len(Q) <= 0:
        return inf, []
    wMCoin = makeChange(Q, v-Q[-1]) # with Max Coin
    woMCoin = makeChange(Q[:-1], v) # without Max Coin
    MCoinLst = [wMCoin, woMCoin]
    maxChangeIndex = min([0, 1],
                         key=lambda i: MCoinLst[i][0])
    returnNum, returnLst = MCoinLst[maxChangeIndex]
    if maxChangeIndex == 0:
        return returnNum + 1, returnLst + [Q[-1]]
    else:
        return returnNum, returnLst

def _stackBoxesOrder(Q):
    # R is a set (non-duplicate) of orientations of items in Q
    # r in R is of the form (height, length, width)
    # length <= width
    # view R as a list ordered by decreasing base height
    # any box stack must be a subsequence
    R = set()
    for q in Q:
        for i in range(3):
            height = q[i]
            j, k = (i-1) % 3, (i+1) % 3
            if q[j] <= q[k]:
                length, width = q[j], q[k]
            else:
                length, width = q[k], q[j]
            r = (height, length, width)
            R.add(r)
    return list(R)

def _stackBoxesDuplicates(R):
    # remove length, width duplicates in favor of greatest height
    i, l = 0, len(R)-1
    while i < l:
        # R[i] = (x, y, z), R[i+1] = (x-1, y, z)
        if (R[i][1], R[i][2]) == (R[i+1][1], R[i+1][2]):
            R.pop(i+1)
            i -= 1
            l -= 1
        i += 1
    return R

def stackBoxes(Q):
    # Q[i] = (h, w, d) are dimensions of each box
    # stack boxes so that each base dimension is smaller than previous
    # maximize height
    R = _stackBoxesOrder(Q)
    R.sort(key=lambda r: -r[1]*r[2])
    # R = _stackBoxesDuplicates(R)
    # solve subsequence problem
    if not R:
        return 0, []
    # H[i] = Max Height Subsequence Ending at index i
    H = [R[0][0]]
    # I[i] = prev index in H[i] subsequence
    I = [None]
    for i in range (1, len(R)):
        # R[j] for j in prevIs are boxes that may be directly under R[i]
        # i.e. length and width are both greater
        prevIs = [j for j in range(i)
                  if R[j][1] > R[i][1] and R[j][2] > R[i][2]]
        if not prevIs:
            H.append(R[i][0])
            I.append(None)
        else:
            index = max(prevIs, key=lambda j: H[j])
            height = H[index]
            H.append(height + R[i][0])
            I.append(index)
    index =  max(range(len(H)), key=lambda i: H[i])
    height = H[index]
    S = Stack()
    while index != None:
        S.push(R[index])
        index = I[index]
    return height, list(S)

def _buildBridgesCross(q, r):
    nDiff = q[0] - r[0]
    sDiff = q[1] - r[1]
    if nDiff * sDiff < 0:
        return True
    return False

def buildBridges(Q):
    # Q[i] = (j, k) where bridge0 connect j to k
    # sort Q to consider city pairs with leftmost north city first
    Q.sort(key=lambda q: q[0])
    N = [1] # max length subsequence ending at
    P = [None] # prev index
    for i in range(1, len(Q)):
        # I = indices of bridges to the left of Q[i] that do not cross Q[i]
        I = [j for j in range(i) if Q[i][1] > Q[j][1]]
        if not I:
            num = 1
            index = None
        else:
            index = max(I, key=lambda i: N[i])
            num = N[index] + 1
        N.append(num)
        P.append(index)
    index = max(range(len(Q)), key=lambda i: N[i])
    num = N[index]
    S = Stack()
    while index != None:
        S.push(Q[index])
        index = P[index]
    return num, list(S)

def balancedPartition(Q):
    pass

def balancedPartitionBruteForce(Q):
    from math import ceil, inf
    from itertools import combinations as comb
    minDiff = inf
    part0 = None
    s = sum(Q)
    n = ceil(len(Q) / 2)
    for pLen in range(1, n+1):
        for C in comb(Q, pLen):
            CDiff = abs(s - 2*sum(C))
            if CDiff < minDiff:
                minDiff = CDiff
                part0 = C
    part1 = [i for i in Q if i not in set(part0)]
    part1.sort()
    part0 = sorted(part0)
    return minDiff, part0, part1

import unittest, random
#
# class LLISTest(unittest.TestCase):
#
#     def testExample0(self):
#         Q = [8, 2, 9, 4, 5, 7, 3]
#         LLISQ = LLIS(Q)
#         shouldBe = 4
#         self.assertEqual(LLISQ, shouldBe)
#
# class LISTest(unittest.TestCase):
#
#     def testExample0(self):
#         Q = [8, 2, 9, 4, 5, 7, 3]
#         R = LIS(Q)
#         shouldBe = [2, 4, 5, 7]
#         self.assertEqual(R, shouldBe)
#
#     def testRandomExamplePrint(self):
#         Q = [random.randint(-50, 50) for _ in range(100)]
#         R = LIS(Q)
#         # print(Q, R, sep='\n')
#
# class MakeChangeTest(unittest.TestCase):
#
#     def testExample0(self):
#         Q = [1, 5, 10, 25]
#         v = 139
#         result = makeChange(Q, v)
#         # print(result)
#
#     def testRandom(self):
#         Q = [random.randint(1, 25) for _ in range(4)]
#         m = max(Q)
#         v = random.randint(3*m, 5*m)
#         result = makeChange(Q, v)
#         # print(result)
#
# class MVCSTest(unittest.TestCase):
#
#     def thetaNSquaredSol(lst):
#         mx, mi, mj = 0, 0, 0
#         for i in range(len(lst)):
#             for j in range(i+1, len(lst)+1):
#                 x = sum(lst[i:j])
#                 if x > mx:
#                     mx, mi, mj = x, i, j
#         return mx, lst[mi : mj]
#
#     def testRandom(self):
#         Q = [random.randint(-50, 50) for _ in
#              range(random.randint(5, 8))]
#         # print(Q)
#         result = MVCS(Q)
#         # print('result', result)
#         actual = MVCSTest.thetaNSquaredSol(Q)
#         # print('actual', actual)
#         self.assertEqual(result, actual)
#
# class StackBoxesTest(unittest.TestCase):
#
#     def stackBoxesBruteForce(Q):
#         from itertools import permutations as perms
#         R = []
#         for q in Q:
#             R.extend(list(perms(q)))
#         print('R', R)
#         maxHeight = 0
#         maxStack = []
#         count = 0
#         for pLen in range(1, len(R)):
#             for p in perms(R, pLen):
#                 if count > 10**7:
#                     for i in range(len(maxStack)):
#                         if maxStack[i][1] > maxStack[i][2]:
#                             maxStack[i] = (maxStack[i][0], maxStack[i][2], maxStack[i][1])
#                     return count, maxHeight, maxStack
#                 count += 1
#                 isStack = True
#                 for i in range(len(p)-1):
#                     if p[i][1] <= p[i+1][1]:
#                         isStack = False
#                         break
#                     if p[i][2] <= p[i+1][2]:
#                         isStack = False
#                         break
#                 if isStack:
#                     height = sum([x[0] for x in p])
#                     if height > maxHeight:
#                         maxHeight = height
#                         maxStack = list(p)
#         for i in range(len(maxStack)):
#             if maxStack[i][1] > maxStack[i][2]:
#                 maxStack[i] = (maxStack[i][0], maxStack[i][2], maxStack[i][1])
#         return 0, maxHeight, maxStack
#
#     # def test(self):
#     #     Q = [tuple(random.randint(1,9) for _ in range(3)) for _ in range(7)]
#     #     # print('Q', Q)
#     #     S = stackBoxes(Q)
#     #     # print('S', S)
#     #     B = StackBoxesTest.stackBoxesBruteForce(Q)
#     #     # print('B', B)
#     #     self.assertEqual((B[1], B[2]), S)
#
#     # def testExample(self):
#     #     Q = [(4,7,9),(9,4,7),(9,3,5),(8,2,4),(5,6,7)]
#     #     print('Q', Q)
#     #     S = stackBoxes(Q)
#     #     print('S', S)
#     #     B = StackBoxesTest.stackBoxesBruteForce(Q)
#     #     print('B', B)
#     #     self.assertEqual((B[1], B[2]), S)
#
# class BuildBridgesTest(unittest.TestCase):
#
#     def cross(q, r):
#         nDiff = q[0] - r[0]
#         sDiff = q[1] - r[1]
#         if nDiff * sDiff < 0:
#             return True
#         return False
#
#     def buildBridgesBruteForce(Q):
#         from itertools import combinations as comb
#         numCrossCalls = 0
#         for combLength in range(len(Q), 0, -1):
#             for C in comb(Q, combLength):
#                 hasCrossing = False
#                 for q, r in comb(C, 2):
#                     numCrossCalls += 1
#                     if BuildBridgesTest.cross(q, r):
#                         hasCrossing = True
#                         break
#                     # if numCrossCalls > 10**7-1 and numCrossCalls % 10**7 == 0:
#                         # print(numCrossCalls)
#                 if not hasCrossing:
#                     return combLength, list(C)
#
#     def test(self):
#         numPairs = 15
#         Qn = random.sample(range(numPairs*5), numPairs)
#         Qs = random.sample(range(numPairs*5), numPairs)
#         Q = list(zip(Qn, Qs))
#         # print('Q', Q)
#         num, B = buildBridges(Q)
#         # print('B', B)
#         self.assertEqual(num, len(B))
#
#         from itertools import combinations as comb
#         for q, r in comb(B, 2):
#             self.assertFalse(BuildBridgesTest.cross(q, r))
#
#         # bruteNum, bruteB = BuildBridgesTest.buildBridgesBruteForce(Q)
#         # print('brute', bruteB)
#         # self.assertEqual(bruteNum, num)
#         # self.assertEqual(set(B), set(bruteB))
#
#         import matplotlib.pyplot as plt, numpy as np
#         plt.title('Map')
#
#         # northern Bank
#         Xn = [q[0] for q in Q]
#         Yn = [1 for _ in Q]
#         plt.scatter(Xn, Yn, linewidth=2, color='red', label='N bank')
#         for i in range(len(Xn)):
#             plt.annotate('n{}'.format(i), xy=(Xn[i], 1), xytext=(0,0), textcoords='offset points')
#
#         # southern Bank
#         Xs = [q[1] for q in Q]
#         Ys = [-1 for _ in Q]
#         plt.scatter(Xs, Ys, linewidth=2, color='blue', label='S bank')
#         for i in range(len(Xs)):
#             plt.annotate('s{}'.format(i), xy=(Xs[i], -1), xytext=(0,0), textcoords='offset points')
#
#         # River
#         Xr = [min(Xn + Xs), max(Xn + Xs)]
#         Yr = [0 for _ in range(2)]
#         plt.plot(Xr, Yr, linewidth=20, color='green', label='river')
#
#         # Bridges
#         for bridge in B:
#             n, s = bridge
#             m = (n - s) / 2
#             xInt = (n + s) / 2
#             equation = lambda y: m*y + xInt
#             Yb = np.arange(-1, 1, .1)
#             Xb = equation(Yb)
#             plt.plot(Xb, Yb, linewidth=1, color='black', label='bridge')
#         # plt.legend(loc='upper left')
#         plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
#         # plt.show()

class BalancedPartitionTest(unittest.TestCase):

    def test(self):
        # n = 100
        # Q = random.sample(range(n), random.randint(int(.1*n), int(.7*n)))
        Q = random.sample(range(100), 20)
        # A = balancedPartition(Q)
        B = balancedPartitionBruteForce(Q)
        print(B)


if __name__ == '__main__':
    unittest.main()
