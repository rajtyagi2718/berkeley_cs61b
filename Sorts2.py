########################
## SORTING ALGORITHMS ##
########################

##############
## Preamble ##
##############

def argMax(lst):
    return max(enumerate(lst), key=lambda x: x[1])[0]

def argMin(lst):
    return min(enumerate(lst), key=lambda x: x[1])[0]

def swapIndices(lst, i, j):
    lst[i], lst[j] = lst[j], lst[i]

def minRange(lst, start, stop):
    result = lst[start]
    for i in range(start+1, stop):
        if lst[i] > result:
            result = lst[i]
    return result


from math import ceil
import random
import unittest
from statistics import median

from collections import deque

######################
## Comparison Sorts ##
######################

# SELECTION
# find smallest, swap to front, fix item
# N^2
def selectionSortInPlace(lst):
    for i in range(len(lst)-1):
        m = argMin(lst[i : ]) + i
        swapIndices(lst, i, m)


# HEAP
# max heapify array
# swap max with last item
# fix last item, sink first item as a heap array of length -1
# N log N

def _maxChildIndex(lst, i, L):
    l = 2*i + 1
    r = l+1
    if l >= L:
        return None
    if r >= L:
        return l
    return max(l, r, key=lambda i: lst[i])

def _sink(lst, i, L=None):
    if L == None:
        L = len(lst)
    m = _maxChildIndex(lst, i, L)
    if m and lst[m] > lst[i]:
        swapIndices(lst, i, m)
        _sink(lst, m, L)

def heapify(lst):
    for i in range(ceil((len(lst)-2)/2), -1, -1):
        _sink(lst, i)

def heapSortInPlace(lst):
    heapify(lst)
    for i in range(len(lst)):
        L = len(lst) - i
        swapIndices(lst, 0, L-1)
        _sink(lst, 0, L-1)


# MERGE
# sort each half recursively
# merge by comparing first items
# N log N

def mergeSort(lst):
    if len(lst) <= 1:
        return lst
    left, right = lst[:len(lst) // 2], lst[len(lst) // 2: ]
    left, right = mergeSort(left), mergeSort(right)
    returnLst = []
    l, r = 0, 0
    while l < len(left) and r < len(right):
        if left[l] < right[r]:
            returnLst.append(left[l])
            l += 1
        else:
            returnLst.append(right[r])
            r += 1
    if l != len(left):
        returnLst.extend(left[l : ])
    elif r!= len(right):
        returnLst.extend(right[r : ])
    return returnLst

def mergeSortAdaptive1(lst):
    if len(lst) <= 1:
        return lst
    left, right = lst[:len(lst) // 2], lst[len(lst) // 2: ]
    left, right = mergeSortAdaptive1(left), mergeSortAdaptive1(right)
    if left == []: # len(right) >= len(left) i.e. len(right) >= 1
        return right
    if left[-1] <= right[0]:
        return left + right
    returnLst = []
    l, r = 0, 0
    while l < len(left) and r < len(right):
        if left[l] < right[r]:
            returnLst.append(left[l])
            l += 1
        else:
            returnLst.append(right[r])
            r += 1
    if l != len(left):
        returnLst.extend(left[l : ])
    elif r!= len(right):
        returnLst.extend(right[r : ])
    return returnLst

# INSERTION
# swap left until greater than all to the left
# best for small n or almost sorted
# 1 to N^2

def insertionSortInPlace(lst):
    for i in range(1, len(lst)):
        for j in range(i, 0, -1):
            if lst[j] < lst[j-1]:
                swapIndices(lst, j, j-1)
            else:
                break


# SHELLS
# insertion has stride 1, generalize strides
# 1 to  N^1.5

def strideSwap(lst, k):
    for i in range(k, len(lst)):
        for j in range(i, 0, -k):
            if lst[j] < lst[j-k]:
                swapIndices(lst, j, j-k)
            else:
                break

def greatestExponent(n, k): # n >= 0, k > 1, result k**i <= n
    i = -1
    prev = 1
    while prev <= n:
        i += 1
        prev *= k
    return i

def ShellsSortInPlace(lst, strideLst=None):
    if strideLst == None:
        g = greatestExponent(len(lst), 2)
        strideLst = [2**k-1 for k in range(g, 0, -1)]
    for s in strideLst:
        strideSwap(lst, s)


# QUICK
# declare a pivot, place in correct position (all left <=, all right >=)
# recursively sort each half
# fastest
# N log N to N^2

def quickSortInPlace(lst, start=0, end=None):
    # pivot always first item
    if end == None:
        end = len(lst)
    if end - start <= 1:
        return
    pivot = start
    for i in range(start+1, end):
        if lst[i] < lst[pivot]:
            for j in range(i, pivot, -1):
                swapIndices(lst, j, j-1)
            pivot += 1
    quickSortInPlace(lst, start, pivot)
    quickSortInPlace(lst, pivot+1, end)

# pick random element as pivot, swap to front, proceed as usual
def quickSortInPlaceRandomPivot(lst, start=0, end=None):
    if end == None:
        end = len(lst)
    if end - start <= 1:
        return
    pivot = random.randint(start, end-1)
    rightStart = pivot+1
    i = start
    while i < pivot: # swapping forward brings next lst[i]
        if lst[i] > lst[pivot]:
            for j in range(i, pivot, 1):
                swapIndices(lst, j, j+1)
            pivot -= 1
        else:
            i += 1
    for i in range(rightStart, end):
        if lst[i] < lst[pivot]:
            for j in range(i, pivot, -1):
                swapIndices(lst, j, j-1)
            pivot += 1
    quickSortInPlaceRandomPivot(lst, start, pivot)
    quickSortInPlaceRandomPivot(lst, pivot+1, end)

# allow for pivot range containing all equal elements
def quickSortInPlaceRandomPivotThreeWay(lst, start=0, end=None):
    # print('\n', lst, sep='')
    if end == None:
        end = len(lst)
    if end - start <= 1:
        return
    pivotLeft = random.randint(start, end-1)
    pivotRight = pivotLeft+1
    # print(' '*(1+start*3), 's', ' '*((end-start)*3-1), 'e', sep='')
    # print(' '*(1+pivotLeft*3), 'l', ' '*((pivotRight-pivotLeft)*3-1), 'r', sep='')
    i = start
    while i < pivotLeft: # swapping forward brings next lst[i]
        # print(' '*(1+i*3), 'i', sep='')
        if lst[i] > lst[pivotLeft]:
            for j in range(i, pivotLeft-1, 1): # swaps to pivotLeft-1, then swap with pivotRight-1
                swapIndices(lst, j, j+1)
            swapIndices(lst, pivotLeft-1, pivotRight-1)
            pivotLeft -= 1
            pivotRight -= 1
            # print(lst)
            # print(' '*(1+pivotLeft*3), 'l', ' '*((pivotRight-pivotLeft)*3-1), 'r', sep='')
        elif lst[i] == lst[pivotLeft]:
            for j in range(i, pivotLeft-1, 1): # do not swap pivotLeft
                swapIndices(lst, j, j+1)
            pivotLeft -= 1
            # print(lst)
            # print(' '*(1+pivotLeft*3), 'l', ' '*((pivotRight-pivotLeft)*3-1), 'r', sep='')
        else:
            i += 1
    # print(' '*(1+pivotLeft*3), 'l', ' '*((pivotRight-pivotLeft)*3-1), 'r', sep='')
    for i in range(pivotRight, end):
        # print(' '*(1+i*3), 'i', sep='')
        if lst[i] < lst[pivotLeft]:
            for j in range(i, pivotRight, -1):
                swapIndices(lst, j, j-1)
            swapIndices(lst, pivotLeft, pivotRight)
            pivotLeft += 1
            pivotRight += 1
            # print(lst)
            # print(' '*(1+pivotLeft*3), 'l', ' '*((pivotRight-pivotLeft)*3-1), 'r', sep='')
        elif lst[i] == lst[pivotLeft]:
            for j in range(i, pivotRight, -1): # swap pivotRight, since out of = group
                swapIndices(lst, j, j-1)
            pivotRight += 1
            # print(lst)
            # print(' '*(1+pivotLeft*3), 'l', ' '*((pivotRight-pivotLeft)*3-1), 'r', sep='')
    quickSortInPlaceRandomPivotThreeWay(lst, start, pivotLeft)
    quickSortInPlaceRandomPivotThreeWay(lst, pivotRight, end)

# one pivot at front, small and big pivots start on ends
# small travels left, large travels right
# pointers stop when reach >= or <= element (resp)
# swap, advance pointers, repeat
# when pointers cross, swap pivot to big position
def quickSortInPlaceTwoPointersRandomPivot(lst, start=0, end=None):
    # print('\n', lst, sep='')
    if end == None:
        end = len(lst)
    if end - start <= 1:
        return
    pivot = random.randint(start, end-1)
    # print('pivot=', lst[pivot])
    swapIndices(lst, start, pivot)
    # print(lst)
    pivot = start
    small = pivot + 1
    big = end - 1
    # print(' '*(1+small*3), 's', ' '*((big-small)*3-1), 'b', sep='')
    count = 0
    while True:
        while small < end and lst[small] < lst[pivot]:
            small += 1
            # print(' '*(1+small*3), 's', sep='')
        while big > start and lst[big] > lst[pivot]:
            big -= 1
            # print(' '*(1+big*3), 'b', sep='')
        count += 1
        if small >= big:
            break
        swapIndices(lst, small, big)
        small += 1
        big -= 1
        # print(lst)
        # print(' '*(1+small*3), 's', sep='')
        # print(' '*(1+big*3), 'b', sep='')
    swapIndices(lst, pivot, big)
    pivot = big
    # print(lst)
    quickSortInPlaceTwoPointersRandomPivot(lst, start, pivot)
    quickSortInPlaceTwoPointersRandomPivot(lst, pivot+1, end)

# find median in N to N^2 time
# place random pivot in front
# partition around pivot
# repeat to left or right of pivot, if pivot is right or left of median position
def quickSelect(lst, start=0, end=None):
    if end == None:
        end = len(lst)
    pivot = random.randint(start, end-1)
    swapIndices(lst, start, pivot)
    pivot = start
    small = pivot + 1
    big = end - 1
    count = 0
    while True:
        while small < end and lst[small] < lst[pivot]:
            small += 1
        while big > start and lst[big] > lst[pivot]:
            big -= 1
        count += 1
        if small >= big:
            break
        swapIndices(lst, small, big)
        small += 1
        big -= 1
    swapIndices(lst, pivot, big)
    pivot = big
    m = (len(lst) - 1) / 2
    if pivot < m:
        if pivot == m - .5:
            sMed = lst[pivot]
            lMed = min(lst[pivot+1: ])
            return (sMed + lMed) / 2
        return quickSelect(lst, pivot, end)
    if pivot > m:
        if pivot == m + .5:
            lMed = lst[pivot]
            sMed = max(lst[ : pivot])
            return (sMed + lMed) / 2
        return quickSelect(lst, start, pivot)
    return lst[pivot]



####################
## Counting Sorts ##
####################

def counts(data, aMap, k=0): #d[k] in alaphabet for d in data
    result = [0 for _ in range(len(aMap))]
    for d in data:
        result[aMap[d[k]]] += 1
    return result

def startingPositions(lst):
    sum = 0
    for i in range(len(lst)):
        lst[i], sum = sum, sum + lst[i]

def countingSort(alphabet, data, k=0):
    aMap = {k:v for v,k in enumerate(alphabet)}
    positions = counts(data, aMap, k)
    startingPositions(positions)
    result = [None for _ in range(len(data))]
    for d in data:
        i = aMap[d[k]]
        p = positions[i]
        result[p] = d
        positions[i] += 1
    return result

def countingSort2(data, k=0):
    aMap0 = {}
    positions0 = []
    for d in data: #aMap and positions always = len
        a = d[k]
        i = aMap0.get(a)
        if i == None:
            i = len(aMap0)
            aMap0[a] = i
            positions0.append(0)
        positions0[i] += 1
    #sort alphabet using any comparison sort
    alphabet = sorted(aMap0.keys())
    aMap = {k:v for v,k in enumerate(alphabet)}
    positions = [positions0[aMap0[alphabet[i]]] for i in range(len(positions0))]
    sum = 0
    for i in range(len(positions)):
        positions[i], sum = sum, sum + positions[i]
    result = [None for _ in range(len(data))]
    for d in data:
        i = aMap[d[k]]
        p = positions[i]
        result[p] = d
        positions[i] += 1
    return result

# LEAST SIGNIFICANT DIGIT
def LSDRadixSort(data, alphabet, length=None): #str entries
    if length == None:
        length = len(max(data, key=len))
    for k in range(length):
        data = _RadixSortIndex(data, alphabet, -(k+1))
    return data

def _RadixSortIndex(data, alphabet, k):
    aMap = {k:v+1 for v,k in enumerate(alphabet)}
    aMap[None] = 0
    positions = [0 for _ in range(len(aMap))]
    for d in data:
        if k < -len(d) or len(d)-1 < k:
            positions[aMap[None]] += 1
        else:
            positions[aMap[d[k]]] += 1
    sum = 0
    for i in range(len(positions)):
        positions[i], sum = sum, sum + positions[i]
    result = [None for _ in range(len(data))]
    for d in data:
        if k < -len(d) or len(d)-1 < k:
            i = aMap[None]
        else:
            i = aMap[d[k]]
        p = positions[i]
        result[p] = d
        positions[i] += 1
    return result

# MOST SIGNIFICANT DIGIT
def MSDRadixSort(data, alphabet, length=None): #str entries
    if length == None:
        length = len(max(data, key=len))
    aMap = {k:v+1 for v,k in enumerate(alphabet)}
    aMap[None] = 0
    return _MSDRSIndex(data, aMap, -length)

def _MSDRSIndexCounts(data, aMap, k):
    counts = [0 for _ in range(len(aMap))]
    for d in data:
        if k < -len(d) or len(d)-1 < k:
            counts[aMap[None]] += 1
        else:
            counts[aMap[d[k]]] += 1
    return counts

def _MSDRIndexPositions(counts):
    positions = []
    sum = 0
    for i in range(len(counts)):
        positions.append(sum)
        sum += counts[i]
    return positions

def _MSDRSIndex(data, aMap, k):
    counts = _MSDRSIndexCounts(data, aMap, k)
    positions = _MSDRIndexPositions(counts)
    currPositions = list(positions)
    result = [None for _ in range(len(data))]
    for d in data:
        if k < -len(d) or len(d)-1 < k:
            i = aMap[None]
        else:
            i = aMap[d[k]]
        p = currPositions[i]
        result[p] = d
        currPositions[i] += 1
    if k < -1:
        positions.append(len(data))
        for i in range(len(counts)):
            if counts[i] > 1:
                start, stop = positions[i], positions[i+1]
                data = result[start: stop]
                result[start: stop] = _MSDRSIndex(data, aMap, k+1)
    return result

# MOST SIGNIFICANT DIGIT w Recursion (could do with Queues instead of lists)
def MSDRadixRecursionSort(data, alphabet, length=None): #str entries
    if length == None:
        length = len(max(data, key=len))
    aMap = {k:v+1 for v,k in enumerate(alphabet)}
    aMap[None] = 0
    lst = list(range(len(data)))
    lst = _MSDRRS(data, aMap, lst, -length)
    result = [data[i] for i in lst]
    return result

def _MSDRRS(data, aMap, lst, k):
    L = [[] for _ in range(len(aMap))]
    for i in lst:
        d = data[i]
        if k < -len(d) or len(d)-1 < k:
            j = aMap[None]
        else:
            j = aMap[d[k]]
        L[j].append(i)
    result = []
    if k < -1:
        for i in range(len(L)):
            lst = L[i]
            if len(lst) > 1:
                L[i] = _MSDRRS(data, aMap, lst, k+1)
    for lst in L:
        result.extend(lst)
    return result



##########################
## Ternary Search Tries ## data structure analogous to radix sorts
##########################
class Node:

    def __init__(self, key=None, value=None,
                 left=None, mid=None, right=None,
                 isLast=False):
        self.key = key
        self.value = value
        self.left = left
        self.mid = mid
        self.right = right
        self.isLast = False

class TST:

    def __init__(self, root=None, size=0):
        self.root = root
        self.size = size

    def _checkKey(key):
        try:
            k = key[0]
        except:
            if len(key) == 0:
                raise IndexError('key must have length > 0')
            raise TypeError('key must be indexable')

    def contains(self, key):
        TST._checkKey(key)
        return TST._getNode(self.root, key) != None

    def get(self, key):
        TST._checkKey(key)
        node = TST._getNode(self.root, key)
        if node != None:
            return node.value
        return None

    def _getNode(node, key):
        return TST._getNodeDigit(node, key, 0)

    def _getNodeDigit(node, key, i):
        if node == None:
            return None
        if node.key < key[i]:
            return TST._getNodeDigit(node.left, key, i)
        if node.key > key[i]:
            return TST._getNodeDigit(node.right, key, i)
        if i == len(key)-1:
            return node
        return TST._getNodeDigit(node.mid, key, i+1)

    def insert(self, key, value=None): #type(s) == str
        TST._checkKey(key)
        if self.root == None:
            self.root = Node()
        self._insertDigit(key, value, 0, self.root)

    # node cannot equal None, else no pointer to object
    def _insertDigit(self, key, value, i, node):
        # children are None
        if node.key == None:
            node.key = key[i]
        # children are nodes, possibly empty
        if node.key == key[i]:
            if i == len(key)-1:
                if not node.isLast:
                    node.value = value
                    node.isLast = True
                    self.size += 1
            else:
                if node.mid == None:
                    node.mid = Node()
                self._insertDigit(key, value, i, node.mid)
        elif node.key < key[i]:
            if node.left == None:
                node.left = Node()
            self._insertDigit(key, value, i, node.left)
        else: # node.key > s[i]:
            if node.right == None:
                node.right = Node()
            self._insertDigit(key, value, i, node.right)

###########
## Tests ##
###########

class PreambleTest(unittest.TestCase):

    def test(self):
        m, M = random.sample(range(10), 2)
        if m > M:
            m, M = M, m
        lst = [random.randint(0, 9) for _ in range(10)]
        lst[m], lst[M] = -1, 10
        # print('preamble', lst, m, M)
        self.assertEqual(argMin(lst), m)
        self.assertEqual(argMax(lst), M)
        swapIndices(lst, m, M)
        self.assertEqual(lst[m], 10)
        self.assertEqual(lst[M], -1)

class SelectionTest(unittest.TestCase):

    def test(self):
        for _ in range(10):
            lst = [random.randint(0, 9) for _ in range(10)]
            s = sorted(lst)
            selectionSortInPlace(lst)
            self.assertEqual(lst, s)
        for _ in range(10):
            lst = [random.randint(0, 9) for _ in range(30)]
            s = sorted(lst)
            heapSortInPlace(lst)
            self.assertEqual(lst, s)

class HeapTest(unittest.TestCase):

    def testHeapify(self):
        lst = [32, 15, 2, 17, 19, 26, 41, 17, 17]
        heapify(lst)
        h = [41, 19, 32, 17, 15, 26, 2, 17, 17]
        self.assertEqual(lst, h)

    def testSortExample(self):
        lst = [32, 15, 2, 17, 19, 26, 41, 17, 17]
        heapSortInPlace(lst)
        h = [2, 15, 17, 17, 17, 19, 26, 32, 41]
        self.assertEqual(lst, h)

    def testSortExample2(self):
        lst = [5, 5, 1, 4, 8, 5, 3, 8, 2, 6]
        h = sorted(lst)
        heapSortInPlace(lst)
        self.assertEqual(lst, h)

    def testSort(self):
        for _ in range(10):
            lst = [random.randint(0, 9) for _ in range(10)]
            s = sorted(lst)
            heapSortInPlace(lst)
            self.assertEqual(lst, s)
        for _ in range(10):
            lst = [random.randint(0, 9) for _ in range(30)]
            s = sorted(lst)
            heapSortInPlace(lst)
            self.assertEqual(lst, s)

class MergeSortTest(unittest.TestCase):

    def testSort(self):
        for _ in range(10):
            lst = [random.randint(0, 9) for _ in range(10)]
            ms = mergeSort(lst)
            s = sorted(lst)
            self.assertEqual(ms, s)
        for _ in range(10):
            lst = [random.randint(0, 9) for _ in range(30)]
            ms = mergeSort(lst)
            s = sorted(lst)
            self.assertEqual(ms, s)

    def testSortAdaptive(self):
        for _ in range(10):
            lst = [random.randint(0, 9) for _ in range(10)]
            msa = mergeSortAdaptive1(lst)
            s = sorted(lst)
            self.assertEqual(msa, s)
        for _ in range(10):
            lst = [random.randint(0, 9) for _ in range(30)]
            msa = mergeSortAdaptive1(lst)
            s = sorted(lst)
            self.assertEqual(msa, s)

class InsertionSortTest(unittest.TestCase):

    def testSort(self):
        for _ in range(10):
            lst = [random.randint(0, 9) for _ in range(10)]
            s = sorted(lst)
            insertionSortInPlace(lst)
            self.assertEqual(lst, s)
        for _ in range(10):
            lst = [random.randint(0, 9) for _ in range(30)]
            s = sorted(lst)
            insertionSortInPlace(lst)
            self.assertEqual(lst, s)

class ShellsSortTest(unittest.TestCase):

    def testStrideSort(self):
        for _ in range(10):
            lst = [random.randint(0, 9) for _ in range(10)]
            s = sorted(lst)
            strideSwap(lst, 1)
            self.assertEqual(lst, s)
        for _ in range(10):
            lst = [random.randint(0, 9) for _ in range(30)]
            s = sorted(lst)
            strideSwap(lst, 1)
            self.assertEqual(lst, s)

    def testExp(self):
        for _ in range(10**2):
            n = random.randint(0, 10**5)
            k = random.randint(2, n // 2)
            i = greatestExponent(n, k)
            self.assertTrue(k**i <= n)
            self.assertTrue(k**(i+1) > n)

    def testSort(self):
        for _ in range(10):
            lst = [random.randint(0, 9) for _ in range(10)]
            s = sorted(lst)
            ShellsSortInPlace(lst)
            self.assertEqual(lst, s)
        for _ in range(10):
            lst = [random.randint(0, 9) for _ in range(30)]
            s = sorted(lst)
            ShellsSortInPlace(lst)
            self.assertEqual(lst, s)

class quickSortTest(unittest.TestCase):

    def testSort(self):
        for _ in range(10):
            lst = [random.randint(0, 9) for _ in range(10)]
            s = sorted(lst)
            quickSortInPlace(lst)
            self.assertEqual(lst, s)
        for _ in range(10):
            lst = [random.randint(0, 9) for _ in range(30)]
            s = sorted(lst)
            quickSortInPlace(lst)
            self.assertEqual(lst, s)

    def testRandomPivotSort(self):
        for _ in range(10):
            lst = [random.randint(0, 9) for _ in range(10)]
            s = sorted(lst)
            quickSortInPlaceRandomPivot(lst)
            self.assertEqual(lst, s)
        for _ in range(10):
            lst = [random.randint(0, 9) for _ in range(30)]
            s = sorted(lst)
            quickSortInPlaceRandomPivot(lst)
            self.assertEqual(lst, s)

    def testRandomPivotThreeWay(self):
        for _ in range(10):
            lst = [random.randint(0, 9) for _ in range(10)]
            lstCopy = list(lst)
            s = sorted(lst)
            quickSortInPlaceRandomPivotThreeWay(lst)
            if s != lst:
                print('original: ', lstCopy)
            self.assertEqual(lst, s)
        for _ in range(10):
            lst = [random.randint(0, 9) for _ in range(30)]
            s = sorted(lst)
            quickSortInPlaceRandomPivotThreeWay(lst)
            self.assertEqual(lst, s)

    def testTwoPointersRandomPivot(self):
        for _ in range(10):
            lst = [random.randint(0, 9) for _ in range(10)]
            lstCopy = list(lst)
            s = sorted(lst)
            quickSortInPlaceTwoPointersRandomPivot(lst)
            if s != lst:
                print('original: ', lstCopy)
            self.assertEqual(lst, s)
        for _ in range(10):
            lst = [random.randint(0, 9) for _ in range(30)]
            s = sorted(lst)
            quickSortInPlaceTwoPointersRandomPivot(lst)
            self.assertEqual(lst, s)

    def testMedian(self):
        for _ in range(10):
            for l in range(1, 10):
                lst = [random.randint(0, 9) for _ in range(l)]
                statsMed = median(lst)
                m = quickSelect(lst)
                self.assertEqual(m, statsMed)
        lst = [random.randint(0, 9)]
        statsMed = median(lst)
        m = quickSelect(lst)
        self.assertEqual(m, statsMed)
        lst = [random.randint(0, 9) for _ in range(2)]
        statsMed = median(lst)
        m = quickSelect(lst)
        self.assertEqual(m, statsMed)

class CountingSortTest(unittest.TestCase):

    def testCounts(self):
        alphabet = random.sample('abcdefghij', random.randint(3, 8))
        data = [[random.randint(0, 100) for _ in range(10)] for _ in range(100)]
        k = random.randint(0, 10)
        C0 = [0 for _ in range(len(alphabet))]
        for d in data:
            a = random.sample(alphabet, 1)[0]
            C0[alphabet.index(a)] += 1
            d.insert(k, a)
        aMap = {}
        for i in range(len(alphabet)):
            aMap[alphabet[i]] = i
        C1 = counts(data, aMap, k)
        self.assertEqual(C0, C1)

    def testSort(self):
        for _ in range(10):
            alphabet = random.sample('abcdefghij', random.randint(3, 8))
            data = [[random.randint(0, 100) for _ in range(10)] for _ in range(100)]
            k = random.randint(0, 10)
            for d in data:
                a = random.sample(alphabet, 1)[0]
                d.insert(k, a)
            cs = countingSort(alphabet, data, k)
            aMap = {}
            for i in range(len(alphabet)):
                aMap[alphabet[i]] = i
            s = sorted(data, key=lambda x: aMap[x[k]])
            self.assertEqual(cs, s)

    def testSort2(self):
        for _ in range(10):
            alphabet = random.sample('abcdefghij', random.randint(3, 8))
            alphabet.sort()
            data = [[random.randint(0, 100) for _ in range(10)] for _ in range(10)]
            k = random.randint(0, 10)
            for d in data:
                a = random.sample(alphabet, 1)[0]
                d.insert(k, a)
            cs2 = countingSort2(data, k)
            s = sorted(data, key=lambda x: x[k])
            self.assertEqual(cs2, s)

class RadixSortTest(unittest.TestCase):

    def testIndexSortEqualLength(self):
        for _ in range(10):
            alphabet = random.sample('abcdefghij', random.randint(3, 8))
            data = [[random.randint(0, 100) for _ in range(10)] for _ in range(100)]
            k = random.randint(0, 10)
            for d in data:
                a = random.sample(alphabet, 1)[0]
                d.insert(k, a)
            rsi = _RadixSortIndex(data, alphabet, k)
            aMap = {}
            for i in range(len(alphabet)):
                aMap[alphabet[i]] = i
            s = sorted(data, key=lambda x: aMap[x[k]])
            self.assertEqual(rsi, s)

    def testIndexSort(self):
        alphabet = list('0123456789')
        data = random.sample(range(0, 100), 10)
        data.extend(random.sample(range(100, 10**6), 10))
        data = list(map(str, data))
        k = -3
        rsi = _RadixSortIndex(data, alphabet, k)
        f = lambda x: '-1' if -len(x) > k else x[k]
        s = sorted(data, key=f)
        self.assertEqual(rsi, s)

    def testLSDSort(self):
        for _ in range(10):
            alphabet = list('0123456789')
            data = random.sample(range(0, 10**6), 100)
            s = list(map(str, sorted(data)))
            data = list(map(str, data))
            lsdrs = LSDRadixSort(data, alphabet)
            self.assertEqual(lsdrs, s)

    def testMSDSort(self):
        for _ in range(10):
            alphabet = list('0123456789')
            data = random.sample(range(0, 10**6), 10)
            s = list(map(str, sorted(data)))
            data = list(map(str, data))
            msdrs = MSDRadixSort(data, alphabet)
            # print('data', *data, sep='\n')
            # print('msdrs', *msdrs, sep='\n')
            # print('s', *s, sep='\n')
            self.assertEqual(msdrs, s)
        alphabet = list('0123456789')
        data = []
        for i in range(1, 6):
            data.extend(random.sample(range(0, 10**i), 5))
        s = list(map(str, sorted(data)))
        data = list(map(str, data))
        msdrs = MSDRadixSort(data, alphabet)
        self.assertEqual(msdrs, s)

    def testMSDRSort(self):
        for _ in range(10):
            alphabet = list('0123456789')
            data = random.sample(range(0, 10**6), 10)
            s = list(map(str, sorted(data)))
            data = list(map(str, data))
            msdrrs = MSDRadixRecursionSort(data, alphabet)
            # print('data', *data, sep='\n')
            # print('msdrs', *msdrs, sep='\n')
            # print('s', *s, sep='\n')
            self.assertEqual(msdrrs, s)
        alphabet = list('0123456789')
        data = []
        for i in range(1, 6):
            data.extend(random.sample(range(0, 10**i), 5))
        s = list(map(str, sorted(data)))
        data = list(map(str, data))
        msdrrs = MSDRadixRecursionSort(data, alphabet)
        self.assertEqual(msdrrs, s)

class TSTTest(unittest.TestCase):

    def testInit(self):
        T = TST()
        key = 'abc'
        self.assertEqual(T.get(key), None)
        self.assertFalse(T.contains(key))
        self.assertEqual(T.size, 0)

    def testInitNode(self):
        key = 'a'
        N = Node(key)
        self.assertEqual(N.key, key)
        T = TST(N, 1)
        self.assertEqual(T.get(key), None)
        self.assertTrue(T.contains(key))
        N.value = 29393
        self.assertEqual(T.get(key), 29393)
        self.assertFalse(T.contains('b'))
        self.assertEqual(T.get('b'), None)
        self.assertEqual(T.size, 1)

    def testInsert(self):
        T = TST()
        T.insert('a', 1)
        self.assertTrue(T.contains('a'))
        self.assertEqual(T.get('a'), 1)
        self.assertFalse(T.contains('b'))
        self.assertTrue(T.size, 1)

if __name__ == '__main__':
    unittest.main()
