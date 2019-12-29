
## SLLists ##

        
class SLList:
    
    class Node:
        
        def __init__(self, value=None, next=None):
            self.value = value
            self.next = next
        
        def __repr__(self):
            return self.value.__repr__() + ' -> ' + self.next.__repr__()
    
    def __init__(self):
        self.sentinel = SLList.Node('sentinel')
        self.size = 0
    
    def __len__(self):
        return self.size
    
    def __repr__(self):
        return self.sentinel.next.__repr__()
    
    def get(self, index): # get value of the indexed item
        node = self._getNode(index)
        return node.value
        
    def _getNode(self, index): # return ith Node
        self._getNodeCheckIndex(index)
        node = self.sentinel
        for _ in range(index + 1):
            node = node.next
        return node
    
    def _getNodeCheckIndex(self, index):
        if index < -1:
            raise IndexError('SLList Node index {} < -1'.format(index))
        if index >= len(self):
            raise IndexError('SLList Node index {} greater or equal to
                              length {}'.format(index, len(self)))
            
    def insert(self, index, value): # insert value at given index, shift rest right
        prev = self._getNode(index-1)
        prev.next = SLList.Node(value, prev.next)
        self.size += 1
    
    # remove the indexed item and return it's value
    def pop(self, index=None):
        if index is None:
            index = len(self)-1
        prev = self._getNode(index-1)
        curr = prev.next
        prev.next = curr.next
        curr.next = None
        self.size -= 1
        return curr.value
    
    # remove first item with given value
    # if item exists, return True, otherwise False
    def remove(self, value):
        prev = self.sentinel
        while prev.next:
            if prev.next.value == value:
                curr = prev.next
                prev.next = curr.next
                curr.next = None
                self.size -= 1
                return True
            prev = prev.next
        return False
        
    # O(1) insertion to beginning of list
    def push(self, value):
        self.insert(0, value)
    
    # O(n) insertion to end of list
    def append(self, value):
        self.insert(len(self), value)
    
    def extend(self, lst): # operation causes self and lst to share nodes, mutability if lst in non empty
        if len(lst) == 0:
            return
        prev = self._getNode(len(self)-1)
        prev.next = lst._getNode(0)
        self.size += len(lst)
    
    def count(self, value): # return the number of items with given value
        node = self.sentinel
        result = 0
        for _ in range(len(self)):
            node = node.next
            if node.value == value:
                result += 1
        return result
    
    def sort(self):
        pass
    
    def reverse(self): # iterative
        prev = None
        curr = self.sentinel.next
        while curr:
            newCurr = curr.next
            curr.next = prev
            prev, curr = curr, newCurr
        self.sentinel.next = prev
    
    def reverseRecursive(self):
        if self.sentinel.next: # if len(self) == 0
            self.sentinel.next = self._reverseNode(self.sentinel.next)
    
    def _reverseNode(self, prev):
        if not prev.next:
            return prev
        curr = prev.next
        newFirstNode = self._reverseNode(curr)
        curr.next = prev
        prev.next = None
        return newFirstNode
    
    def reverseRecursive2(self):
        firstNode = self.sentinel.next
        if firstNode:
            self.sentinel.next = self._reverseNode2(firstNode)
            firstNode.next = None
    
    def _reverseNode2(self, prev):
        if not prev.next:
            return prev
        curr = prev.next
        newFirstNode = self._reverseNode2(curr)
        curr.next = prev
        return newFirstNode
        
    def reverseRecursive3(self):
        firstNode = self.sentinel.next
        if firstNode:
            lastNode = []
            self._reverseNode3(firstNode, lastNode)
            self.sentinel.next = lastNode[0]
            firstNode.next = None

    def _reverseNode3(self, prev, lastNode):
        if not prev.next:
            lastNode.append(prev)
        else:
            curr = prev.next
            self._reverseNode3(curr, lastNode)
            curr.next = prev
            
    def getFirstMiddleNode(self): # O(n), O(1)
        middle = self.sentinel.next
        end = self.sentinel.next
        while True:
            if not end.next or not end.next.next:
                return middle
            middle = middle.next
            end = end.next.next
    
    def getSecondMiddleNode(self):
        middle = self.sentinel.next
        end = self.sentinel.next
        while True:
            if not end.next:
                return middle
            if not end.next.next:
                return middle.next
            middle = middle.next
            end = end.next.next
    # s  0  1  ...  -3  -2  -1
    # l                  r
    #                   u
    
    def sortInsertion(self):
        if len(self) <= 1:
            return
        right = self._getNode(len(self) - 2)
        left = self.sentinel
        while left.next is not right:
            runner = right
            while runner.next and left.next.value > runner.next.value:
                runner = runner.next
            SLList._insertPrevNode(left, runner)
        while right.next and left.next.value > right.next.value:
            right = right.next
        if left.next.value > right.value:
            SLList._insertPrevNodeGeneral(left, right)
        
    def _insertPrevNode(prev0, prev1): # prev0 before prev1
        node0 = prev0.next
        prev0.next = node0.next
        node0.next = prev1.next
        prev1.next = node0
    
    def _insertPrevNodeGeneral(prev0, prev1):
        if prev0.next is prev1:
            prev0.next = prev1.next
            prev1.next = prev1.next.next
            prev0.next.next = prev1
        else:
            SLList._insertPrevNode(prev0, prev1)

    def _swapNode(self, prev0, prev1):
        if prev0.next is prev1:
            prev0.next = prev1.next
            prev1.next = prev1.next.next
            prev0.next.next = prev1
        elif prev1.next is prev0:
            prev1.next = prev0.next
            prev0.next = prev0.next.next
            prev1.next.next = prev0
        else:
            temp = prev0.next
            prev0.next = prev1.next
            prev1.next = temp
            temp = prev1.next.next
            prev0.next.next = prev1.next.next
            prev1.next.next = temp

            
def altSplit(lst):
    odd, even = SLList(), SLList()
    node = lst.sentinel.next
    prev, curr = odd.sentinel, even.sentinel
    for _ in range(len(lst)):
        curr.next = Node(node.value)
        prev, curr = curr.next, prev
        node = node.next
    return odd, even
    
def SLLZip(lst0, lst1):
    result = SLList()
    node = result.sentinel
    prev, curr = lst1.sentinel.next, lst0.sentinel.next
    while curr:
        node.next = SLList.Node(curr.value)
        prev, curr = curr.next, prev
        node = node.next
    if not curr:
        curr = prev
    while curr:
        node.next = SLList.Node(curr.value)
        node, curr = node.next, curr.next
    return result

def mergeSort(lst):
    if len(lst) == 0:
        return SLList()
    return _mergeSortHelper(lst, lst.sentinel.next)

def _mergeSortHelper(lst, start, stop=None):
    if start.next is stop:
        result = SLList()
        result.push(start.value)
        return result
    mid = _mergeSortGetMiddleNode(start, stop)
    leftLst = _mergeSortHelper(lst, start, mid)
    rightLst = _mergeSortHelper(lst, mid, stop)
    return _mergeSortCombine(leftLst, rightLst)

def _mergeSortCombine(lst0, lst1): # left and right nonempty
    result = SLList()
    node = result.sentinel
    node0, node1 = lst0._getNode(0), lst1._getNode(0)
    while node0 and node1:
        if node0.value < node1.value:
            node.next = SLList.Node(node0.value)
            node0 = node0.next
        else:
            node.next = SLList.Node(node1.value)
            node1 = node1.next
        node = node.next
        result.size += 1
    curr: SLList.Node
    if node0:
        curr = node0
    else:
        curr = node1
    while curr:
        node.next = SLList.Node(curr.value)
        node, curr = node.next, curr.next
        result.size += 1
    return result


def _mergeSortGetMiddleNode(start, stop=None): # start.next is not stop
    middle = start.next
    end = start.next.next
    while True:
        if end is stop or end.next is stop:
            return middle
        middle = middle.next
        end = end.next.next
    

def getIntersectionNode(headA, headB):
    """
    Find the difference of the lengths of the two lists.
    Start pointers in both lists, the longer list starts at the difference.
    Walk each pointer until they meet or not.
    """
    a, b = length(headA), length(headB)
    d = a - b
    pA, pB = headA, headB
    if d > 0:
        pA = advance(pA, d)
    elif d < 0:
        pB = advance(pB, -d)
    while pA:
        if pA is pB:
            return pA
        pA, pB = pA.next, pB.next
    return None
            
    def length(self, head):
        count = 0
        node = head
        while node:
            count += 1
            node = node.next
        return count
    
    def advance(self, node, n): # n < len(list)
        for _ in range(n):
            node = node.next
        return node
    

## DLList ##

class DLList:
    
    class Node:
        
        def __init__(self, value=None, prev=None, next=None):
            self.value = value
            self.prev = prev
            self.next = next
            
        def __repr__(self):
            node = self
            result = ''
            while node:
                result += node.value.__repr__() + ' <-> '
                node = node.next
            if result:
                result = result[0 : -5]
            return result
    
    def __init__(self, itr=None):
        self._size = 0
        self._sentinel = DLList.Node('sentinel')
        self._sentinel.prev = self._sentinel
        self._sentinel.next = self._sentinel
        if itr:
            self._initItr(itr)
    
    def __len__(self):
        return self._size
    
    def __repr__(self):
        return self._sentinel.next.__repr__()
    
    def _getNode(self, index):
        """Return the Node at index or sentinel if index is length."""
        self._getNodeCheckIndex(index)
        node = self._sentinel
        # Optimize retrieval using next or prev from sentinel.
        direction: str
        if index < 0:
            if index < -len(self) // 2:
                index = len(self) + index
                direction = 'next'
            else:
                direction = 'prev'
        else:
            if index > len(self) // 2:
                index = len(self) - index
                direction = 'prev'
            else:
                direction = 'next'
        for _ in range(index):
            node = getattr(node, direction)
        return node
    
    def _getNodeCheckIndex(self, index):
        if index > len(self):
            raise IndexError('DLList Node index {} > list length {}.'
                             .format(index, len(self)))
        if index < -len(self):
            raise IndexError('DLList Node index {} < neagtive list length {}.'
                             .format(index, len(self)))
    
    def _getItem(self, index):
        if index == len(self):
            raise IndexError('DLList index {} == list length {}.'
                             .format(index, len(self)))
        return self._getNode(index).value
    
    def _getSlice(self, slc):
        node = _getNode(self, slc.start)
        # lst = DLList()
        # step = None
        # if slc.stop
        direction: str
        pass
    
    
    def _copyNode(self, node, prev, next):
        return DLList.Node(node.value, prev, next)
    
    def __getitem__(self, key):
        if isinstance(key, int):
            self._getItem(key)
        elif isinstance(key, slice):
            self._getSlice(key)
        else:
            raise TypeError('Index must be int, not {}'.format(type(key).__name__))

    def insert(self, index, value):
        """Insert value and shift items right.  Allow indices out of range."""
        if index < -len(self):
            index = -len(self)
        elif index > len(self):
            index = len(self)
        if index == len(self):
            node = self._sentinel
        else:
            node = self._getNode(index)
        newNode = DLList.Node(value, node.prev, node)
        node.prev.next, node.prev = newNode, newNode
    
    def pop(self, index=-1):
        if index == len(self):
            raise IndexError('DLList index {} == list length {}.'
                             .format(index, len(self)))
        node = self._getNode(index).value
        returnVal = node.value
        self._removeNode(node)
    
    def _removeNode(self, node):
        if node is self._sentinel:
            raise Error('Cannot remove sentinel node.')
        node.prev.next = node
        node.next.prev = node
        

## Disjoint Sets ##

class QuickUnionDisjointSet:
    
    def __init__(self, size):
        self._parent = list(range(size))
    
    def __len__(self):
        return self._parent.__len__()
    
    def isConnected(self, i, j): # O(n)
        """Return boolean of whether i and j are in the same set."""
        pi, pj = self._find(i), self._find(j)
        return pi == pj
    
    def _find(self, i):
        """Return deepest parent of i, whose parent is itself."""
        pi = self._parent[i]
        while pi != self._parent[pi]:
            pi = self._parent[pi]
        return pi
    
    def connect(self, i, j): # O(n)
        """
        Union the sets containing i and j.
        One parent becomes that of other.
        """
        pi, pj = self._find(i), self._find(j)
        self._parent[pj] = pi



class Queue:

    def __init__(self):
        self._arr = []
        self._last = 0

    def __len__(self):
        return len(self._arr) - self._last

    def __iter__(self):
        return reversed(self._arr[self._last : ])

    def __repr__(self):
        result = '['
        for i in range(-1, -len(self)-1, -1):
            result += '{}, '.format(self._arr[i])
        return result.strip(', ') + ']'

    def push(self, item):
        self._arr.append(item)

    def pushItr(self, itr):
        for item in itr:
            self.push(item)

    def pop(self):
        if not len(self):
            raise IndexError('Cannot pop from empty list')
        returnItem = self._arr[self._last]
        self._arr[self._last] = None
        self._last = self._last + 1
        self._checkResize()
        return returnItem

    def _checkResize(self):
        if self._last == len(self._arr):
            self._arr = []
            self._last = 0
        elif len(self._arr) >=16 and len(self) / len(self._arr) < .25:
            self._arr = self._arr[self._last: self._last + len(self)]
            self._last = 0


class Deque:

    def __init__(self):
        self._arr = []
        self._first = 0
        self._last = 0

    def __len__(self):
        return len(self._arr) - self._first

    def __iter__(self):
        return iter(self._arr[self._first : ])

    def __repr__(self):
        return self._arr[self._first : ].__repr__()

    def push(self, item):
        self._arr.append(item)

    def pushLeft(self, item):
        self._arr

    def pushItr(self, itr):
        for item in itr:
            self.push(item)

    def pop(self):
        if not len(self):
            raise IndexError('Cannot pop from empty list')
        returnItem = self._arr[self._first]
        self._arr[self._first] = None
        self._first = self._first + 1
        self._checkResize()
        return returnItem

    def _checkResize(self):
        if self._first == len(self._arr):
            self._arr = []
            self._last = 0
        elif len(self._arr) >=16 and len(self) / len(self._arr) < .25:
            self._arr = self._arr[self._first: self._first + len(self)]
            self._first = 0
    

class WeightedQUDS:
    
    def __init__(self, length):
        self._parent = list(range(length))
        self._size = [1 for _ in range(length)]
    
    def __len__(self):
        return self._parent.__len__()
    
    def isConnected(self, i, j): # O(log n)
        """Return boolean of whether i and j are in the same set."""
        pi, pj = self._find(i), self._find(j)
        return pi == pj
    
    def _find(self, i):
        """Return deepest parent of i, whose parent is itself."""
        pi = self._parent[i]
        while pi != self._parent[pi]:
            pi = self._parent[pi]
        return pi
    
    def connect(self, i, j): # O(log n)
        """
        Union the sets containing i and j.
        Connect the item of smaller set directly to parent of larger set.
        """
        pi, pj = self._find(i), self._find(j)
        if pi == pj:
            return
        wi, wj = self._size(pi), self._size(pj)
        if wj > wi:
            self._parent[i] = pj
            self._size[pj] += self._size[pi]
        else:
            self._parent[j] = pi
            self._size[pi] += self._size[pj]

class WQUDSWithPathCompression:

    def __init__(self, length=0):
        self._parent = list(range(length))
        self._size = [1 for _ in range(length)]

    def __len__(self):
        return self._parent.__len__()

    def isConnected(self, i, j): # O(log n)
        """
        Return boolean of whether i and j are in the same set.
        Implicitly update parents using _find method.
        """
        pi, pj = self._find(i), self._find(j)
        return pi == pj

    def _find(self, i):
        """
        Return deepest parent of i, whose parent is itself.
        Use recursion to flatten branch.
        """
        if i == self._parent[i]:
            return i
        self._parent[i] = self._find(self._parent[i])
        return self._parent[i]

    def connect(self, i, j): # O(log n)
        """
        Union the sets containing i and j.
        Connect the item of smaller set directly to parent of larger set.
        Implicitly update parents using _find method.
        """
        pi, pj = self._find(i), self._find(j)
        if pi == pj:
            return
        wi, wj = self._size[pi], self._size[pj]
        if wj > wi:
            self._parent[pi] = pj
            self._size[pj] += self._size[pi]
        else:
            self._parent[pj] = pi
            self._size[pi] += self._size[pj]

    def largestConnectedComponentSize(self): # O(n)
        """Return the size of the largest set."""
        return max(self._size)
    
    def getRepresentative(self, i):
        return self._find(i)
    
    def getSize(self, i):
        return self._size[self.getRepresentative(i)]
    
    def listRepresentives(self):
        result = []
        for p in self._parent:
            if p == self._parent[p]:
                result.append(p)
        return result
    
    def numComponents(self):
        count = 0
        for i in range(len(self)):
            if i == self._parent[i]:
                count += 1
        return count

    def addItem(self):
        """Add a singleton set to the collection."""
        self._parent.append(len(self))
        self._size.append(1)

def HoshenKopelman(grid):
    DS = WQUDSWithPathCompression()
    label = [[None for _ in range(grid.numCols)]
             for _ in range(grid.numRows)]
    for i in range(grid.numRows):
        for j in range(grid.numCols):
            if grid.occupied(i, j):
                left, above = False, False
                if i:
                    left = grid.occupied(i-1, j)
                if j:
                    above = grid.occupied(i, j-1)
                if not left and not above:
                    DS.addItem()
                    label[i][j] = len(DS) - 1
                elif left and not above:
                    if label[i-1][j] is None:
                        assert False
                    label[i][j] = DS._find(label[i-1][j])
                elif not left and above:
                    if label[i][j-1] is None:
                        assert False
                    label[i][j] = DS._find(label[i][j-1])
                else:
                    if label[i][j-1] is None or label[i-1][j] is None:
                        assert False
                    DS.connect(label[i-1][j], label[i][j-1])
                    label[i][j] = DS._find(label[i-1][j])
    for i in range(grid.numRows):
        for j in range(grid.numCols):
            if label[i][j] is not None:
                label[i][j] = DS._find(label[i][j])
    return label

class Grid:

    def __init__(self, numRows, numCols):
        self._arr = np.zeros((numRows, numCols), int)
        self.numRows = numRows
        self.numCols = numCols

    def occupy(self, i, j):
        self._arr[i, j] = 1

    def occupied(self, i, j):
        return bool(self._arr[i, j])

    def __repr__(self):
        return self._arr.__repr__()
            
                    

## Dynamic Programming ##

# Longest Increasing Subsequence
def lengthoflongestincreasingsubsequence(lst):
    # L[i] is length of longest increasing subsequence with last item lst[i]
    if not lst:
        return 0
    maxL = 1
    L = [1]
    for j in range(1, len(lst)):
        Lj = 1+max(L[j] for j in range(i) if lst[i] <= lst[j])
        L.append(Lj)
        maxL = max(maxL, Lj)
    return maxL
        
def longestincreasingsubsequence(lst):
    if not lst:
        return []
    maxlength = 1
    edgeto = [None]
    maxend = 0
    L = [1]
    for j in range(1, len(lst)):
        maxi = max([i for i in range(j) if lst[i] <= lst[j]],
                   key=lambda i: L[i],
                   default=-1)
        if maxi == -1:
            edgeto.append(None)
            L.append(1)
        else:
            edgeto.append(maxi)
            jlength = 1 + L[maxi]
            L.append(jlength)
            if jlength > maxlength:
                maxlength = jlength
                maxend = j
    maxsubseq = []
    while maxend is not None:
        maxsubseq.append(lst[maxend])
        maxend = edgeto[maxend]
    maxsubseq.reverse()
    return maxsubseq
    
    
def LIS(S):
    LISEAPI = [None] #LISEA Previous Index
    LLISEA = [1] # Length of LIS Ending At
    for i in range(len(S)):
        # all increasing subsequences ending at i
        Indices = [j for j in range(i) if S[j] <= S[i]]
        if not Indices:
            LISEAPI.append(None)
            LLISEA.append(1)
        else:
            maxInd = max(Indices, key=lambda j: LLISEA[j])
            maxLen = 1 + LLISEA[maxInd]
            LISEAPI.append(maxInd)
            LLISEA.append(maxLen)
    maxInd = max(range(len(S)), key=lambda i: LLISEA[i])
    maxLen = LLISEA[i]
    seq = []
    i = maxInd
    for _ in range(maxLen):
        seq.append(i)
        i = LISEAPI[i]
    seq.reverse()
    return maxLen, seq
    
# Trapping water

def trap(H): # Heights array, water trapped between barriers on left and right
    if not H:
        return 0
    LMHEA = [H[0]] # Left Max H Ending At
    for i in range(1, len(H)):
        LMHEA.append(max(H[i], LMHEA[i-1]))
    RMHEA = [None for _ in range(len(H) - 1)] + [H[-1]]
    for i in range(len(H)-2, -1, -1):
        RMHEA[i] = max(H[i], RMHEA[i+1])
    result = 0
    for i in range(len(H)):
        waterLevel = min(LMHEA[i], RMHEA[i])
        result += max(0, waterLevel - H[i])
    return result
    
    
# Number of ways to make change for n given sorted values V
def makeChange(V, n):
    if n < 0:
        return 0
    if n == 0:
        return 1
    if V == []:
        return 0
    withMax = makeChange(V, n - V[-1])
    withoutMax = makeChange(V[:-1], n)
    return withMax + withoutMax

# Min number of coins to make change
def minChange(V, n):
    # Degenerative case: no coins but n is positive
    if not V and n > 0:
        return None
    # Base case: no coins is 0, min coin > n is 0
    if not V or V[0] > n:
        return 0
    # Base case: i < V[0] is 0, i == V[0] is 1
    minC = 1
    C = [0] * V[0] + [1]
    # Inductive step:
    for i in range(V[0]+1, n+1):
        mini = 1 + min(C[i-V[j]] for j in range(i) if V[j] <= i,
                       default=-1)
        if not mini:
            C.append(0)
        else:
            C.append(mini)
            minC = min(minC, mini)
    return minC


def editdistance(word1, word2):
    # SUNN_Y   S to S is pass
    # S_NOWY   N to O is sub
    #          U to _ is del
    #          _ to W is ins
    # E[i][j] = number of edits from word1[:i] to word2[:j]
    # find E[m][n]
    m, n = len(word1), len(word2)
    E = [[None for _ in range(n+1)] for _ in range(m+1)]
    # Base case
    for i in range(1, m+1): # 'abc' to '' is 3
        E[i][0] = i
    for j in range(n): # '' to 'abc' is 3
        E[0][j] = j
    # Inductive: +0 for pass, +1 for sub, del, ins
    for i in range(1, m+1):
        for j in range(1, n+1):
            # 'a' to 'a' is pass, d=0
            # 'a' to 'b' is sub, d=1
            d = int(word1[i] != word2[j])
            # sub/pass, del, ins
            E[i][j] = min(d + E[i-1][j-1], 1 + E[i-1][j], 1 + E[i][j-1])
    return E[m][n]
                
def knapsackwithrepetition(values, weights, W):
    # K[w] is the result for weight w
    K = [None for _ in range(W+1)]
    K[0] = 0
    for w in range(1, W+1):
        K[w] = max((values[i] + K(w-weights[i])
                    for i in range(len(values)) if weights[i] <= w),
                   default=0)
    return K[W]
    # def helper(w):
    #     if K[w] is None:
    #         result = max((values[i] + helper(w-weights[i])
    #                       for i in range(len(values)) if weights[i] <= w)),
    #                      default=0)
    #         K[w] = result
    #     return K[w]
    # return helper(W)

def knapsackwithoutrepetition(values, weights, W):
    # K[i][w] = result for w restricted to th ith item
    K = [[0]+[None for _ in range(W)] for _ in range(len(values))]
    for i in range(len(values)):
        for w in range(1, W+1):
            if weights[i] > w:
                K[i][w] = K[i-1][w]
            else:
                K[i][w] = max(K[i-1][w],
                              values[i] + K[i-1][w-weights[i]])
    retur K[len(values)][W]
    # def helper(i, w):
    #     if K[i][w] is None:
    #         if weights[i] > w:
    #             result = helper(i-1, w)
    #         else:
    #             result = max(helper(i-1, w),
    #                          values[i] + helper(i-1, w-weights[i]))
    #         K[i][w] = result
    #     return K[i][w]
    # return helper(len(values), W)
    
from string import ascii_lowercase
def longestcommonsubsequence(s, t):
    # L[i][j] = longestcommonsubsequence(s[:i+1], t[:j+1])
    m, n = len(s), len(t)
    L = [[None for _ in range(n)] for _ in range(m)]
    # Base case: 'a' and 'bbabbbbb' is 0,0,1,...,1
    for j in range(0, n):
        if s[0] == t[j]:
            for k in range(j, n):
                L[0][k] = 1
            break
        else:
            L[0][j] = 0
    for i in range(0, m):
        if s[i] == t[0]:
            for k in range(i, m):
                L[k][0] = 1
            break
        else:
            L[i][0] = 0
    # Inductive: if same last char, 1+diagonal; else max(left, right)
    for i in range(1, m):
        for j in range(1, n):
            if s[i] == t[j]:
                L[i][j] = 1 + L[i-1][j-1]
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    return L[m][n]
    
def hassubsetsum(lst, total):
    lst.sort()
    # H[i][j] = lst[:i+1] has sum j
    # Find H[len(lst)-1][total]
    H = [[False for _ in range(total+1)] for _ in range(len(lst))]
    # Base case: sum([]) = 0
    for i in range(len(lst)):
        H[i][0] = True
    if not total:
        return True
    # Base case: empty lst only sums to 0, lst[0] > total is False
    if not lst or lst[0] > total:
        return False
    # Base case: H[i][j] is False for all j < lst[0], True for j == lst[0]
    for i in range(m):
        for j in range(1, lst[0]):
            H[i][j] = False
        H[i][lst[0]] = True
    # Base case: H[0][j] is False for all j > lst[0]
    for j in range(lst[0]+1, total+1):
        H[0][j] = False
    # row 0:        T F ... F T F ... F
    # row 1 to m-1: T F ... F T ? ... ?
    # Inductive step: check down each col, consider lst[i] in sum
    for j in range(lst[0]+1, total+1):
        for i in range(1, len(lst)): # H[i-1][j] == False
            if lst[i] <= j:
                H[i][j] = H[i-1][j-lst[i]]
            else:
                H[i][j] = False
                for k in range(i+1, len(lst)):
                    H[k][j] = False
                break
            if H[i][j]:
                for k in range(i+1, len(lst)):
                    H[k][k] = True
                break
    return H[m-1][total]
    
[0, 1, 2, 3, 1, 2, 3, 4, 2, 1, 2, 3, 3]
 0  1  2  3  4  5  6  7  8  9  10 11 12
[0, 1, 2, 3, 1, 2, 3, 4, 2, 1, 2, 3, 3]

## HashSets ##

class HashSet:
    
    min_load_factor = .25
    max_load_factor = .75
    default_length = 16
    
    def __init__(self, length=None):
        self._numBuckets: length
        if not length:
            self._numBuckets = HashSet.default_length
        else:
            self._numBuckets = length
        self._buckets = [[] for _ in range(self._numBuckets)]
        self._size = 0
    
    def __len__(self):
        return self._size
    
    def _hashCode(item):
        return item.__hash__()
    
    def _key(self, item):
        return _hashCode(item) % self._numBuckets
    
    def put(item):
        key = self._key(item)
        self._buckets[key].append(item)
        self._size += 1
        self._checkResize()
    
    def remove(item):
        key = self._key(item)
        try:
            self._buckets[key].remove(item)
        except ValueError:
            print('HashSet.remove(item): item is not a member.')
        self._size -= 1
        self._checkResize()
    
    def contains(self, item):
        key = self._key(item)
        return item in self._buckets[key]
    
    def items(self):
        result = []
        b = self._numBuckets
        while len(self) > len(result) and b > 0:
            result.extend(self._buckets[b])
            b -= 1
        return result
        
    def _loadFactor(self):
        return len(self) / self._numBuckets
    
    def _checkResize(self):
        if self._loadFactor() > HashSet.max_load_factor:
            self._resize(self._numBuckets * 2)
        elif (self._loadFactor() < HashSet.min_load_factor and
              self._numBuckets > 2 * HashSet.default_length):
            self._resize(self._numBuckets // 2))
    
    def _reize(self, size):
        self._numBuckets = size
        newBuckets = [[] for _ in range(self._numBuckets)]
        for bucket in self._buckets:
            for item in bucket:
                key = self._key(item)
                newBuckets[key].append(item)
        self._buckets = newBuckets
    
    def __iter__(self):
        
        class itemIterator:
            
            def __init__(self, buckets, size):
                self.buckets = buckets
                self.size = size
                self.b = 0
                self.i = 0
            
            def __iter__(self):
                return self
            
            def __next__(self):
                if self.size = 0:
                    raise StopIteration
                else:
                    returnItem = self.buckets[b][i]
                    i = (i+1) % len(self.buckets[b])
                    if not i:
                        b += 1
                    self.size -= 1
                    return returnItem
        
        return itemIterator(self._buckets, len(self))
        
            

# Note hashCode functions:
#   n = number of items
#   b = number of buckets
#   b prime >= 2n
#   N = number of possible items
#   i.e. N is size of alphabet * length str, items are str of letters
#   k s.t. N >= b**k
#   item = (i0, ..., ik)
#   a = (a0, ..., ak)
#   _key(item) = a * i mod b


## Priority Queues ##

class MinHeap:
    
    def __init__(self):
        self._arr = [None] * 8
        self._size = 0
        if itr:
            self._initItr()
    
    def __len__(self):
        return self._size
        
    def __iter__(self):
        return iter(self._arr[1:len(self)+1])
    
    # [None, 1, 2, 3] len == 3, lastItemIndex == len
    
    def add(self, i):
        self._arr[len(self)+1] = i
        self._swim(len(self)+1)
        self._size += 1
        self._checkResize()
    
    def getMin(self):
        return self._arr[1]
    
    def pop(self, i=0):
        if i >= len(self):
            raise IndexError('pop index {} > length {}'.format(i, len(self)))
        if i < 0:
            raise IndexError('pop index {} < 0'.format(i))
        returnVal = self._arr[i+1]
        self._arr[i+1], self._arr[len(self)] = self._arr[len(self)], None
        self._arr[len(self)] = None
        self._size -= 1
        self._sink(i+1)
        self._swim(i+1)
        self._checkResize()
    
    def _minChild(self, index):
        child1 = 2 * index
        if child1 > len(self):
            return None
        if child1 == len(self):
            return child1
        return min([child1, child1+1], key=lambda c: self._arr[c])
    
    def _parent(self, index):
        return index // 2
    
    def _sink(self, index):
        child = self._minChild(index)
        if child and self._arr[index] > self._arr[child]:
            self._swap(index, child)
            self._sink(child)
    
    def _swim(self, index):
        par = self._parent(index)
        if par and self._arr[par] > self._arr[index]:
            self._swap(par, index)
            self._swim(par)
    
    def _swap(self, index1, index2):
        self._arr[index1], self._arr[index2] = self._arr[index2], self._arr[index1]

    def _checkResize(self):
        if len(self) == len(self._arr) - 1:
            self._resize(len(self._arr) * 2)
        elif len(self._arr) > 31 and len(self)/len(self._arr) < .25:
            self._resize(len(self._arr) // 2)
    
    def _resize(self, size):
        # 15, 16 to 32 -> [0:16] + [None]*16
        #  7, 32 to 16 -> [0: 8] + [None]*8
        self._arr = self._arr[0:len(self)+1] + [None] * (size-(len(self)+1))



class MinMaxHeap:
    
    pass



## Sorting ##

from math import log

def minRange(lst, i, j):
    return min(lst[i: j])

def swap(lst, i, j):
    lst[i], lst[j] = lst[j], lst[i]



def selectionSort(lst):
    result = []
    if not lst:
        return result
    for i in range(len(lst)):
        result.append(min(lst[i:]))
    return result

def selectionSortInPlace(lst): # Theta(n**2), O(1)
    for i in range(len(lst)-1):
        m = min(range(i, len(lst)), key=lambda x: lst[x])
        swap(lst, i, m)



def insertionSortInPlace(lst): # Omega(n) to O(n**2), O(1)
    # good for almost sorted or small n
    # lot of swaps
    for i in range(1, len(lst)):
        for j in range(i, 0, -1):
            if lst[j] > lst[j-1]:
                swap(lst, j, j-1)
            else:
                break

def _stride(lst, k): # k=1 is insertionSortInPlace
    for i in range(k, len(lst)):
        for j in range(i, 0, -k):
            if lst[j] > lst[j-k]:
                swap(lst, j, j-k)
            else:
                break

def _strideLengths(n): # n == len(lst) > max stride length
    result = []
    curr = 2
    while curr <= n:
        result.append(curr-1)
        curr *= 2
    return result

def ShellsSortInPlace(lst):
    lengths = _strideLengths(len(lst))
    for i in range(len(lengths)-1, -1, -1):
        _stride(lst, lenghts[i])
    


def heapSortInPlace(lst): # Theta(nlogn), O(1)
    heapify(lst)
    for i in range(len(lst)-1):
        swap(lst, i, len(lst)-i-1)
        _sink(lst, 0, len(lst)-i-1)

def heapify(lst): # max heap, bottom up, Theta(n)
    for i in range(len(lst)//2 -1, -1, -1):
        _sink(lst, i, len(lst))

def _sink(lst, i, length): # lst[length-1] is last item in heap
    child = _maxChild(lst, i, length)
    if child and lst[child] > lst[i]:
        swap(lst, i, child)
        _sink(lst, child, length)

def _maxChild(lst, i, length):
    child1 = 2*i + 1
    if child1 > length - 1:
        return None
    if child1 == length - 1:
        return child1
    return max([child1, child1+1], key=lambda x: lst[x])



def mergeSort(lst): # Theta(nlogn), Theta(n)
    if len(lst) < 2
        return lst
    left = mergeSort(lst[0: len(lst)//2])
    right = mergeSort(lst[len(lst)//2 :])
    return _merge(left, right)

def _merge(left, right): # len(right) == or -=1 len(left), >= 1
    result = []
    i, j = 0, 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    if i == len(left):
        result.extend(right[j : ])
    else:
        result.extend(left[i : ])
    return result

def _mergeAdaptive(left, right): # if _merge(left, right) = left + right
    if left[-1] <= right[0]:
        return left + right
    return _merge(left, right)
    
from collections import deque

def bottomUpMergeSortPerm(lst): # return permuation list
    queue = deque(map(lambda x: [x], range(len(lst))))
    while len(queue) > 1:
        sub1, sub2 = queue.popleft(), queue.popleft()
        queue.append(_mergeIndices(lst, sub1, sub2))
    return queue[0]

def _mergeIndices(lst, sub1, sub2): # len(sub) > 0
    i, j = 0, 0
    result = []
    while i < len(sub1) and j < len(sub2):
        if lst[sub1[i]] < lst[sub2[j]]:
            result.append(sub1[i])
            i += 1
        else:
            result.append(sub2[j])
            j += 1
    if i == len(sub1):
        result.extend(sub2[j : ])
    else:
        result.extend(sub1[i : ])
    print(lst, sub1, sub2, result, '', sep='\n')
    return result


def sortedMergeInto(a, b):
    """Merge lists b into a using binary search and insertion. O(1) memory."""
    for x in b:
        i = _sortedInsInd(a, 0, len(a), x)
        a.insert(i, x)

def _sortedInsInd(a, start, stop, x):
    if start == stop:
        return start
    m = (start + stop) // 2
    if a[m] < x:
        return _sortedInsInd(a, m+1, stop, x)
    if a[m] == x:
        return m
    else:
        return _sortedInsInd(a, start, m, x)
        

def binarySortedSearch(lst, item):
    return _bsRange(lst, 0, len(lst), item)

def _bssRange(lst, start, stop, item):
    if start == stop:
        return None
    m = (start + stop) // 2
    if item == lst[m]:
        return m
    left = _bssRange(lst, start, m, item)
    right = _bssRange(lst, m+1, stop, item)
    if left != None:
        return left
    if right != None:
        return right
    return None
    

def getMajority(lst):
    if len(lst) < 2:
        if not len(lst):
            return NOne
        return lst[0]
    left, right = lst[0 : len(lst)//2], lst[len(lst)//2 : ]
    lMaj, rMaj = getMajority(left), getMajority(right)
    if left == right:
        return left
    if lMaj + right.count(lMaj) > n/2:
        return lMaj
    if rMaj + left.count(rMaj) > n/2:
        return rMaj
    return None

def medianTwoSorted(lst0, lst1):
    # [0   1  2]             3 4 5 | 3
    #  p0  p1 p2
    #  0  [1  2  3  4]  5    6 4 5 | 5
    #  p3  q3 q2 q1 q0
    #  0 ... s1 ... e2/s1 ... e1 ... -1
    lst0, lst1 = sorted([lst0, lst1], key=len)
    total = len(lst0) + len(lst1)
    m0 = (len(lst0) + len(lst1)) // 2
    m1 = m0
    if not total % 2:
        m1 +=1
    lst = lst0 + lst1[:m1+1]
    e0 = len(lst0)
    s0 = 0
    e1 = m1+1
    s1 = m0 - len(lst0)
    
    
    a, b = lst0, lst1
    p, q = len(lst0) // 2, len(lst1)//2
    if a[p] == b[q]:
        med0 = a[p]
        if not len(lst0) % 2 and not len(lst1) % 2:
            med1 = min(a[p+1], b[q+1])
            return (med0 + med1) / 2
        return med0
    
    
    
def kNearHeapSort(lst, k): # Theat(nlogk)
    _heapifyRange(lst, 0, k)
    

def heapSortInPlace(lst): # Theta(nlogn), O(1)
    heapify(lst)
    for i in range(len(lst)-1):
        swap(lst, i, len(lst)-i-1)
        _sink(lst, 0, len(lst)-i-1)

def heapify(lst): # max heap, bottom up, Theta(n)
    for i in range(len(lst)//2 -1, -1, -1):
        _sink(lst, i, len(lst))

def _sink(lst, i, length): # lst[length-1] is last item in heap
    child = _maxChild(lst, i, length)
    if child and lst[child] > lst[i]:
        swap(lst, i, child)
        _sink(lst, child, length)

def _maxChild(lst, i, length):
    child1 = 2*i + 1
    if child1 > length - 1:
        return None
    if child1 == length - 1:
        return child1
    return max([child1, child1+1], key=lambda x: lst[x])
    
    
    
import random

def quickSort(lst):
    _quickSortRange(lst, 0, len(lst))

def _quickSortRange(lst, start, stop):
    if stop - start < 2:
        return
    swap(lst, start, randint(start, stop - 1))
    pivot = lst[start]
    left, right = start + 1, stop - 1
    while True:
        while left < stop and lst[left] <= pivot:
            left += 1
        while lst[right] > pivot:
            right -= 1
        if right > left:
            swap(lst, left, right)
            left += 1
            right -= 1
        else:
            break
    swap(lst, start, right)
    _quickSortRange(lst, start, right)
    _quickSortRange(lst, right+1, stop)

def _quickSortThreeRange(lst, start, stop):
    if stop - start < 2:
        return
    swap(lst, start, random.randint(start, stop - 1))
    pivot = lst[start]
    pstart, pstop = start, start + 1
    prstart = stop
    left, right = start + 1, stop - 1
    while True:
        while True:
            while left < stop and lst[left] < pivot:
                left += 1
            if left < stop and lst[left] == pivot:
                swap(lst, pstop, left)
                pstop += 1
                left += 1
            else:
                break
        while True:
            while right >= pstop and lst[right] > pivot:
                right -= 1
            if right >= pstop and lst[right] == pivot:
                prstart -= 1
                swap(lst, prstart, right)
                right -= 1
            else:
                break
        if right > left:
            swap(lst, left, right)
            left += 1
            right -= 1
        else:
            break
    plen = pstop - pstart
    for i in range(plen):
        swap(lst, pstop-1 - i, right - i)
    prlen = stop - prstart
    for i in range(prlen):
        swap(lst, right+1 + i, prstart + i)
    _quickSortThreeRange(lst, start, right - plen + 1)
    _quickSortThreeRange(lst, right+1 + prlen, stop)

def _quickSortThreeRange(lst, start, stop):
    if stop - start < 2:
        return
    swap(lst, start, random.randint(start, stop - 1))
    pivot = lst[start]
    pstart, pstop = start, start + 1
    gstart = stop
    while pstop < gstart:
        if lst[pstop] <= pivot:
            if lst[pstop] < pivot:
                swap(lst, pstart, pstop)
                pstart += 1
            pstop += 1
        else:
            gstart -= 1
            swap(lst, pstop, gstart)
    _quickSortThreeRange(lst, start, pstart)
    _quickSortThreeRange(lst, pstop, stop)

def _quickSortTwoPivotsRange(lst, start, stop):
    pass
    
    0   1   2   3   4   5   6   7
    p   <=  >           <=  >   >
            l           r

def quickSelect(lst):
    if len(lst) < 2:
        if not len(lst):
            return None
        return lst[0]
    return _quickSelectRange(lst, 0, len(lst))

def _quickSelectRange(lst, start, stop): # m in [start: stop]
    swap(lst, random.randint(start, stop-1), start)
    pivot = lst[start]
    pstart, pstop = start, start + 1
    gstart = stop
    while pstop < gstart:
        if lst[pstop] <= pivot:
            if lst[pstop] < pivot:
                swap(lst, pstart, pstop)
                pstart += 1
            pstop += 1
        else:
            gstart -= 1
            swap(lst, pstop, gstart)
    # start ... pstart ... pstop ... stop-1
    m = len(lst) // 2
    if m < pstart:
        if not len(lst) % 2 and m+1 == pstart:
            return (max(lst[start: pstart]) + pivot)/2
        else:
            return _quickSelectRange(lst, start, pstart)
    if m < pstop:
        if len(lst) % 2:
            return pivot
        elif m+1 < pstop:
            return pivot
        else: # even length and m+1 == pstop
            return (pivot + min(lst[pstop: stop])) / 2
    else:
        return _quickSelectRange(lst, pstop, stop)


def quickSelectK(lst, k):
    if k >= len(lst) or k < 0:
        return None
    if len(lst) < 2:
        return lst[0]
    return _quickSelectKRange(lst, 0, len(lst), k)

def _quickSelectKRange(lst, start, stop, k): # k in [start: stop]
    swap(lst, random.randint(start, stop-1), start)
    pivot = lst[start]
    pstart, pstop = start, start + 1
    gstart = stop
    while pstop < gstart:
        if lst[pstop] <= pivot:
            if lst[pstop] < pivot:
                swap(lst, pstart, pstop)
                pstart += 1
            pstop += 1
        else:
            gstart -= 1
            swap(lst, pstop, gstart)
    # start ... pstart ... pstop ... stop-1
    if k < pstart:
        return _quickSelectKRange(lst, start, pstart, k)
    if k < pstop:
        return pivot
    else:
        return _quickSelectKRange(lst, pstop, stop, k)

def quick_select_floor(lst, x):
    """Find the greatest item in lst less than or equal to x. Return its index.
       If no item exists, return -1. i.e. x < lst[0] xor lst[i] <= x < lst[i+1] xor lst[-1] <= x."""
    start, stop = 0, len(lst)
    while start > stop-1:
        m = (start + stop)//2
        if x >= lst[m]:
            start = m
        else:
            stop = m
    if start == stop-1 and x >= lst[start]:
        return start
    # if (start == stop-1 and x < lst[start]) or start == stop i.e. empty lst
    return -1
    

def quick_select_ceil(lst, x):
    """Find the smallest item in lst greater than or equal to x. Return its index.
       If no item exists, return len(lst). i.e. x <= lst[0] xor lst[i-1] < x <= lst[i] xor lst[-1] < x."""
    start, stop = 0, len(lst)
    while start > stop-1:
        m = (start+stop-1)//2
        # 0 1 2 3  goes to  0 1  xor  2 3
        #   m                 m         m
        if x <= lst[m]:
            stop = m+1
        else:
            start = m+1
    if start == stop-1 and x <= lst[start]:
        return start
    # if (start == stop-1 and x > lst[start]) or start == stop i.e. empty lst
    return len(lst)
        

def nutsAndBolts(lst):
    nuts, bolts = [None], []
    i = 0
    while i < len(lst):
        piece = lst[i]
        i += 1
        if piece.isNut():
            nuts.append(piece)
        else:
            bolts.append(piece)
            break
    pivot = bolts[0]
    lStop, gStart = 1, len(lst) // 2
    for piece in lst[i:]:
        if piece.isNut():
            if piece.isGreater(pivot):
                gStart -= 1
                nuts[gStart] = piece
            elif piece.isLess(pivot):
                nuts[lStop] = piece
                lStop +=1
            else:
                nuts[0] = piece
        else:
            bolts.append(piece)
    assert lStop == gStart
    swap(nuts, 0, lStop - 1)
    swap(bolts, 0, lStop - 1)
    

def lineSegmentsIntersect(p0, p1, p2, p3):
    s1x, s1y = p0
    e1x, e1y = p1
    s2x, s2y = p2
    e2x, e2y = p3
    def slope(sx, sy, ex, ey):
        if sy-ey:
            return (sy-ey)/(sx-ex)
        else:
            return None
    m1 = slope(s1x, s1y, e1x, e1y)
    m2 = slope(s2x, s2y, e2x, e2y)
    if m1 is None and m2 is None:
        if s1x != s2x:
            return None
        s2y, e2y = sorted([s2y, e2y])
        return (s2y <= s1y)
    if m1 is not None and m2 is None:
        pass

from functools import reduce

def countingSort(alphabet, charLst): # sort multiset of alphabet
    count = [0] * len(alphabet)
    aMap = {v:i for i,v in enumerate(alphabet)}
    for char in charLst:
        count[aMap[char]] += 1
    return reduce(lambda x,y: x+y,
                  [[alphabet[i]]*count[i] for i in range(len(alphabet))])
                    
def radixLSDSort(alphabet, wordLst, maxLen=None):
    aMap = {v:i+1 for i,v in enumerate(alphabet)}
    aMap[None] = 0
    for i in range(1, maxLen+1):
        wordLst = _radixCountingSort(aMap, wordLst, -i)
        
def _radixCountingSort(aMap, wordLst, charIndex):
    words = [[] for _ in range(len(aMap)+1)]
    for word in wordLst:
        if charIndex < -len(word):
            words[aMap[word[charIndex]]].append(word)
        else:
            words[0].append(word)
    return reduce(lambda x,y: x+y, words)


## Trees ##

class BSTMap:

    class Node:

        def __init__(self, key, value, size=1, left=None, right=None):
            self.key = key
            self.value = value
            self.left = left
            self.right = right
            self.size = size

    def __init__(self):
        self.root = None

    def __len__(self):
        return self._size(self.root)

    def _size(self, node):
        if not node:
            return 0
        return node.size

    def get(self, key):
        node = self._getNode(self.root, key)
        if node:
            return node.key
        else:
            return None

    def _getNode(self, node, key):
        if not node:
            return None
        if key < node.key:
            return self._getNode(node.left, key)
        elif key > node.key:
            return self._getNode(node.right, key)
        else:
            return node

    def put(self, key, value):
        self.root = self._putNode(self.root, key, value)

    def _putNode(self, node, key, value):
        if not node:
            node = BSTMap.Node(key, value)
        elif key < node.key:
            node.left = self._putNode(node.left, key, value)
        elif key > node.key:
            node.right = self._putNode(node.right, key, value)
        else:
            node.value = value
        node.size = 1 + self._size(node.left) + self._size(node.right)
        return node

    def remove(self, key):
        self.root, returnVal = self._removeNode(self.root, key)
        return returnVal

    def _removeNode(self, node, key):
        if key < node.key:
            node.left, returnVal = self._removeNode(node.left, key)
        elif key > node.key:
            node.right, returnVal = self._removeNode(node.right, key)
        else:
            returnVal = node.value
            node = self._removeRoot(node)
        node.size = 1 + self._size(node.left) + self._size(node.right)
        return node, returnVal

    def _removeRoot(self, root):
        if not root.left:
            return root.right
        if not root.right:
            return root.left
        else:
            return self._removeSuccessor(root)

    # def _removeSuccessor(self, node):
    #     prev = node
    #     successor = node.right
    #     if not successor.left:
    #         successor.left = node.left
    #         return successor
    #     while successor.left:
    #         prev = successor
    #         successor = successor.left
    #     prev.left = successor.right
    #     successor.left, successor.right = node.left, node.right
    #     return successor

    def _removeSuccessor(self, node):
        prev = node
        successor = node.right
        while successor.left:
            prev = successor
            successor = successor.left
        node.key, node.value = successor.key, successor.value
        if prev is node:
            prev.right = successor.right
        else:
            prev.left = successor.right
        return node

    def items(self, order='inOrder'):
        return [(node.key, node.value) for node in
                getattr(self, '{}TraversalNode'.format(order))(self.root)]

    def inOrderTraversalNode(self, node):
        if not node:
            return []
        result = self.inOrderTraversalNode(node.left)
        result += [node]
        result += self.inOrderTraversalNode(node.right)
        return result

    def preOrderTraversalNode(self, node):
        if not node:
            return []
        result = [node]
        result += self.preOrderTraversalNode(node.left)
        result += self.preOrderTraversalNode(node.right)
        return result

    def postOrderTraversalNode(self, node):
        if not node:
            return []
        result = self.postOrderTraversalNode(node.left)
        result += self.postOrderTraversalNode(node.right)
        result += [node]
        return result

    def levelOrderTraversalNode(self, node):
        if not node:
            return []
        result = []
        N = deque([node])
        while N:
            node = N.popleft()
            result.append(node)
            if node.left:
                N.append(node.left)
            if node.right:
                N.append(node.right)
        return result


    def min(self, key='value'):
        node = self._getMinNode(self.root)
        if not node:
            return None
        if key == 'item':
            return (node.key, node.value)
        else:
            return getattr(node, key)

    def _getMinNode(self, node):
        if not node or not node.left:
            return node
        return self._getMinNode(node.left)

    def max(self, key='value'):
        node = self._getMaxNode(self.root)
        if not node:
            return None
        if key == 'item':
            return (node.key, node.value)
        else:
            return getattr(node, key)

    def _getMaxNode(self, node):
        if not node or not node.right:
            return node
        return self._getMaxNode(node.right)

    def floor(self, key):
        node = self._getFloorNode(self.root, key)
        if not node:
            return None
        return node.key

    def _getFloorNode(self, node, key):
        if not node:
            return None
        if key < node.key:
            return self._getFloorNode(node.left, key)
        if key > node.key:
            if node.right:
                return self._getFloorNode(node.right, key)
        return node

    def ceiling(self, key):
        node = self._getCeilingNode(self.root, key)
        if not node:
            return None
        return node.key

    def _getCeilingNode(self, node, key):
        if not node:
            return None
        if key > node.key:
            return self._getCeilingNode(node.right, key)
        if key < node.key:
            if node.left:
                return self._getCeilingNode(node.left, key)
        return node

    def select(self, k):
        return self._selectNode(self.root, k).key

    def _selectNode(self, node, k):
        if not node:
            return None
        t = self._size(node.left)
        if k < t:
            return self._selectNode(node.left, k)
        if k > t:
            return self._selectNode(node.right, k-t-1)
        else:
            return node

def mutateDLList(B):
    head, tail = _DLListNode(B.root)
    head.left = tail
    tail.right = head
    return head

def _DLListNode(node):
    if not node:
        return None, None
    middle = node
    head, leftTail = _DLListNode(node.left)
    if not leftTail:
        head = middle
    else:
        leftTail.right = middle
        middle.left = leftTail
    rightHead, tail = _DLListNode(node.right)
    if not rightHead:
        tail = middle
    else:
        middle.right = rightHead
        rightHead.left = middle
    return head, tail

def reverse(B):
    _reverseNode(B.root)

def _reverseNode(node):
    if not node:
        return
    node.left, node.right = _reverseNode(node.right), _reverseNode(node.left)
    return node

from collections import deque
from math import inf

def getBSTMapFromLevel(lst):
    B = BSTMap()
    if not lst:
        return B
    nodes = deque()
    B.root = BSTMap.Node(*lst[0])
    nodes.append((B.root, -inf, inf))
    for key, value in lst[1:]:
        child = BSTMap.Node(key, value)
        while True:
            parent, low, high = nodes.popleft()
            if child.key < parent.key and low < child.key:
                parent.left = child
                nodes.appendleft((parent, parent.key, high))
                nodes.append((child, low, parent.key))
                break
            if child.key > parent.key and high > child.key:
                parent.right = child
                nodes.append((child, parent.key, high))
                break
    return B
    
def checkBSTBalancedHeights(B):
    """Check every node has left and right subtrees of height within 1."""
    #  P    P    P
    # C    C C
    return _isBalancedHeight(root)[0]

def _isBalancedHeight(node):
    if not node:
        return True, 0
    leftBool, leftHeight = _isBalancedHeight(node.left)
    if not leftBool:
        return False, None
    rightBool, rightHeight = _isBalancedHeight(node.right)
    if rightBool and abs(leftHeight - rightHeight) < 2:
        return True, 1 + max(leftHeight, rightHeight)
    return False, None
    


class LLRBTMap: # Left leaning red black tree map
    
    class Node:
         
        def __init__(self, key, value, color=False,
                      left=None, right=None, size=1):
            self.key = key
            self.value = value
            self.color = color
            self.left = left
            self.right = right
            self.size = size
            
    def __init__(self):
        self.root = None
    
    def __len__(self):
        return self._size(self.root)
    
    def _size(self, node):
        if not node:
            return 0
        return node.size
    
    def _recalcSize(self, node): # node is not None
        node.size = self._size(node.left) + self._size(node.right) + 1
    
    def put(self, key, value):
        self.root = self._putNode(self.root, key, value)
    
    def _putNode(self, node, key, value):
        if not node:
            node = LLRBTMap.Node(key, value, True)
        elif key < node.key:
            node.left = self._putNode(node.left, key, value)
        elif key > node.key:
            node.right = self._putNode(node.right, key, value)
        else:
            node.value = value
        node.size = self._recalcSize(node)
        node = self._fixColors(node)
        return node
    
    def _fixColors(self, node):
        if self._color(node.right) and not self._color(node.left):
            node = self._rotateLeft(node)
        elif self._color(node.left) and self._color(node.left.left):
            node = self._rotateRight(node)
        elif self._color(node.left) and self._color(node.right):
            node = self._flip(node)
        return node
    
    def _rotateLeft(self, node):
        #   N              R
        #  b r           r   b
        # L   R    ->   N     RR
        #    ? b       b b
        #   RL RR     L   RL
        R = node.right
        node.right = R.left
        R.left = node
        if node.right:
            node.right.color = False
        R.color = node.color
        node.color = True
        R.size = node.size
        self._recalcSize(node)
        return R
    
    def _rotateRight(self, node)
        #     N              L
        #    r ?            r b
        #   L   R    ->   LL   N
        #  r b                b b
        # LL  LR             LR  R
        L = node.left
        node.left = L.right
        L.right = node
        if node.right:
            node.right.color = False
        L.color = node.color
        node.color = False
        L.size = node.size
        self._recalcSize(node)
        return L
    
    def _flip(self, node):
        #    ?       r
        #    N       N
        #   r r     b b
        #  L   R   L   R
        node.left.color = False
        node.right.color = False
        node.color = True
        return node
    
    def _color(self, node):
        if not node:
            return False
        return node.color
    
    def get(self, key):
        node = self._getNode(self.root, key)
        if node:
            return node.value
        return None
    
    def _getNode(self, node, key):
        if not node:
            return None
        if key < node.key:
            return self._putNode(node.left, key)
        if key > node.key:
            return self._putNode(node.right, key)
        else:
            return node
            
            

## Graphs ##

class UndirectedGraph:
    
    def __init__(self, numVertices):
        self._adjList = [[] for _ in range(numVertices)]
        self.numEdges = 0
    
    @property
    def numVertices(self):
        return len(self._adjList)
    
    def addEdge(self, i, j):
        self._inBounds(i); self._inBounds(j)
        if i > j:
            i, j = j, i
        self._addEdge(i, j)
    
    def _addEdge(self, i, j):
        self._adjList[i].append(j)
        self.numEdges += 1
    
    def hasEdge(self, i, j):
        self._inBounds(i); self._inBounds(j)
        if i > j:
            i, j = j, i
        self._hasEdge(i, j)
    
    def _hasEdge(self, i, j):
        return j in self._adjList[i]
    
    def removeEdge(self, i, j):
        self._inBounds(i); self._inBounds(j)
        if not self.hasEdge(i, j):
            raise ValueError('Graph has no edge between {} and {}'.format(i, j))
        if i > j:
            i, j = j, i
        self._removeEdge(i, j)
    
    def _removeEdge(self, i, j):
        adjI = self._adjList[i]
        k = adjI.index(j)
        adjI[k] = adjI[-1]
        adjI[].pop()
        self.numEdges -= 1
    
    def degree(self, i):
        smaller = sum(self._hasEdge(j, i) for j in range(i))
        larger = len(self._adjList[i])
        return smaller + larger
    
    def kthEdge(self, i, k):
        if k < 0:
            raise IndexError('kthEdge starts with index 0 i.e. first edge')
        iDeg = self.degree(i)
        if k > iDeg:
            raise IndexError('Vertex {} has less than {} edges'.format(i, k))
        largerDeg = len(self._adjList[i])
        smallerDeg = iDeg - largerDeg
        if k > smallerDeg:
            return self._adjList[i][k - smallerDeg - 1]
        return self._smallerKthEdge(i, k)
    
    def _smallerKthEdge(self, i, k):
        for j in range(i):
            if self._hasEdge(j, i):
                if k == 1:
                    return (i, j)
                k -= 1
    
    def _kthEdge(self, i, k):
        for j in range(i):
            if self._hasEdge(j, i):
                if k == 1:
                    return (i, j)
                k -= 1
        return self._adjList[i][k-1]
    
    def listEdges(self, i):
        result = [j for j in range(i) if self._hasEdge(j, i)]
        result.extend(self._adjList[i])
        return result
    
    def neighbors(self, i):
        return self.listEdges(i)
    
    def addVertex(self):
        self._adjList.append([])

    def _inBounds(self, i):
        if i < 0:
            raise IndexError('Graph edge index {} cannot be negative'.format(i))
        if i >= self.numVertices:
            raise IndexError('Graph edge index {} must be less than
                              numVertices {}'.format(i, self.numVertices))


class WeightedDirectedGraph:
    
    
    def __init__(self, numVertices):
        self._entries = [[] for _ in range(numVertices)]
        self._exits = [[] for _ in range(numVertices)]
        self.numEdges = 0
    
    @property
    def numVertices(self):
        return len(self._entries)
        
    
    class Edge:
        
        def __init__(self, source, target, weight):
            self.source = source
            self.target = target
            self.weight = weight
    
    
    def addEdge(self, source, target, weight):
        self._inBounds(source); self._inBounds(target)
        self._addEdge(source, target, weight)
    
    def _addEdge(self, source, target, weight):
        E = Edge(source, target, weight)
        self._exits[source].append(E)
        self._entries[target].append(E)
        self.numEdges += 1
        
    
    def hasEdge(self, source, target):
        self._inBounds(source); self._inBounds(target)
        return self._getEdge(source, target) is not None
    
    def _getEdge(self, source, target):
        if len(self._exits[source]) > len(self._entries[target]):
            return self._entries[target][self._getEntryIndex(source, target)]
        return self._exits[target][self._getExitIndex(source, target)]
        
    def _getExitIndex(self, source, target):
        exits = self._exits[source]
        for i in range(len(exits)):
            if exits[i].target == target:
                return i
    
    def _getEntryIndex(self, source, target):
        entries = self.entries[target]
        for i in range(len(entries)):
            if entries[i].source == source:
                return i
                
    
    def changeWeight(self, source, target, weight):
        self._inBounds(source); self._inBounds(target)
        E = self._getEdge(source, target)
        E.weight = weight
    
    
    def listOutEdges(self, source):
        return list(self._exits[source]))
    
    def listInEdges(self, target):
        return list(self._entries[target])
    
    def neighbors(self, source):
        return [E.target for E in self._exits[source]]
    
    def removeEdge(self, source, target):
        if not self.hasEdge(source, target):
            raise ValueError('Graph has no edge
                              from {} to {}'.format(source, target))
        self._inBounds(source); self._inBounds(target)
        self._removeOutEdge(source, target)
        self._removeInEdge(source, target)
    
    def _removeOutEdge(self, source, target):
        exits = self._exits[source]
        i = self._getExitIndex(source, target)
        exits[i] = exits[-1]
        exits.pop()
    
    def _removeInEdge(self, source, target):
        entries = self.entries[target]
        i = self._getEntryIndex(source, target)
        entries[i] = entries[-1]
        entries.pop()
    
    def removeOutEdges(self, source):
        self._inBounds(source)
        self._removeOutEdges(source)
    
    def _removeOutEdges(self, source):
        for E in self._exits[source]:
            self._removeInEdge(source, E.target)
        self._exits[source] = []
    
    def removeInEdges(self, target):
        self._inBounds(target)
        self._removeInEdges(target)
    
    def _removeInEdges(self, target):
        for E in self._entries[target]:
            self._removeOutEdge(E.source, target)
        self._entries[target] = []
    
    def inDegree(self, target):
        return len(self._entries[target])
    
    def outDegree(self, source):
        return len(self._exits[source])
    
    def degree(self, vertex):
        return self.inDegree(vertex) + self.outDegree(vertex)
    
    def _inBounds(self, i):
        if i < 0:
            raise IndexError('Graph vertex {} must be nonnegative'.format(i))
        if i >= self.numVertices:
            raise IndexError('Graph vertex {} must be less than
                              numVertices {}'.format(i, self.numVertices))
                              
class DFSPaths: # Depth First Search Paths
    
    def __init__(self, G, source):
        self.graph = G
        self._marked = [False] * G.numVertices
        self._sourceFrom = [None] * G.numVertices
        self.source = source
        self.preOrder = []
        self.postOrder = []
        self._search(source)
    
    def _search(self, v):
        self.preOrder.append(v)
        self._marked[v] = True
        for n in self.graph.neighbors(v):
            if not self._marked[n]:
                self._sourceFrom[n] = v
                self._search(n)
        self.postOrder.append(v)
    
    def hasPath(self, target):
        return self._marked(target)
    
    def getPath(self, target):
        result = []
        if not self.hasPath(target):
            return result
        while target != self.source:
            result.append(target)
            target = self._sourceFrom(target)
        result.append(self.source)
        result.reverse()
        return result


class BFSPaths:
    
    def __init__(self, G, source):
        self.graph = G
        self._marked = [False] * G.numVertices
        self._sourceFrom = [None] * G.numVertices
        self._distTo = [None] * G.numVertices
        self.source = source
        self.levelOrder = []
        self._search()
    
    def _search(self):
        self.levelOrder.append(self.source)
        self._marked[self.source] = True
        self._distTo[self.source] = 0
        i = 0
        while i < len(self.levelOrder):
            v = self.levelOrder[i]
            for n in self.graph.neighbors(v):
                if not self._marked[n]:
                    self.levelOrder.append(n)
                    self._marked[n] = True
                    self._sourceFrom[n] = v
                    self._distTo[n] = 1 + self._distTo[v]
            i += 1
    
    def hasPath(self, target):
        return self._marked[target]
    
    def getPath(self, target):
        result = []
        if not self.hasPath(target):
            return result
        while target != self.source:
            result.append(target)
            target = self._sourceFrom[target]
        result.append(self.source)
        result.reverse()
        return result
    
    def distTo(self, target):
        return self._distTo[target]

from itertools import combinations as comb
from collections import defaultdict

class ConnectedComponents: # Undirected Graph
    
    def __init__(self, G)
        self.graph = G
        self._marked = [False] * G.numVertices
        self._sourceFrom = [None] * G.numVertices
        self._DSet = WQUDSWithPathCompression(G.numVertices)
        self._cycleEnds = []
        self._search()
    
    def _search(self):
        for v in range(self.graph.numVertices):
            if not self._marked[v]:
                self._dfs(v)
                
    def _dfs(self, v):
        self._marked[v] = True
        for n in self.graph.neighbors(v):
            if not self._marked[n]:
                self._sourceFrom[n] = v
                self._DSet.connect(v,n)
                self._dfs(n)
            else:
                self._cycleEnds.append(n,v)
    
    def hasCycle(self):
        return bool(self._cycleEnds)
    
    def getCycles(self):
        result = []
        for v,w in self._cycleEnds:
            result.append(self._getPathToIntersection(v,w))
        dup = self._getDuplicates(self._cycleEnds, key=lambda x: x[1])
        # dup = list(list(tuple))
        for d in dup:
            for a,b in comb(d,2): # tuple(tuple(int)) in comb(d,2)
            #   i      a = (l, n)
            #  j k     b = (m, n)
            # l   m    want: l j i k m + n
            #   n
                deg1Cycle = self._getPathToIntersection(a[0],b[0]) + [a[1]]
                result.append(deg1Cycle)
        return result
    
    def _getDuplicates(lst, key):
        TupleMap = defaultdict(list)
        for v,w in lst:
            TupleMap[w].append((v,w))
        return [val for val in TupleMap.values() if len(val) > 1]
    
    def _getPathToIntersection(self, v, w):
        pv = self.getPathFrom(v)
        pw = self.getPathFrom(w)
        diff = len(pv) - len(pw)
        if diff < 0:
            return pv[:0:-1] + pw[-diff:]
        else:
            return pv[:diff: -1] + pw[::-1]
            
    
    def getPathFrom(self, source):
        result = []
        if not self.hasPath(source):
            return result
        while source != None:
            result.append(source)
            source = self._sourceFrom[source]
        return result
    
    def isConnected(self, v, w):
        return self._DSet.isConnected(v, w)
    
    def getRepresentative(self, v):
        return self._DSet.getRepresentative[v]
    
    def size(self, v):
        return self._DSet.getSize(v)
    
    def numComponents(self):
        return self._DSet.numComponents()
        

class Bridges:
    
    def __init__(self, G):
        self.graph = G
        self._preOrder = [None] * G.numVeritices
        # preOrder[i] = order in which i was visited
        self._low = [None] * G.numVertices
        # low[i] = min(preOrder[k] for k >= i
        #              where k is in the dfs subtree root at i so that
        #              preOrder[k] > preOrder[i])
        # if u has a child v with low[v] >= pre[u]

class DirectedCycle:
    
    def __init__(self, G):
        self.graph = G
        self._marked = [False] * G.numVertices
        self._onStack = []
        self._sourceFrom = [None] * G.numVertices
        self.cycle = []
        for v in range(G.numVertices):
            if self.cycle:
                break
            if not self._marked[v]:
                self.dfs(v)
        
    def dfs(self, v):
        self._marked[v] = True
        self._onStack[v] = True
        for w in G.neighbors(v):
            if self.cycle:
                return
            if not self._marked[w]:
                self._sourceFrom[w] = v
                self.dfs[w]
            elif self._onStack[w]:
                self.cycle = self._getCycle(w, v)
        self._onStack[v] = False
    
    def _getCycle(self, source, target):
        while target != source:
            self.cycle.append(target)
            target = self._sourceFrom[target]
        self.cycle.append(source)
            
class TopologicalSort: # Directed Graph

    def __init__(self, G):
        self.graph = G
        self._marked = [False] * G.numVertices
        self._sourceFrom = [None] * G.numVertices
        self._postOrder = []
        self._onStack = []
        self._hasCycle = False
        for v in self.zeroVertices():
            self.dfs(v)
            if self._hasCycle:
                return None
        return reversed(self._postOrder))
        
    def zeroVertices(self):
        returnLst = []
        for v in range(G.numVertices):
            if not G.inDegree(v):
                returnLst.append(v)
        return returnLst

    def dfs(self, v):
        self._onStack[v] = True
        self._marked[v] = True
        for w in G.neighbors(v):
            if not self._marked[w]:
                self.dfs(w)
            elif self._onStack[w]:
                self._hasCycle = True
                return
        self._onStack[v] = False
        self._postOrder.append(v)

class LazyPrimsMST: # Weighted Undirected Graph
                    # Minimum Spanning Tree, PQ by weight of Edges
                    # Bottum up
    
    def __init__(self, G):
        self.graph = G
        self._fringe = MinHeap()
        self._marked = [False] * G.numVertices
        self.edgeList = []
        for v in range(G.numVertices):
            if not self._marked[v]:
                self._prims(v)
    
    def _prims(self, v):
        self._addFringe(v)
        self._search()
        self._fringe.clear()
    
    def _addFringe(self, v):
        self._marked[v] = True
        for E in self.graph.outEdges(v):
            self._fringe.add(E.weight, E)
    
    def _search(self):
        while self._fringe and not all(self._marked):
            E = self._fringe.pop()
            v = None
            if not self._marked[E.source]:
                v = E.source
            elif not self._marked[E.target]
                v = E.target
            if v is not None:
                self.edgeList.append(E)
                self._addFringe(v)

class EagerPrimsMST: # Prims with only minimum non tree edges in PQ
    
    def __init__(self, G):
        self.graph = G
        self._fringe = MinHeap()
        self._marked = [False] * G.numVertices
        self._minWeight = [None] * G.numVertices
        self.edgeList = []
        for v in range(G.numVertices):
            if not self._marked[v]:
                self._prims(v)
    
    def _prims(self, v):
        self._addFringe(v)
        self._search()
        self._fringe.clear()
    
    def _addFringe(self, v):
        self._marked[v] = True
        for E in self.graph.listEdges(v):
            # v in MST, w is not
            if not self._marked[E.source]:
                w = E.source
            else:
                w = E.target
            if not self._minWeight[w]:
                self._fringe.add(E.weight, E)
            elif self._minWeight[w].weight > E.weight:
                self._fringe.replace(self._minWeight[w], E)
                self._minWeight[w] = E
    
    def _search(self):
        while self._fringe and not all(self._marked):
            E = self._fringe.pop()
            v = None
            if not self._marked[E.source]:
                v = E.source
            elif not self._marked[E.target]
                v = E.target
            if v is not None:
                self.edgeList.append(E)
                self._addFringe(v)

class KruskalsMST: # Top down

    def __init__(self, G):
        self.graph = G
        self._treeDS = WQUDSWithPathCompression(G.numVertices)
        self._fringePQ = MinHeap()
        self.edgeList = []
        for E in G.listAllEdges():
            self._fringePQ.add(E.weight, E)
        self._kruskals()
    
    def _kruskals(self):
        while self._fringePQ and len(self.edgeList) < self.graph.numVertices-1:
            E = self._fringePQ.pop()
            if not self._treeDS.isConnected(E.source, E.target):
                self._treeDS.connect(E.source, E.target)
                self.edgeList.append(E)

from math import inf
class DijkstrasSPT: # Positive Weighted Directed Graph
                    # Shortest Paths Tree
                    # BFS to PQ, sourceFrom, distTo, marked
    def __init__(self, G, source):
        self.graph = G
        self.source = source
        self._edgeTo = [None] * G.numVertices
        self._distTo = [inf] * G.numVertices
        self._fringe = MinHeap()
        self._dijkstras()
    
    def _dijkstras(self):
        self._distTo[self.source] = 0
        self._fringe.add((0, self.source))
        self._search()
    
    def _search(self):
        while self._fringe:
            v = self._fringe.pop()[1]
            for E in self.graph.listOutEdges(v):
                self._relax(E, v)
    
    def _relax(self, E, v):
        w = E.target
        dist = self._distTo[v] + E.weight
        if dist < self._distTo[w]:
            self._edgeTo[w] = v
            self._distTo[w] = dist
            if w in self._fringe:
                self._fringe.changePriority(w, dist)
            else:
                self._fringe.add(self._distTo[w], w)
    
    def hasPathTo(self, target):
        return self._distTo[target] < inf
    
    def getPathTo(self, target):
        path = []
        while target != self.source:
            path.append(target)
            target = self.edgeTo[target]
        path.append(self.source)
        return reversed()
    

class AStarSP: # Dijkstras with Heuristic toward target
               # 0 < Heuristic(v) < actual distance from v to target

    def __init__(self, G, source, target, heuristic=lambda x: x):
        self.graph = G
        self.source = source
        self.target = target
        self._sourceFrom = [None] * G.numVertices
        self._distTo = [inf] * G.numVertices
        self._fringe = MinHeap()
        self.heuristic = heurisitc
        self._aStar()
    
    def _aStar(self):
        self._distTo[self.source] = 0
        self._fringe.add(self.source, 0)
        self._search()
    
    def _search(self):
        while self._fringe:
            v = self._fringe.pop()
            for E in self.graph.listOutEdges(v):
                self._relax(E)
    
    def _relax(self, E):
        v, w = E.source, E.target
        dist = E.weight + self._distTo[v]
        if dist < self._distTo[w]:
            self._sourceFrom[w] = v
            self._distTo[w] = dist
            prior = dist + self.heuristic(w)
            if w in self._fringe:
                self._fringe.changePriority(w, prior)
            else:
                self._fringe.add(w, prior)
    
    def dist(self):
        return self._distTo[self.target]
    
    def hasPath(self):
        return self.dist() < inf
    
    def pathTo(self):
        assert self.hasPath()
        path = []
        target = self.target
        while target != self.source:
            path.append(target)
            target = self._sourceFrom[target]
        path.append(self.source)
        return reversed(path)
                

class Trie:
    
    def Node(self, R):
        self.exists = False
        self.children = [None] * R
    
    def __init__(self, alphabet):
        self.R = len(alphabet)
        self.root = Node(R)
        self.alphabetMap = {v:i for i,v in enumerate(alphabet)}
    
    def has(self, word):
        N = self._getNode(word)
        if N is None:
            return False
        return N.exists
    
    def _getNode(self, word):
        N = self.root
        for char in word:
            if N is None:
                break
            N = N.children[self.alphabetMap[char]]
    
    def insert(self, word):
        self.root = self._insert(self.root, word, 0)
    
    def _insert(node, word, i):
        if not node:
            node = Node(self.R)
        if i == len(word):
            node.exists = True
        else:
            k = self.alphabetMap[word[i]]
            node = self._insert(node.children[k], word, i+1)
        return node
    
    def keysWithPrefix(self, word):
        stack, result = [], []
        N = self._getNode(word)
        if N:
            stack.append((N, word))
        while stack:
            N, word = stack.pop()
            if N.exists:
                result.append(word)
                for i in range(self.R):
                    if N.children[i]:
                        stack.append((N.children[i], word + self.alphabet[i]))
    
    def longestPrefix(self, word):
        N = self.root
        for i in range(len(word)):
            if N:
                N = N.children[alphabetMap[word[i]]]
            else:
                return word[:i]
    
    
Deck
    Cards

Player
    Hand
    Money
Dealer

Table
    Game
    Players
        Hand
        Strategy
        Money
    Dealer
        Hand
        Strategy
    Deck
        Cards
    

BlackJack
    Game
        Hand
        
    

## Bit Manipulation ##
# 0011 1000 0110 1001
# 1100 0111 1001 0110
# add + 1 = 1 0000 0000 0000 0000

def sol(lst):
    if len(lst) < 3:
        return lst[0]
    for i in range(0:len(lst)-2:3):
        if lst[i] != lst[i+2]:
            return lst[i]

import defaultdict

def sol(lst):
    d = defaultdict(int)
    for x in lst:
        d[x] += 1
    return [k for k,v in d.items() if v == 2][0]

a3[b2[c1[d]]]e
'a' + 3*('b' + 2*('c' + 1*('d'))) + 'e'
eval("'a' + 3*('v' +")

from string import ascii_lowercase as asc
def sol(s):
    if len(s) < 2:
        return s
    # 'a' -> "'a'+"
    # '3' -> "3*"
    # '[' -> "("
    # ']' -> ")" or ")+"
    # take care of case: ) + 'a'
    charMap = {}
    for a in asc:
        charMap[a] = "'{}'+".format(a)
    for d in '0123456789':
        charMap[d] = "{}*".format(d)
    charMap['['] = "("
    charMap[']'] = ")"
    result = ""
    prev = ""
    for curr in s:
        if result[-1] == "+" and curr == ']':
            result = result[:-1]
        result += charMap[curr]
        if prev == ']' and curr != ']':
            result += "+"
        prev = curr
    if result[-1] == "+":
        result = result[:-1]
    return eval(result)


class Rectangle:
    
    def __init__(self, arr):
        self.arr = arr
        self.numrows = len(arr)
        self.numcols = len(arr[0])
        self.rowsums = [[[None]*(self.numcols-i-1) for i in range(self.numcols)]
                        for _ in range(self.numrows)]
        self._initrowsums()
        
    def _initrowsums(self):
        for row in range(self.numrows):
            for i in range(self.numcols):
                for j in range(i+1, self.numcols):
                    self.rowsums[row][i][j-i-1] = sum(self.arr[row][i:j])
    
    def query(self, row1, col1, row2, col2): # O(row2-row1)
        return sum(self.rowsums[j][col1][col2-col1-1]
                   for j in range(row1, row2+1)))
    
    def update(self, row, col, val):
        diff = val - self.arr[row][col]
        self.arr[row][col] = val
        for i in range(col+1):
            for j in range(col+1, self.numcols+1):
                self.rowsums[row][i][j-i-1] += diff

    
    
                
from collections import defaultdict

class Solution:
    def areSentencesSimilarTwo(self, words1, words2, pairs):
        # Use quickunion ds to get equivalence classes
        DS = WeightedQuickUnionDSWithPathCompression()
        print(DS)
        WordMap = {}
        for u,v in pairs:
            if u not in WordMap:
                WordMap[u] = len(WordMap)
                DS.additem()
            if v not in WordMap:
                WordMap[v] = len(WordMap)
                DS.additem()
            DS.connect(WordMap[u], WordMap[v])
        print(WordMap)
        print(DS._parent)
        ComponentMap = {}
        cnum = 0
        for C in DS.getComponents():
            print(ComponentMap)
            for j in C:
                ComponentMap[j] = cnum
            cnum += 1
        # compare counts for each component of each sentence
        counts = [0] * cnum
        for x in words1:
            counts[ComponentMap[WordMap[x]]] += 1
        for x in words2:
            counts[ComponentMap[WordMap[x]]] -= 1
        return not any(counts)



# C[i][j] is cost of choosing j first from i numbers
        C = [[0 for _ in range(i+1)] for i in range(n+1)]
        # M[i] is getMoneyAmount(i) is max(C[i])
        M = [0 for _ in range(n+1)]
        # Base case: C[i][0] = 0
        # Inductive step:
        for i in range(1, n+1):
            maxcost = i
            for j in range(1, i+1):
                left = max(C[j-1])
                right = max
                C[i][j] = j +
                

Str Manipulation
-generate perms w for loops
-convert to int to check properties
    ex: Closest Times
-replace, find(x, start, stop)
-read backwards, middle out, out in
-shift, cycles, equivalence class, group by len
    ex: letter shifts equivalence classes
    
Ranges
-sort before merging
-query: sum[:j] - sum[:i]

Stack
-str manipulation: file path, parentheses
    
Sliding window
-two pointers
    ex: contiguous arrays, quicksort, next perm

Binary search
-median, supremem, infimum
    ex: twosum, sorted

HashMap
-frequency counter: H[x] = i
-number of types: len(H) if not any H[x] == 0
    ex: Fruit basket counter
    
PriorityQueue
-access or delete min attribute


Queue
-stream of data
    ex: message stream with timestamps
-root to leaf path, iterate through root
    ex: count node complete binary tree
    
Array
-O(1) remove, arr[i] = arr[-1], arr.pop()




