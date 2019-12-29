
class QuickFindDS:
# Represented as connected components as an array.
# connect slow, isConnected is fast
# constructor Theta(n), connect Theta(1), isConnected Theta(n)
# 0 1 2 3 4 5
# 0 1 0 0 4 5

    def __init__(self, length):
        """Create a collection of length of disjoint (singleton)
           sets.
        >>> S = QuickFindDS(0)
        >>> S.array
        []
        >>> T = QuickFindDS(7)
        >>> T.array
        [0, 1, 2, 3, 4, 5, 6]
        """
        self.array = [None] * length
        for i in range(len(self.array)):
            self.array[i] = i

    def connect(self, i, j):
        """Connect items i and j by pointing all items
           that point to where j points, to point to
           where i points.
        >>> S = QuickFindDS(6)
        >>> S.connect(1, 3)
        >>> S.array
        [0, 1, 2, 1, 4, 5]
        >>> S.connect(3, 4)
        >>> S.array
        [0, 1, 2, 1, 1, 5]
        >>> S.connect(1, 0)
        >>> S.array
        [1, 1, 2, 1, 1, 5]
        >>> S.connect(2, 0)
        >>> S.array
        [2, 2, 2, 2, 2, 5]
        """
        x = self.array[i]
        y = self.array[j]
        for k in range(len(self.array)):
            if self.array[k] == y:
                self.array[k] = x

    def isConnected(self, i, j):
        """Check if i and j are connected.
        >>> S = QuickFindDS(6)
        >>> S.array = [0, 2, 1, 4, 2, 2]
        >>> S.isConnected(1, 0)
        False
        >>> S.isConnected(1, 2)
        False
        >>> S.isConnected(4, 5)
        True
        """
        return self.array[i] == self.array[j]

class QuickUnionDS():
# Represented as connected components as an array.
# We connect parents as a Tree.
# connect can be slow, isConnected can be slow
# constructor Theta(n), connect O(n), isConnected O(n)

    def __init__(self, length):
        self.parent = [None] * length
        for i in range(len(self.parent)):
            self.parent[i] = i

    def find(self, i):
        """ Find parent of item by following tree.
        >>> S = QuickUnionDS(5)
        >>> S.parent = [0, 0, 1, 2, 4]
        >>> S.find(3)
        0
        >>> S.find(4)
        4
        """
        while i != self.parent[i]:
            i = self.parent[i]
        return i

    def connect(self, i, j):
    # maybe optimize by counting iterations of parent calculation
    # connect the smaller to the larger
        """Connect i and j by pointing the parent of j
           to the parent of i.
        >>> S = QuickUnionDS(6)
        >>> S.connect(1, 3)
        >>> S.parent
        [0, 1, 2, 1, 4, 5]
        >>> S.connect(3, 4)
        >>> S.parent
        [0, 1, 2, 1, 1, 5]
        >>> S.connect(1, 0)
        >>> S.parent
        [1, 1, 2, 1, 1, 5]
        >>> S.connect(2, 0)
        >>> S.parent
        [1, 2, 2, 1, 1, 5]
        """
        pi = self.find(i)
        pj = self.find(j)
        self.parent[pj] = pi

    def isConnected(self, i, j):
        """Check if i and j are connected.
        >>> S = QuickUnionDS(7)
        >>> S.parent = [1, 2, 2, 1, 4, 5, 2]
        >>> S.isConnected(0, 0)
        True
        >>> S.isConnected(0, 3)
        True
        >>> S.isConnected(3, 6)
        True
        >>> S.isConnected(4, 6)
        False
        """
        return self.find(i) == self.find(j)



class WeightedQuickUnionDS():
# Represented as connected components as an array.
# Connect parents as a Tree.
# Track size of tree so that smaller connects to larger.
# constructor Theta(n), connect O(logn), isConnected O(logn)

    def __init__(self, length):
        """ Track parent and size of tree.
        >>> S = WeightedQuickUnionDS(8)
        >>> S.parent
        [0, 1, 2, 3, 4, 5, 6, 7]
        >>> S.size
        [1, 1, 1, 1, 1, 1, 1, 1]
        """
        QuickUnionDS.__init__(self, length)
        self.size = [None] * length
        for i in range(length):
            self.size[i] = 1

    def find(self, i):
        return QuickUnionDS.find(self, i)

    def connect(self, i, j):
        """ Connect i and j by pointing the parent of smaller size
            to the parent of larger size.
        >>> S = WeightedQuickUnionDS(6)
        >>> S.connect(1, 3)
        >>> S.parent
        [0, 1, 2, 1, 4, 5]
        >>> S.size
        [1, 2, 1, 1, 1, 1]
        >>> S.connect(3, 4)
        >>> S.parent
        [0, 1, 2, 1, 1, 5]
        >>> S.size
        [1, 3, 1, 1, 1, 1]
        >>> S.connect(1, 0)
        >>> S.parent
        [1, 1, 2, 1, 1, 5]
        >>> S.size
        [1, 4, 1, 1, 1, 1]
        >>> S.connect(2, 0)
        >>> S.parent
        [1, 1, 1, 1, 1, 5]
        >>> S.size
        [1, 5, 1, 1, 1, 1]
        """
        pi, pj = self.find(i), self.find(j)
        if pi == pj:
            return None
        if self.size[pi] >= self.size[pj]:
            self.parent[pj] = pi
            self.size[pi] += self.size[pj]
        else:
            self.parent[pi] = pj
            self.size[pj] += self.size[pi]

    def isConnected(self, i, j):
        return QuickUnionDS.isConnected(self, i, j)

    def disConnect(self, i, j): # ???
        if i == j:
            return None
        def findFirstMutualParent(self, i, j):
            if i == j:
                return i
            maxHeight = math.floor(math.log(len(self.parent)))
            iparents, jparents = list(), list()
            iparents.append(i); jparents.append(j)
            while i != self.parent[i] and j != self.parent[j]:
                if i == j:
                    return i
                if i in jparents:
                    return i
                iparents.append(i)
                if j in iparents:
                    return j
                jparents.append(j)
                i, j = self.parent[i], self.parent[j]

        pi, pj = self.find(i), self.find(j)
        if self.size[pi] > self.size[pj]:
            newpj = self.findSecond(j)
            self.parent[pj] = newpj
            self.size[pi] -= self.size[pj]
        else:
            newpi = self.findSecond(i)
            self.parent[pi] = newpi
            self.size[pj] -= self.size[pi]


class WeightedQuickUnionDSWithPathCompression():
# Represented as connected components as an array.
# Connect parents as a Tree.
# Track size of tree so that smaller connects to larger.
# constructor Theta(n), connect O(logn), isConnected O(logn)

    def __init__(self, length):
        WeightedQuickUnionDS.__init__(self, length)

    def find(self, i):
        """ Find parent of item by following tree.
            Update children to parent directly.
        >>> S = WeightedQuickUnionDSWithPathCompression(5)
        >>> S.parent = [0, 0, 1, 2, 4]
        >>> S.find(3)
        0
        >>> S.parent
        [0, 0, 0, 0, 4]
        >>> S.find(4)
        4
        >>> S.parent
        [0, 0, 0, 0, 4]
        """
        if i == self.parent[i]:
            return i
        else:
            self.parent[i] = self.find(self.parent[i])
            return self.parent[i]

    def connect(self, i, j):
        return WeightedQuickUnionDS.connect(self, i, j)

    def isConnected(self, i, j):
        """ Check if i and j are connected.
            Implicitly update parents with find method.
        """
        return WeightedQuickUnionDS.isConnected(self, i, j)


###############
## EXERCISES ##
###############



###########
## TESTS ##
###########

#import unittest
#class InsertionsTest(unittest.TestCase):

    #def test1(self):


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    #unittest.main()




class WeightedQuickUnionDSWithPathCompression():
    def __init__(self, length):
        self.parent = list(range(length))
        self.size = [1 for _ in range(length)]
        self._length = length
        
    def __len__(self):
        return self._length

    def find(self, i):
        if i == self.parent[i]:
            return i
        else:
            self.parent[i] = self.find(self.parent[i])
            return self.parent[i]

    def connect(self, i, j):
        pi, pj = self.find(i), self.find(j)
        if pi == pj:
            return None
        if self.size[pi] >= self.size[pj]:
            self.parent[pj] = pi
            self.size[pi] += self.size[pj]
        else:
            self.parent[pi] = pj
            self.size[pj] += self.size[pi]

    def isConnected(self, i, j):
        return self.find(i) == self.find(j)
        
    def sizeLargestConnectedComponent(self):
        marked = [False for _ in range(self.size)]
        for i in range(self.size):
            if not marked[i]:
                self._pathCompress(marked, i)
        parentCount = {}
        for p in self.parent:
            count = parentCount.get(p, 0)
            parentCount[p] = count + 1
        return max(parentCount.values())
    
    def _pathCompress(marked, i):
        if i != self.parent[i]:
            self.parent[i] = self._pathCompress(marked, i)
        marked[i] = True
        return self.parent[i]