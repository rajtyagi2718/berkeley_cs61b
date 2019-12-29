

#################
## Graph Types ##
#################

class UndirectedGraph:

    def __init__(self, numVertices):
        self.adjList = [[] for _ in range(numVertices)]
        self.numEdges = 0

    @property
    def numVertices(self):
        return len(self.adjList)

    def addEdge(self, i, j):
        if self.hasEdge(i, j):
            return None
        i, j = min(i, j), max(i, j)
        self.adjList[i].append(j)
        self.numEdges += 1

    def hasEdge(self, i, j):
        self.inBounds(i)
        self.inBounds(j)
        i, j = min(i, j), max(i, j)
        return j in self.adjList[i]

    def _hasEdge(self, i, j):
        return j in self.adjList[i]

    def addVertices(self, num=1):
        self.adjList += [[] for _ in range(num)]

    def degree(self, i):
        self.inBounds(i)
        return self.smallerDegree(i) + self.largerDegree(i)

    def smallerDegree(self, i):
        return sum([self._hasEdge(j, i) for j in range(i)])

    def largerDegree(self, i):
        return len(self.adjList[i])

    def kthEdge(self, i, k):
        for j in range(i):
            if self._hasEdge(j, i):
                if k == 0:
                    return j
                k -= 1
        self.inBounds(i)
        for j in self.adjList[i]:
            if k == 0:
                return j
            k -= 1
        raise ValueError('vertex {} has too few edges'.format(i))

    def listEdges(self, i):
        returnList = []
        for j in range(i):
            if self._hasEdge(j, i):
                returnList.append(j)
        returnList.extend(self.adjList[i])
        return returnList

    def removeEdge(self, i, j):
        if not self.hasEdge(i, j):
            raise ValueError('graph has no edge here')
        i, j = min(i, j), max(i, j)
        jIndex = self.adjList[i].index(j)
        self.adjList[i][jIndex], self.adjList[i][-1] = (
                self.adjList[i][-1], self.adjList[i][jIndex])
        self.adjList[i].pop()
        self.numEdges -= 1

    def _removeEdge(self, i, j):
        jIndex = self.adjList[i].index(j)
        self.adjList[i][jIndex], self.adjList[i][-1] = (
                self.adjList[i][-1], self.adjList[i][jIndex])
        self.adjList[i].pop()
        self.numEdges -= 1

    def removeEdges(self, i):
        self.inBounds(i)
        for j in range(i):
            if self._hasEdge(j, i):
                self._removeEdge(j, i)
        self.numEdges -= len(self.adjList[i])
        self.adjList[i] = []

    def inBounds(self, i):
        if i < 0:
            raise IndexError("vertex '{}' must be non negative".format(i))
        if i >= self.numVertices:
            raise IndexError("vertex '{}' greater than graph size \
                '{}'".format(i, self.numVertices))

class WeightedUndirectedGraph:

    def __init__(self, numVertices):
        self.adjList = [[] for _ in range(numVertices)]
        self.numEdges = 0

    @property
    def numVertices(self):
        return len(self.adjList)

    class Edge:

        def __init__(self, source, target, weight=1):
            self.source = source
            self.target = target
            self.weight = weight

    def addEdge(self, i, j, w=1):
        if self.hasEdge(i, j):
            return None
        i, j = min(i, j), max(i, j)
        E = WeightedUndirectedGraph.Edge(i, j, w)
        self.adjList[i].append(E)
        self.numEdges += 1

    def hasEdge(self, i, j):
        self.inBounds(i)
        self.inBounds(j)
        i, j = min(i, j), max(i, j)
        for E in self.adjList[i]:
            if E.target == j:
                return True
        return False

    def _hasEdge(self, i, j):
        for E in self.adjList[i]:
            if E.target == j:
                return True
        return False

    def addVertices(self, num=1):
        self.adjList += [[] for _ in range(num)]

    def degree(self, i):
        self.inBounds(i)
        return self.smallerDegree(i) + self.largerDegree(i)

    def smallerDegree(self, i):
        return sum([self._hasEdge(j, i) for j in range(i)])

    def largerDegree(self, i):
        return len(self.adjList[i])

    def kthEdge(self, i, k):
        for j in range(i):
            if self._hasEdge(j, i):
                if k == 0:
                    return j
                k -= 1
        self.inBounds(i)
        for j in self.adjList[i]:
            if k == 0:
                return j
            k -= 1
        raise ValueError('vertex {} has too few edges'.format(i))

    def listEdges(self, i): # only indices, as used in algorithms
        returnLst = []
        for j in range(i):
            if self._hasEdge(j, i):
                returnList.append(j)
        returnList.extend([E.target for E in self.adjList[i]])
        return returnLst

    def listAllEdges(self, i=None):
        if i == None:
            return [E for A in self.adjList for E in A]
        returnLst = []
        for j in range(i):
            for E in self.adjList[j]:
                if E.target == i:
                    returnLst.append(E)
        returnLst.extend(self.adjList[i])
        return returnLst

    def removeEdge(self, i, j):
        if not self.hasEdge(i, j):
            raise ValueError('graph has no edge here')
        i, j = min(i, j), max(i, j)
        jIndex = self.adjList[i].index(j)
        self.adjList[i][jIndex], self.adjList[i][-1] = (
                self.adjList[i][-1], self.adjList[i][jIndex])
        self.adjList[i].pop()
        self.numEdges -= 1

    def _removeEdge(self, i, j):
        jIndex = self.adjList[i].index(j)
        self.adjList[i][jIndex], self.adjList[i][-1] = (
                self.adjList[i][-1], self.adjList[i][jIndex])
        self.adjList[i].pop()
        self.numEdges -= 1

    def removeEdges(self, i):
        self.inBounds(i)
        for j in range(i):
            if self._hasEdge(j, i):
                self._removeEdge(j, i)
        self.numEdges -= len(self.adjList[i])
        self.adjList[i] = []

    def inBounds(self, i):
        if i < 0:
            raise IndexError("vertex '{}' must be non negative".format(i))
        if i >= self.numVertices:
            raise IndexError("vertex '{}' greater than graph size \
                '{}'".format(i, self.numVertices))

class DirectedGraph:

    def __init__(self, n):
        self.entries = [[] for _ in range(n)]
        self.exits = [[] for _ in range(n)]
        self.numEdges = 0

    @property
    def numVertices(self):
        return len(self.entries)

    def addEdge(self, i, j):
        if self.hasEdge(i, j):
            return None
        self.exits[i].append(j)
        self.entries[j].append(i)
        self.numEdges += 1

    def hasEdge(self, i, j):
        self.inBounds(i)
        self.inBounds(j)
        return i in self.entries[j] and j in self.exits[i]

    def _hasEdge(self, i, j):
        return i in self.entries[j]

    def addVertices(self, num=1):
        self.entries += [[] for _ in range(num)]
        self.exits += [[] for _ in range(num)]

    def inDegree(self, i):
        self.inBounds(i)
        return len(self.entries[i])

    def outDegree(self, i):
        self.inBounds(i)
        return len(self.exits[i])

    def kthInEdge(self, i, k):
        self.inBounds(i)
        if k >= self.inDegree(i):
            raise ValueError('vertex {} has too few edges'.format(i))
        return self.entries[i][k]

    def kthOutEdge(self, i, k):
        self.inBounds(i)
        if k >= self.outDegree(i):
            raise ValueError('vertex {} has too few edges'.format(i))
        return self.exits[i][k]

    def listInEdges(self, i):
        return self.entries[i]

    def listOutEdges(self, i):
        return self.exits[i]

    def listEdges(self, i): # searchPaths looks at OutEdges
        return self.listOutEdges(i)

    def _swapIndices(lst, i, j):
        lst[i], lst[j] = lst[j], lst[i]

    def removeEdge(self, i, j):
        if not self.hasEdge(i, j):
            raise ValueError('graph has no edge here')
        self._removeInEdge(i, j)
        self._removeOutEdge(i, j)
        self.numEdges -= 1

    def _removeInEdge(self, i, j):
        entryIndex = self.entries[j].index(i)
        self._swapIndices(self.entries[j], entryIndex, -1)
        self.entries[j].pop()

    def _removeOutEdge(self, i, j):
        exitIndex = self.exits[i].index(j)
        self._swapIndices(self.exits[i], exitIndex, -1)
        self.exits[i].pop()

    def removeOutEdges(self, i):
        self.inBounds(i)
        for j in self.exits[i]:
            self._removeInEdge(i, j)
        self.numEdges -= len(self.exits[i])
        self.exits[i] = []

    def removeInEdges(self, i):
        self.inBounds(i)
        for j in self.entries[i]:
            self._removeOutEdge(j, i)
        self.numEdges -= len(self.entries[i])
        self.entries[i] = []

    def inBounds(self, i):
        if i < 0:
            raise IndexError("vertex '{}' must be non negative".format(i))
        if i >= self.numVertices:
            raise IndexError("vertex '{}' greater than graph size \
                '{}'".format(i, self.numVertices))

class WeightedDirectedGraph:

    def __init__(self, n):
        self.entries = [[] for _ in range(n)]
        self.exits = [[] for _ in range(n)]
        self.numEdges = 0

    @property
    def numVertices(self):
        return len(self.entries)

    class Edge:

        def __init__(self, source, target, weight=1):
            self.source = source
            self.target = target
            self.weight = weight

    def addEdge(self, i, j, w=1):
        if self.hasEdge(i, j):
            return None
        E = WeightedDirectedGraph.Edge(i, j, w)
        self.exits[i].append(E)
        self.entries[j].append(E)
        self.numEdges += 1

    def hasEdge(self, i, j):
        self.inBounds(i)
        self.inBounds(j)
        for E in self.exits[i]:
            if E.target == j:
                return True
        for E in self.entries[j]:
            if E.source == i:
                return True
        return False

    def _hasEdge(self, i, j):
        for E in self.exits[i]:
            if E.target == j:
                return True
        for E in self.entries[j]:
            if E.source == i:
                return True
        return False

    def changeWeight(self, i, j, w):
        for E in self.entries[j]:
            if E.source == i:
                E.weight = w
                break

    def addVertices(self, num=1):
        self.entries += [[] for _ in range(num)]
        self.exits += [[] for _ in range(num)]

    def inDegree(self, i):
        self.inBounds(i)
        return len(self.entries[i])

    def outDegree(self, i):
        self.inBounds(i)
        return len(self.exits[i])

    def kthInEdge(self, i, k):
        self.inBounds(i)
        if k >= self.inDegree(i):
            raise ValueError('vertex {} has too few edges'.format(i))
        return self.entries[i][k]

    def kthOutEdge(self, i, k):
        self.inBounds(i)
        if k >= self.outDegree(i):
            raise ValueError('vertex {} has too few edges'.format(i))
        return self.exits[i][k]

    def listInEdges(self, i):
        return self.entries[i]

    def listOutEdges(self, i):
        return self.exits[i]

    def listEdges(self, i): # searchPaths looks at OutEdges targets
        return [E.target for E in self.listOutEdges(i)]

    def _swapIndices(lst, i, j):
        lst[i], lst[j] = lst[j], lst[i]

    def removeEdge(self, i, j):
        if not self.hasEdge(i, j):
            raise ValueError('graph has no edge here')
        self._removeInEdge(i, j)
        self._removeOutEdge(i, j)
        self.numEdges -= 1

    def _removeInEdge(self, i, j):
        for k in range(len(self.entries[j])):
            if self.entries[j][k].source == i:
                entryIndex = k
                break
        self._swapIndices(self.entries[j], entryIndex, -1)
        self.entries[j].pop()

    def _removeOutEdge(self, i, j):
        for k in range(len(self.exits[i])):
            if self.exits[i][k].target == j:
                exitIndex = k
                break
        self._swapIndices(self.exits[i], exitIndex, -1)
        self.exits[i].pop()

    def removeOutEdges(self, i):
        self.inBounds(i)
        for j in self.exits[i]:
            self._removeInEdge(i, j)
        self.numEdges -= len(self.exits[i])
        self.exits[i] = []

    def removeInEdges(self, i):
        self.inBounds(i)
        for j in self.entries[i]:
            self._removeOutEdge(j, i)
        self.numEdges -= len(self.entries[i])
        self.entries[i] = []

    def inBounds(self, i):
        if i < 0:
            raise IndexError("vertex '{}' must be non negative".format(i))
        if i >= self.numVertices:
            raise IndexError("vertex '{}' greater than graph size \
                '{}'".format(i, self.numVertices))



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

class WeightedQuickUnionDisjointSetWithPathCompression:

    def __init__(self, length):
        self.size = [None] * length
        self.parent = [None] * length
        for i in range(length):
            self.size[i] = 1
            self.parent[i] = i

    def find(self, i):
        if self.parent[i] == i:
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



###################
## Path Searches ##
###################

class DepthFirstPaths:

    def __init__(self, graph, source):
        self.graph = graph
        self.source = source
        self.marked = [False for _ in range(self.graph.numVertices)]
        self.edgeTo = [None for _ in range(self.graph.numVertices)]
        self.preOrder = []
        self.postOrder = []
        self.search(self.source)


    def mark(self, vertex):
        self.marked[vertex] = True

    def isMarked(self, vertex):
        return self.marked[vertex]

    def fillEdgeTo(self, a, v):
        self.edgeTo[a] = v


    def search(self, vertex):
        self.mark(vertex)
        self.preOrder.append(vertex)
        for adjVertex in self.graph.listEdges(vertex):
            if not self.isMarked(adjVertex):
                self.fillEdgeTo(adjVertex, vertex)
                self.search(adjVertex)
        self.postOrder.append(vertex)

    def hasPathTo(self, target):
        return self.isMarked(target)

    def pathTo(self, target):
        if not self.hasPathTo(target):
            return None
        path = Stack()
        while target != self.source:
            path.push(target)
            target = self.edgeTo[target]
        path.push(self.source)
        return list(path)


    def searchIterative(self, vertex):
        self.marked = [False for _ in range(self.graph.numVertices)]
        self.edgeTo = [None for _ in range(self.graph.numVertices)]
        # self.preOrder = []
        # self.postOrder = []
        callStack = Stack()
        callStack.push(vertex)
        self.mark(vertex)
        while not callStack.isEmpty():
            vertex = callStack.pop()
            # self.preOrder.append(vertex)
            self.pushUnmarkedAdj(vertex, callStack)

    def pushUnmarkedAdj(self, vertex, callStack):
        for v in self.graph.listEdges(vertex):
            if not self.isMarked(v):
                callStack.push(v)
                self.mark(v)
                self.fillEdgeTo(v, vertex)

class BreadthFirstPaths:

    def __init__(self, graph, source):
        self.graph = graph
        self.source = source
        self.marked = [False for _ in range(self.graph.numVertices)]
        self.edgeTo = [None for _ in range(self.graph.numVertices)]
        self.order = []
        self.search(self.source)

    def mark(self, vertex):
        self.marked[vertex] = True

    def isMarked(self, vertex):
        return self.marked[vertex]

    def fillEdgeTo(self, curr, prev):
        self.edgeTo[curr] = prev

    def search(self, vertex):
        callQueue = Queue()
        callQueue.enqueue(vertex)
        self.mark(vertex)
        while not callQueue.isEmpty():
            vertex = callQueue.dequeue()
            self.order.append(vertex)
            self.enqueueUnmarkedAdj(vertex, callQueue)

    def enqueueUnmarkedAdj(self, vertex, callQueue):
        for v in self.graph.listEdges(vertex):
            if not self.isMarked(v):
                callQueue.enqueue(v)
                self.mark(v)
                self.fillEdgeTo(v, vertex)

    def hasPathTo(self, target):
        return self.isMarked(target)

    def pathTo(self, target):
        if not self.hasPathTo(target):
            return None
        path = Stack()
        while target != self.source:
            path.push(target)
            target = self.edgeTo[target]
        path.push(self.source)
        return list(path)

class TopologicalSort:

    def __init__(self, graph):
        self.graph = graph
        self.sorts = []
        self.search()

    def listInDegreeZeroVertices(self):
        zeroLst = []
        for i in range(self.graph.numVertices):
            if self.graph.inDegree(i) == 0:
                zeroLst.append(i)
        return zeroLst

    def search(self):
        for i in self.listInDegreeZeroVertices():
            DFPi = DepthFirstPaths(self.graph, i)
            #if len(DFPi.postOrder) == self.graph.numVertices:
            revPostOrder = list(reversed(DFPi.postOrder))
            self.sorts.append(revPostOrder)

    def listSort(self):
        if not self.sorts:
            return None
        returnSort = self.sorts[0]
        for i in range(1, len(self.sorts)):
            if len(returnSort) == self.graph.numVertices:
                break
            for x in self.sorts[i]:
                if x not in returnSort:
                    returnSort.append(x)
        return returnSort

class ShortestPathsTree:

    from math import inf

    def __init__(self, graph, source):
        self.graph = graph
        self.source = source
        self.distTo = [ShortestPathsTree.inf for _ in
                       range(self.graph.numVertices)]
        self.edgeTo = [None for _ in range(self.graph.numVertices)]
        self.fringe = PriorityQueue()
        self.initFringe()
        self.search()

    def initFringe(self):
        self.fringe.add(self.source, 0)
        for vertex in range(self.source):
            self.fringe.add(vertex, ShortestPathsTree.inf)
        for vertex in range(self.source + 1, self.graph.numVertices):
            self.fringe.add(vertex, ShortestPathsTree.inf)
        self.distTo[self.source] = 0

    def search(self):
        while not self.fringe.isEmpty():
            vertex = self.fringe.removeSmallest()
            for edge in self.graph.listOutEdges(vertex):
                self.relax(edge, vertex)

    def relax(self, edge, vertex):
        newDist = self.distTo[vertex] + edge.weight
        if newDist < self.distTo[edge.target]:
            self.distTo[edge.target] = newDist
            self.edgeTo[edge.target] = vertex
            self.fringe.changePriority(edge.target, newDist)

    def pathTo(self, target):
        path = Stack()
        while target != self.source:
            path.push(target)
            target = self.edgeTo[target]
        path.push(self.source)
        return list(path)

class ShortestPathsTreeHeuristic:

    from math import inf

    def __init__(self, graph, source, target, heuristic):
        self.graph = graph
        self.source = source
        self.target = target
        self.heuristic = heuristic
        self.distTo = [ShortestPathsTreeWithTarget.inf
                       for _ in range(self.graph.numVertices)]
        self.edgeTo = [None for _ in range(self.graph.numVertices)]
        self.fringe = PriorityQueue()
        self.initFringe()
        self.search()

    def initFringe(self):
        self.fringe.add(self.source, 0)
        for vertex in range(self.source):
            self.fringe.add(vertex, ShortestPathsTreeWithTarget.inf)
        for vertex in range(self.source + 1, self.graph.numVertices):
            self.fringe.add(vertex, ShortestPathsTreeWithTarget.inf)
        self.distTo[self.source] = 0

    def search(self):
        while not self.fringe.isEmpty():
            vertex = self.fringe.removeSmallest()
            if vertex == self.target:
                break
            for edge in self.graph.listOutEdges(vertex):
                self.relax(edge, vertex)

    def relax(self, edge, vertex):
        newDist = self.distTo[vertex] + edge.weight
        if (newDist + self.heuristic(edge.target) <
            self.distTo[edge.target]):
            self.distTo[edge.target] = newDist
            self.edgeTo[edge.target] = vertex
            self.fringe.changePriority(edge.target, newDist)

    def pathTo(self):
        path = Stack()
        target = self.target
        while target != self.source:
            path.push(target)
            target = self.edgeTo[target]
        path.push(self.source)
        return list(path)



####################
## Spanning Trees ##
####################

class MinimumSpanningTreeKruskals:

    def __init__(self, graph):
        self.graph = graph
        self.fringe = PriorityQueue()
        self.treeDisjointSet = WeightedQuickUnionDisjointSetWithPathCompression(graph.numVertices)
        self.treeEdgeList = []
        self.initFringe()
        self.search()

    def initFringe(self):
        for E in self.graph.listAllEdges():
            self.fringe.add(E, E.weight)

    def search(self):
        while (not self.fringe.isEmpty() and
               len(self.treeEdgeList) < self.graph.numVertices - 1):
            E = self.fringe.removeSmallest()
            if not self.treeDisjointSet.isConnected(E.source, E.target):
                self.treeDisjointSet.connect(E.source, E.target)
                self.treeEdgeList.append(E)

class MinimumSpanningTreePrims:

    def __init__(self, graph):
        self.graph = graph
        self.fringe = PriorityQueue()
        self.marked = [False for _ in range(graph.numVertices)]
        self.treeEdgeList = []
        self.initFringe()
        self.search()

    def initFringe(self):
        self.marked[0] = True
        for E in self.graph.listAllEdges(0):
            self.fringe.add(E, E.weight)

    def search(self):
        while (not self.fringe.isEmpty() and
               True in self.marked):
            E = self.fringe.removeSmallest()
            self.relax(E)

    def relax(self, E):
        i = None
        if self.marked[E.source]:
            if not self.marked[E.target]:
                i = E.target
        else: #self.marked[E.target] == True
            i = E.source
        if i != None:
            self.treeEdgeList.append(E)
            self.marked[i] = True
            for E in self.graph.listAllEdges(i):
                self.fringe.add(E, E.weight)



#########################
## Dynamic Programming ##
#########################

class DirectedAcyclicGraphShortestPathsTree:

    from math import inf

    def __init__(self, graph):
        self.graph = graph
        self.distTo = [ShortestPathsTree.inf for _ in
                       range(self.graph.numVertices)]
        self.edgeTo = [None for _ in range(self.graph.numVertices)]
        self.fringe = []
        self.initFringe()
        self.search()

    def initFringe(self):
        TS = TopologicalSort(self.graph)
        self.fringe = TS.listSort()
        source = self.fringe[0]
        self.distTo[source] = 0

    def search(self):
        i = 0
        while i < len(self.fringe):
            vertex = self.fringe[i]
            for edge in self.graph.listOutEdges(vertex):
                self.relax(edge, vertex)
            i += 1

    def relax(self, edge, vertex):
        newDist = self.distTo[vertex] + edge.weight
        if newDist < self.distTo[edge.target]:
            self.distTo[edge.target] = newDist
            self.edgeTo[edge.target] = vertex

    def listTreeEdges(self):
        returnLst = []
        for t in range(1, self.graph.numVertices):
            s = self.edgeTo[t]
            w = self.distTo[t] - self.distTo[s]
            for edge in self.graph.listOutEdges(s):
                if edge.target == t and edge.weight == w:
                    returnLst.append(edge)
                    break
        return returnLst



###########
## Tests ##
###########

import unittest, random
class UndirectedeSearchPathsTests(unittest.TestCase):

    def testDFSPathTo(self):
        G = UndirectedGraph(10)
        edgeList = [(1,2), (1,3), (4,3), (5,3), (5,6),
                    (8,5), (5,7), (7,9), (8,6)]
        for i, j in edgeList:
            G.addEdge(i, j)
        DFP1 = DepthFirstPaths(G, 1)
        self.assertTrue(DFP1.hasPathTo(4))
        self.assertEqual(DFP1.pathTo(9), [1, 3, 5, 7, 9])
        DFP8 = DepthFirstPaths(G, 8)
        self.assertEqual(DFP8.pathTo(4), [8, 5, 3, 4])
        DFP0 = DepthFirstPaths(G, 0)
        for i in range(1, 10):
            self.assertFalse(DFP0.hasPathTo(i))
        DFP6 = DepthFirstPaths(G, 6)
        self.assertEqual(DFP6.pathTo(2), [6, 5, 3, 1, 2])

    def testDFSOrder(self):
        G = UndirectedGraph(10)
        edgeList = [(1,2), (1,3), (4,3), (5,3), (5,6),
                    (8,5), (5,7), (7,9), (8,6)]
        for i, j in edgeList:
            G.addEdge(i, j)
        DFP1 = DepthFirstPaths(G, 1)
        self.assertEqual(DFP1.preOrder, [1, 2, 3, 4, 5, 6, 8, 7, 9])
        self.assertEqual(DFP1.postOrder, [2, 4, 8, 6, 9, 7, 5, 3, 1])
        DFP5 = DepthFirstPaths(G, 5)
        self.assertEqual(DFP5.preOrder, [5, 3, 1, 2, 4, 6, 8, 7, 9])
        self.assertEqual(DFP5.postOrder, [2, 1, 4, 3, 8, 6, 9, 7, 5])

    def testDFSIterative(self):
        G = UndirectedGraph(10)
        edgeList = [(1,2), (1,3), (4,3), (5,3), (5,6),
                    (8,5), (5,7), (7,9), (8,6)]
        for i, j in edgeList:
            G.addEdge(i, j)
        DFP1 = DepthFirstPaths(G, 1)
        DFP1.searchIterative(1)
        self.assertTrue(DFP1.hasPathTo(4))
        self.assertEqual(DFP1.pathTo(9), [1, 3, 5, 7, 9])
        DFP8 = DepthFirstPaths(G, 8)
        DFP1.searchIterative(8)
        self.assertEqual(DFP8.pathTo(4), [8, 5, 3, 4])
        DFP0 = DepthFirstPaths(G, 0)
        DFP1.searchIterative(0)
        for i in range(1, 10):
            self.assertFalse(DFP0.hasPathTo(i))
        DFP6 = DepthFirstPaths(G, 6)
        DFP1.searchIterative(6)
        self.assertEqual(DFP6.pathTo(2), [6, 5, 3, 1, 2])

    # def testDFSIterativeOrder(self):
    #     G = UndirectedGraph(10)
    #     edgeList = [(1,2), (1,3), (4,3), (5,3), (5,6),
    #                 (8,5), (5,7), (7,9), (8,6)]
    #     for i, j in edgeList:
    #         G.addEdge(i, j)
    #     DFP1 = DepthFirstPaths(G, 1)
    #     DFP1.searchIterative(1)
    #     self.assertEqual(DFP1.preOrder, [1, 2, 3, 4, 5, 6, 8, 7, 9])
    #     self.assertEqual(DFP1.postOrder, [2, 4, 8, 6, 9, 7, 5, 3, 1])
    #     DFP5 = DepthFirstPaths(G, 5)
    #     DFP1.searchIterative(5)
    #     self.assertEqual(DFP5.preOrder, [5, 3, 1, 2, 4, 6, 8, 7, 9])
    #     self.assertEqual(DFP5.postOrder, [2, 1, 4, 3, 8, 6, 9, 7, 5])

    def testBFSPathTo(self):
        G = UndirectedGraph(10)
        edgeList = [(0, 1), (2, 1), (4, 5), (5, 2), (4, 3),
                    (8, 5), (5, 6), (6, 7), (1, 4)]
        for i, j in edgeList:
            G.addEdge(i, j)
        BFP0 = BreadthFirstPaths(G, 0)
        self.assertTrue(BFP0.hasPathTo(7))
        self.assertEqual(BFP0.pathTo(8), [0, 1, 2, 5, 8])
        BFP5 = BreadthFirstPaths(G, 5)
        self.assertEqual(BFP5.pathTo(0), [5, 2, 1, 0])
        BFP9 = BreadthFirstPaths(G, 9)
        for i in range(9):
            self.assertFalse(BFP9.hasPathTo(i))
        BFP7 = BreadthFirstPaths(G, 7)
        self.assertEqual(BFP7.pathTo(8), [7, 6, 5, 8])

    def testBFSOrder(self):
        G = UndirectedGraph(10)
        edgeList = [(0, 1), (2, 1), (4, 5), (5, 2), (4, 3),
                    (8, 5), (5, 6), (6, 7), (1, 4)]
        for i, j in edgeList:
            G.addEdge(i, j)
        BFP0 = BreadthFirstPaths(G, 0)
        self.assertEqual(BFP0.order, [0, 1, 2, 4, 5, 3, 8, 6, 7])
        BFP5 = BreadthFirstPaths(G, 5)
        self.assertEqual(BFP5.order, [5, 2, 4, 8, 6, 1, 3, 7, 0])
        BFP9 = BreadthFirstPaths(G, 9)
        self.assertEqual(BFP9.order, [9])

class DirectedeSearchPathsTests(unittest.TestCase):

    def testDFSPathTo(self):
        G = DirectedGraph(10)
        edgeList = [(1,2), (1,3), (4,3), (5,3), (5,6),
                    (8,5), (5,7), (7,9), (8,6)]
        for i, j in edgeList:
            G.addEdge(i, j)
            G.addEdge(j, i)
        DFP1 = DepthFirstPaths(G, 1)
        self.assertTrue(DFP1.hasPathTo(4))
        self.assertEqual(DFP1.pathTo(9), [1, 3, 5, 7, 9])
        DFP8 = DepthFirstPaths(G, 8)
        self.assertEqual(DFP8.pathTo(4), [8, 5, 3, 4])
        DFP0 = DepthFirstPaths(G, 0)
        for i in range(1, 10):
            self.assertFalse(DFP0.hasPathTo(i))
        DFP6 = DepthFirstPaths(G, 6)
        self.assertEqual(DFP6.pathTo(2), [6, 5, 3, 1, 2])

    def testDFSOrder(self):
        G = DirectedGraph(10)
        edgeList = [(1,2), (1,3), (4,3), (5,3), (5,6),
                    (8,5), (5,7), (7,9), (8,6)]
        for i, j in edgeList:
            G.addEdge(i, j)
            G.addEdge(j, i)
        DFP1 = DepthFirstPaths(G, 1)
        self.assertEqual(DFP1.preOrder, [1, 2, 3, 4, 5, 6, 8, 7, 9])
        self.assertEqual(DFP1.postOrder, [2, 4, 8, 6, 9, 7, 5, 3, 1])
        DFP5 = DepthFirstPaths(G, 5)
        self.assertEqual(DFP5.preOrder, [5, 3, 1, 2, 4, 6, 8, 7, 9])
        self.assertEqual(DFP5.postOrder, [2, 1, 4, 3, 8, 6, 9, 7, 5])

    def testDFSIterative(self):
        G = DirectedGraph(10)
        edgeList = [(1,2), (1,3), (4,3), (5,3), (5,6),
                    (8,5), (5,7), (7,9), (8,6)]
        for i, j in edgeList:
            G.addEdge(i, j)
            G.addEdge(j, i)
        DFP1 = DepthFirstPaths(G, 1)
        DFP1.searchIterative(1)
        self.assertTrue(DFP1.hasPathTo(4))
        self.assertEqual(DFP1.pathTo(9), [1, 3, 5, 7, 9])
        DFP8 = DepthFirstPaths(G, 8)
        DFP1.searchIterative(8)
        self.assertEqual(DFP8.pathTo(4), [8, 5, 3, 4])
        DFP0 = DepthFirstPaths(G, 0)
        DFP1.searchIterative(0)
        for i in range(1, 10):
            self.assertFalse(DFP0.hasPathTo(i))
        DFP6 = DepthFirstPaths(G, 6)
        DFP1.searchIterative(6)
        self.assertEqual(DFP6.pathTo(2), [6, 5, 3, 1, 2])


    def testBFSPathTo(self):
        G = DirectedGraph(10)
        edgeList = [(0, 1), (2, 1), (4, 5), (5, 2), (4, 3),
                    (8, 5), (5, 6), (6, 7), (1, 4)]
        for i, j in edgeList:
            G.addEdge(i, j)
            G.addEdge(j, i)
        BFP0 = BreadthFirstPaths(G, 0)
        self.assertTrue(BFP0.hasPathTo(7))
        self.assertEqual(BFP0.pathTo(8), [0, 1, 2, 5, 8])
        BFP5 = BreadthFirstPaths(G, 5)
        self.assertEqual(BFP5.pathTo(0), [5, 4, 1, 0])
        BFP9 = BreadthFirstPaths(G, 9)
        for i in range(9):
            self.assertFalse(BFP9.hasPathTo(i))
        BFP7 = BreadthFirstPaths(G, 7)
        self.assertEqual(BFP7.pathTo(8), [7, 6, 5, 8])

    def testBFSOrder(self):
        G = DirectedGraph(10)
        edgeList = [(0, 1), (2, 1), (4, 5), (5, 2), (4, 3),
                    (8, 5), (5, 6), (6, 7), (1, 4)]
        for i, j in edgeList:
            G.addEdge(i, j)
            G.addEdge(j, i)
        BFP0 = BreadthFirstPaths(G, 0)
        self.assertEqual(BFP0.order, [0, 1, 2, 4, 5, 3, 8, 6, 7])
        BFP5 = BreadthFirstPaths(G, 5)
        self.assertEqual(BFP5.order, [5, 4, 2, 8, 6, 3, 1, 7, 0])
        BFP9 = BreadthFirstPaths(G, 9)
        self.assertEqual(BFP9.order, [9])

class WeightedDirectedeSearchPathsTests(unittest.TestCase):

    def testDFSPathTo(self):
        G = WeightedDirectedGraph(10)
        edgeList = [(1,2), (1,3), (4,3), (5,3), (5,6),
                    (8,5), (5,7), (7,9), (8,6)]
        for i, j in edgeList:
            G.addEdge(i, j, random.randint(1, 1000))
            G.addEdge(j, i, random.randint(1, 1000))
        DFP1 = DepthFirstPaths(G, 1)
        self.assertTrue(DFP1.hasPathTo(4))
        self.assertEqual(DFP1.pathTo(9), [1, 3, 5, 7, 9])
        DFP8 = DepthFirstPaths(G, 8)
        self.assertEqual(DFP8.pathTo(4), [8, 5, 3, 4])
        DFP0 = DepthFirstPaths(G, 0)
        for i in range(1, 10):
            self.assertFalse(DFP0.hasPathTo(i))
        DFP6 = DepthFirstPaths(G, 6)
        self.assertEqual(DFP6.pathTo(2), [6, 5, 3, 1, 2])

    def testDFSOrder(self):
        G = WeightedDirectedGraph(10)
        edgeList = [(1,2), (1,3), (4,3), (5,3), (5,6),
                    (8,5), (5,7), (7,9), (8,6)]
        for i, j in edgeList:
            G.addEdge(i, j, random.randint(1, 1000))
            G.addEdge(j, i, random.randint(1, 1000))
        DFP1 = DepthFirstPaths(G, 1)
        self.assertEqual(DFP1.preOrder, [1, 2, 3, 4, 5, 6, 8, 7, 9])
        self.assertEqual(DFP1.postOrder, [2, 4, 8, 6, 9, 7, 5, 3, 1])
        DFP5 = DepthFirstPaths(G, 5)
        self.assertEqual(DFP5.preOrder, [5, 3, 1, 2, 4, 6, 8, 7, 9])
        self.assertEqual(DFP5.postOrder, [2, 1, 4, 3, 8, 6, 9, 7, 5])

    def testDFSIterative(self):
        G = WeightedDirectedGraph(10)
        edgeList = [(1,2), (1,3), (4,3), (5,3), (5,6),
                    (8,5), (5,7), (7,9), (8,6)]
        for i, j in edgeList:
            G.addEdge(i, j, random.randint(1, 1000))
            G.addEdge(j, i, random.randint(1, 1000))
        DFP1 = DepthFirstPaths(G, 1)
        DFP1.searchIterative(1)
        self.assertTrue(DFP1.hasPathTo(4))
        self.assertEqual(DFP1.pathTo(9), [1, 3, 5, 7, 9])
        DFP8 = DepthFirstPaths(G, 8)
        DFP1.searchIterative(8)
        self.assertEqual(DFP8.pathTo(4), [8, 5, 3, 4])
        DFP0 = DepthFirstPaths(G, 0)
        DFP1.searchIterative(0)
        for i in range(1, 10):
            self.assertFalse(DFP0.hasPathTo(i))
        DFP6 = DepthFirstPaths(G, 6)
        DFP1.searchIterative(6)
        self.assertEqual(DFP6.pathTo(2), [6, 5, 3, 1, 2])


    def testBFSPathTo(self):
        G = WeightedDirectedGraph(10)
        edgeList = [(0, 1), (2, 1), (4, 5), (5, 2), (4, 3),
                    (8, 5), (5, 6), (6, 7), (1, 4)]
        for i, j in edgeList:
            G.addEdge(i, j, random.randint(1, 1000))
            G.addEdge(j, i, random.randint(1, 1000))
        BFP0 = BreadthFirstPaths(G, 0)
        self.assertTrue(BFP0.hasPathTo(7))
        self.assertEqual(BFP0.pathTo(8), [0, 1, 2, 5, 8])
        BFP5 = BreadthFirstPaths(G, 5)
        self.assertEqual(BFP5.pathTo(0), [5, 4, 1, 0])
        BFP9 = BreadthFirstPaths(G, 9)
        for i in range(9):
            self.assertFalse(BFP9.hasPathTo(i))
        BFP7 = BreadthFirstPaths(G, 7)
        self.assertEqual(BFP7.pathTo(8), [7, 6, 5, 8])

    def testBFSOrder(self):
        G = WeightedDirectedGraph(10)
        edgeList = [(0, 1), (2, 1), (4, 5), (5, 2), (4, 3),
                    (8, 5), (5, 6), (6, 7), (1, 4)]
        for i, j in edgeList:
            G.addEdge(i, j, random.randint(1, 1000))
            G.addEdge(j, i, random.randint(1, 1000))
        BFP0 = BreadthFirstPaths(G, 0)
        self.assertEqual(BFP0.order, [0, 1, 2, 4, 5, 3, 8, 6, 7])
        BFP5 = BreadthFirstPaths(G, 5)
        self.assertEqual(BFP5.order, [5, 4, 2, 8, 6, 3, 1, 7, 0])
        BFP9 = BreadthFirstPaths(G, 9)
        self.assertEqual(BFP9.order, [9])

class TopSort(unittest.TestCase):

    def test(self):
        G = DirectedGraph(8)
        edgeList = [(2,3), (2,5), (0,1), (0,3),
                    (4,7), (5,6), (3,4), (1,4), (5,4)]
        for i, j in edgeList:
            G.addEdge(i, j)
        TS = TopologicalSort(G)
        self.assertEqual(TS.listInDegreeZeroVertices(), [0, 2])
        self.assertEqual(TS.listSort(), [0, 3, 1, 4, 7, 2, 5, 6])

        G = WeightedDirectedGraph(8)
        edgeList = [(2,3), (2,5), (0,1), (0,3),
                    (4,7), (5,6), (3,4), (1,4), (5,4)]
        for i, j in edgeList:
            G.addEdge(i, j)
        TS = TopologicalSort(G)
        self.assertEqual(TS.listInDegreeZeroVertices(), [0, 2])
        self.assertEqual(TS.listSort(), [0, 3, 1, 4, 7, 2, 5, 6])

class ShortestPaths(unittest.TestCase):

    def test(self):
        G = WeightedDirectedGraph(7)
        edgeList = [(0,1, 2), (0,2, 1), (1,2, 5), (1,3,11),
                    (1,4, 3), (2,5,15), (3,4, 2), (4,2, 1),
                    (4,5, 4), (4,6, 5), (6,5, 1), (6,3, 1)]
        for i, j, w in edgeList:
            G.addEdge(i, j, w)
        SPT = ShortestPathsTree(G, 0)
        self.assertEqual(SPT.distTo, [0, 2, 1, 11, 5, 9, 10])
        self.assertEqual(SPT.edgeTo, [None, 0, 0, 6, 1, 4, 4])
        self.assertEqual(SPT.pathTo(2), [0, 2])
        self.assertEqual(SPT.pathTo(5), [0, 1, 4, 5])
        self.assertEqual(SPT.pathTo(3), [0, 1, 4, 6, 3])

class MinimumSpanningTreeKruskalsTest(unittest.TestCase):

    def test(self):
        G = WeightedUndirectedGraph(7)
        edgeList = [(0,1, 2), (0,2, 1), (1,2, 5), (1,3,11),
                    (1,4, 3), (2,5,15), (3,4, 3), (4,2, 1),
                    (4,5, 4), (4,6, 3), (6,5, 1), (6,3, 1)]
        for i, j, w in edgeList:
            G.addEdge(i, j, w)
        MST = MinimumSpanningTreeKruskals(G)
        MSTSet = set([(E.source, E.target, E.weight)
                      for E in MST.treeEdgeList])
        EqualSet = set([(0,1, 2), (0,2, 1), (2,4, 1),
                       (3,4, 3), (3,6, 1), (5,6, 1)])
        self.assertEqual(MSTSet, EqualSet)

class MinimumSpanningTreePrimsTest(unittest.TestCase):

    def test(self):
        G = WeightedUndirectedGraph(7)
        edgeList = [(0,1, 2), (0,2, 1), (1,2, 5), (1,3,11),
                    (1,4, 3), (2,5,15), (3,4, 2), (4,2, 1),
                    (4,5, 4), (4,6, 3), (6,5, 1), (6,3, 1)]
        for i, j, w in edgeList:
            G.addEdge(i, j, w)
        MST = MinimumSpanningTreePrims(G)
        MSTSet = set([(E.source, E.target, E.weight)
                      for E in MST.treeEdgeList])
        EqualSet = set([(0,1, 2), (0,2, 1), (2,4, 1),
                       (3,4, 2), (3,6, 1), (5,6, 1)])
        # print('MST', MSTSet)
        self.assertEqual(MSTSet, EqualSet)

class DAGSPTTest(unittest.TestCase):

    def test(self):
        G = WeightedDirectedGraph(6)
        edgeLst = [(0,3, 2), (0,1, 2), (3,4, 3), (3,1, 4),
                    (1,2, 6), (2,4, 1), (2,5, 1), (4,5, 1)]
        for i, j, w in edgeLst:
            G.addEdge(i, j, w)
        DAGSPT = DirectedAcyclicGraphShortestPathsTree(G)
        DAGSPTLst = DAGSPT.listTreeEdges()
        DAGSPTSet = set([(E.source, E.target, E.weight)
                      for E in DAGSPTLst])
        EqualSet = set([(0,3, 2), (0,1, 2), (3,4, 3),
                        (1,2, 6), (4,5, 1)])
        self.assertEqual(DAGSPTSet, EqualSet)

# class DepthFirstDirectedPathsExperiment(unittest.TestCase):
#
#     import random
#
#     #Probability DFS finds path btw 2 rand vert, avg path found
#     def testAvgPathLen(self):
#         numGraphs = 10
#         minVertices = 1
#         maxVertices = 20
#         numEdgesFactor = 1.5
#         for _ in range(10):
#             numVertices = random.randint(minVertices, maxVertices)
#             G = DirectedGraph(numVertices)
#             numEdges = int(numEdgesFactor * numVertices)
#             i, j = random.randint(0, numEdges)


##########
## Main ##
##########

if __name__ == '__main__':

    unittest.main()
