class Node:

    def __init__(self, item, prev=None, next=None):
        self.prev = prev
        self.item = item
        self.next = next

    def __repr__(self):
        s = ' <- {0} -> '.format(self.item)
        if self.prev:
            s = str(self.prev.item) + s
        if self.next:
            s = s + str(self.next.item)
        return s

class DLList:

    def __init__(self):
        self.sentinel = Node(-1.1)
        self.sentinel.next = self.sentinel
        self.sentinel.prev = self.sentinel
        self.size = 0

    def __iter__(self):

        class nodeItemIterator:

            def __init__(self, sentinel, size):
                self.node = sentinel
                self.size = size
                self.position = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.position == self.size:
                    raise StopIteration
                else:
                    self.node = self.node.next
                    returnVal = self.node.item
                    self.position += 1
                    return returnVal

        return nodeItemIterator(self.sentinel, self.size)

    def isEmpty(self):
        return (self.sentinel == self.sentinel.prev and
                self.sentinel == self.sentinel.next)


    def addFirst(self, x):
        self.sentinel.next = Node(x, self.sentinel, self.sentinel.next)
        self.sentinel.next.next.prev = self.sentinel.next
        self.size += 1

    def getFirst(self):
        return self.sentinel.next.item

    def removeFirst(self):
        self.sentinel.next = self.sentinel.next.next
        self.sentinel.next.prev = self.sentinel
        self.size -= 1


    def addLast(self, x):
        self.sentinel.prev = Node(x, self.sentinel.prev, self.sentinel)
        self.sentinel.prev.prev.next = self.sentinel.prev
        self.size += 1

    def getLast(self):
        return self.sentinel.prev.item

    def removeLast(self):
        self.sentinel.prev = self.sentinel.prev.prev
        self.sentinel.prev.next = self.sentinel
        self.size -= 1


    def insert(self, x, i): # i >= 0
        node = self.sentinel
        while i > 0:
            node = node.next
            i -= 1
        node.next = Node(x, node.prev, node.next)
        node.next.next.prev = node.next
        self.size -= 1

    def get(self, i): # 0 is the first item
        if i >= self.size or self.size == 0 or i < -1 * self.size:
            return None
        node = self.sentinel
        if i >= 0:
            while i >= 0:
                node = node.next
                i -= 1
            return node.item
        if i < 0:
            while i < 0:
                node = node.prev
                i += 1
            return node.item

    def remove(self, i): # could optimize for approaching node from front or back
        node = self.sentinel
        while i > 0:
            node = node.next
            i -= 1
        node.next = node.next.next
        node.next.prev = node
        self.size -= 1



    def printDLList(self):
        node = self.sentinel
        s = ''
        while node.next is not self.sentinel:
            node = node.next
            s += str(node.item) + ' '
        print(s)


    def __repr__(self):
        node = self.sentinel
        str = ''
        while node.next is not self.sentinel:
            str += ' <=> {}'.format(node.next.item)
            node = node.next
        return str[5 : ]

class LinkedListDeque(DLList):

    def __init__(self):
        DLList.__init__(self)

    def addFirst(self, x):
        return DLList.addFirst(self, x)

    def addLast(self, x):
        return DLList.addLast(self, x)

    def isEmpty(self):
        return DLList.isEmpty(self)

    def printDeque(self):
        return DLList.printDLList(self)

    def removeFirst(self):
        return DLList.removeFirst(self)

    def removeLast(self):
        return DLList.removeLast(self)

    def get(self, index):
        return DLList.get(self, index)

    def getRecursive(self, index):
        if index >= self.size or self.size == 0 or index < 0:
            return None
        node = self.sentinel.next
        def helper(node, index):
            if index == 0:
                return node.item
            return helper(node.next, index - 1)
        return helper(node, index)
