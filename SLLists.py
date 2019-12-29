class Node:

    def __init__(self, item, next=None):
        assert next is None or isinstance(next, Node)
        self.item = item
        self.next = next

class SLList:

    def __init__(self):
        self.sentinel = Node(-1.1)
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
                    returnItem = self.node.item
                    self.position += 1
                    return returnItem

        return nodeItemIterator(self.sentinel, self.size)

    def isEmpty(self):
        return self.sentinel.next is None


    def addFirst(self, item):
        self.sentinel.next = Node(item, self.sentinel.next)
        self.size += 1

    def getFirst(self):
        return self.sentinel.next.item

    def removeFirst(self):
        node = self.sentinel
        first = node.next
        node.next = node.next.next
        first.next = None
        self.size -= 1
        return first.item


    def getLastNode(self):
        node = self.sentinel
        while node.next:
            node = node.next
        return node

    def addLast(self, item):
        self.getLastNode().next = Node(item)
        self.size += 1

    def getLast(self):
        return self.getLastNode().item

    def removeLast(self):
        node = self.sentinel
        while node.next.next:
            node = node.next
        last = node.next
        node.next = None
        self.size -= 1
        return last.item


    def get(self, index):
        node = self.sentinel.next
        while index > 0:
            node = node.next
            index -= 1
        return node.item

    def insert(self, item, index):
        node = self.sentinel
        while index > 0:
            node = node.next
            index -= 1
        node.next = Node(item, node.next)
        self.size += 1

    def remove(self, index):
        node = self.sentinel
        while index > 0:
            node = node.next
            index -= 1
        indexedNode = node.next
        node.next = node.next.next
        indexedNode.next = None
        self.size -= 1
        return indexedNode.item

    def printList(self): # can be done with iter now
        node = self.sentinel.next
        result = ''
        while node:
            result += str(node.item) + ' '
            node = node.next
        print(result.strip())


    def reverse(self):
        """ Reverse the list in place.  Does not make new Nodes.
        >>> L = SLList()
        >>> L.addFirst(1)
        >>> L.addFirst(0)
        >>> L.addLast(2)
        >>> L.printList()
        0 1 2
        >>> L.reverse()
        >>> L.printList()
        2 1 0
        """
        before = None
        after = self.sentinel.next
        while after:
            before, after.next, after = (
            after, before, after.next)
        self.sentinel.next = before

    def reverseRecursive(self):
        """ Reverse the list in place.  Does not make new Nodes.
        >>> L = SLList()
        >>> L.addFirst(1)
        >>> L.addFirst(0)
        >>> L.addLast(2)
        >>> L.printList()
        0 1 2
        >>> L.reverseRecursive()
        >>> L.printList()
        2 1 0
        """
# sentinel -> node0    node1 -> node2
#   None   <- node0 <- node1
#
        node = self.sentinel.next
        def helper(node):
            if not node.next:
                return node
            else:
                before = node.next
                node.next = None
                after = helper(before)
                before.next = node
                return after
        self.sentinel.next = helper(node)

    def extendArray(self, arr):
        for x in arr:
            self.addLast(x)

    def __repr__(self): # can be done with iter now
        node = self.sentinel
        str = ''
        while node.next:
            str += ' -> {}'.format(node.next.item)
            node = node.next
        return str[4 : ]

class Stack(SLList):

    def __init__(self):
        SLList.__init__(self)
        self.size = 0

    def push(self, item):
        SLList.addFirst(self, item)

    def pop(self):
        return SLList.removeFirst(self)

#class GrabBag():

#    def __init__()

if __name__ == '__main__':
    import doctest
    doctest.testmod()
