
class MinPQ:
    # Allowing tracking and removal of the smallest item
    # in a priority queue.
    # Two rules: minimal height and left-leaning,
    #            children are larger

    def __init__(self):
        self.array = ['-'] + [None] * 8
        self.size = 0
    # index:  0 1 2 3 4 5 6 7 8
    # values: - a b c d e f g h
    # parents:0 0 1 1 2 2 3 3 4
        # last item is self.array[self.size]

    def resize(self, capacity):
        """ Resizes underlying array to target capacity."""
        newArray = ['-'] + [None] * capacity
        newArray[1 : self.size + 1] = self.array[1 : self.size + 1]
        self.array = newArray

    def checkResize(self):
        """ Resizes if necessary: items/capacity = 1 or <.25"""
        if self.size == len(self.array) - 1:
            newCapacity = self.size * 2
            self.resize(newCapacity)
        elif (self.size <= (len(self.array) - 1) // 4 and
              len(self.array) - 1 >= 16):
            newCapacity = (len(self.array) - 1) // 2
            self.resize(newCapacity)

    def add(self, item):
        """ Adds the item to the priority queue."""
        # place item to maintain structure i.e. at end of array
        # promote as needed
        # assert item is comparable
        index = self.size + 1
        self.array[index] = item
        self.addSwim(index)
        self.size += 1
        self.checkResize()

    def addSwim(self, index):
        parent = index // 2
        if parent and self.array[parent] > self.array[index]:
            self.array[parent], self.array[index] = (
                self.array[index], self.array[parent])
            self.addSwim(parent)

    def getSmallest(self):
        """ Returns the smallest item in the priority queue."""
        return self.array[1]

    def removeSmallest(self):
        """ Removes the smallest item in the priority queue."""
        # place last item at root
        # demote as needed, pick strongest child
        returnVal = self.array[1]
        self.array[1] = self.array[self.size]
        self.array[self.size] = None
        self.size -= 1
        self.checkResize()
        self.removeSwim(1)
        return returnVal

    def removeSwim(self, index):
        child = self.smallestChild(index)
        if child and self.array[child] < self.array[index]:
            self.array[child], self.array[index] = (
                self.array[index], self.array[child])
            self.removeSwim(child)

    def smallestChild(self, index):
        child1 = index * 2
        if child1 == self.size:
            return child1
        if child1 > self.size:
            return None
        child2 = child1 + 1
        smallerChild = min([child1, child2],
                          key=lambda x: self.array[x])
        return smallerChild

    def max(self):
        return max(self.array[self.size // 2 + 1 : self.size + 1])
