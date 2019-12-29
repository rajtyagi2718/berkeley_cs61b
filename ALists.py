class AList:

    def __init__(self):
        """Creates an empty list."""
        self.items = [None] * 2
        self.size = 0 # number of items in the list

    def __iter__(self):

        class itemsIterator:

            def __init__(self, items, size):
                self.items = items
                self.size = size
                self.next = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.next == self.size:
                    raise StopIteration
                else:
                    returnVal = self.items[self.next]
                    self.next += 1
                    return returnVal

        return itemsIterator(self.items, self.size)

    def isEmpty(self):
        """Returns True if there are no items in list.

        >>> L = AList()
        >>> L.size
        0
        """
        return not self.size

    def resize(self, capacity):
        """Resizes the underlying array to the target capacity."""
        new_items = [None] * (capacity)
        new_items[0 : self.size] = self.items[0 : self.size]
        self.items = new_items

    def addLast(self, x):
        """Inserts x into the back of the list.

        >>> L = AList()
        >>> L.addLast(99)
        >>> L.addLast(99)
        >>> L.size
        2
        """
        if self.size == len(self.items):
            self.resize(2 * self.size)
        self.items[self.size] = x
        self.size += 1

    def getLast(self):
        """Returns the item from the back of the list.

        >>> L = AList()
        >>> L.addLast(99)
        >>> L.getLast()
        99
        >>> L.addLast(36)
        >>> L.getLast()
        36
        """
        return self.items[self.size - 1]

    def removeLast(self):
        """Deletes item from back of the list and
           returns deleted item.

        >>> L = AList()
        >>> L.addLast(99)
        >>> L.get(0)
        99
        >>> L.addLast(36)
        >>> L.get(0)
        99
        >>> L.removeLast()
        36
        >>> L.getLast()
        99
        >>> L.addLast(100)
        >>> L.getLast()
        100
        >>> L.size
        2
        """
        x = self.getLast()
        self.items[self.size - 1] = None
        self.size -= 1
        if self.size <= len(self.items) // 4:
            self.resize(len(self.items) // 2)
        return x

    def get(self, i):
        """Gets the ith item in the list (0 is the front).

        >>> L = AList()
        >>> L.addLast(99)
        >>> L.get(0)
        99
        >>> L.addLast(36)
        >>> L.get(0)
        99
        >>> L.get(1)
        36
        """
        return self.items[i]

    def insert(self, x, i):
        """Inserts item x into the ith position in the list.

        >>> L = AList()
        >>> L.insert(1, 0)
        >>> L.insert(0, 0)
        >>> L.get(0)
        0
        >>> L.get(1)
        1
        >>> L.insert(2, 2)
        >>> L.get(2)
        2
        """
        if self.size == len(self.items):
            new_items = [None] * (2 * self.size)
            new_items[0 : i] = self.items[0 : i]
            new_items[i] = x
            new_items[i+1 : self.size + 1] = self.items[i : self.size]
            self.items = new_items
        else:
            for j in range(self.size - 1, i-1, -1):
                self.items[j+1] = self.items[j]
            self.items[i] = x
        self.size += 1

    def remove(self, i):
        """Removes the ith item of the list.

        >>> L = AList()
        >>> L.insert(1, 0)
        >>> L.insert(0, 0)
        >>> L.remove(0)
        0
        >>> L.get(0)
        1
        >>> L.insert(2, 1)
        >>> L.remove(1)
        2
        """
        x = self.get(i)
        if self.size - 1 <= len(self.items) // 4:
            new_items = [None] * (len(self.items) // 2)
            new_items[0 : i] = self.items[0 : i]
            new_items[i : self.size - 1] = self.items[i+1 : self.size]
            self.items = new_items
        else:
            for j in range(i, self.size - 1):
                self.items[j] = self.items[j+1]
            self.items[self.size - 1] = None
        self.size -= 1
        return x

    def printList(self):
        """Prints the list with items separated by spaces.
        >>> L = AList()
        >>> L.printList()
        <BLANKLINE>
        >>> L.addLast(0)
        >>> L.printList()
        0
        >>> L.addLast(1)
        >>> L.printList()
        0 1
        """
        result = ''
        for i in range(self.size):
            result += '{} '.format(self.items[i])
        print(result.strip())

    def reverse(self):
        """Reverse list in place.
        >>> L = AList()
        >>> L.reverse()
        >>> L.printList()
        <BLANKLINE>
        >>> L.addLast(0)
        >>> L.reverse()
        >>> L.printList()
        0
        >>> L.addLast(1)
        >>> L.reverse()
        >>> L.printList()
        1 0
        """
        for i in range(0, self.size // 2):
            self.items[i], self.items[self.size - i - 1] = (
                self.items[self.size - i - 1], self.items[i])

    def extendArray(self, arr):
        """Extend the list by items given by array arr.
        >>> L = AList()
        >>> L.extendArray([0, 1, 2, 3, 4])
        >>> L.printList()
        0 1 2 3 4
        >>> L.extendArray([5])
        >>> L.printList()
        0 1 2 3 4 5
        """
        for x in arr:
            self.addLast(x)

def replicate(alst):
    """Non-destructive, returns a copy of alst
       with the number at index i replaced
       with Alst[i] copies of itself.
    >>> L = AList()
    >>> L.extendArray([0, 1, 0, 3, 2])
    >>> repL = replicate(L)
    >>> repL.printList()
    1 3 3 3 2 2
    """
    L = AList()
    for i in range(alst.size):
        x = alst.get(i)
        for _ in range(x):
            L.addLast(x)
    return L

# 0 1 2 3   4 5 6 7 8 9 10 11   12 13 14 15
# 0 1 2 3 4 5 6 7
#       x x

class ArrayDeque: #if self.first == 0 condition, cleaner way?

    def __init__(self):
        self.items = [None] * 8
        self.first = 0
        self.last = 0
        self.size = 0

    def isEmpty(self):
        return self.size == 0

    def resize(self, capacity):
        newItems = [None] * capacity
        newItems[0 : self.last + 1] = (
            self.items[0 : self.last + 1])
        if self.first > self.last: # [x x L F x x x x]
            newFirst = capacity - (self.size - self.first)
            newItems[newFirst : capacity] = (
                self.items[self.first : self.size])
            self.first = newFirst
        #else: diff == self.size - 1   [F x x x x x x L]
        self.items = newItems

    def addFirst(self, x):
        if self.size == 0:
            self.items[self.first] = x
        else:
            m = (self.first - 1) % len(self.items)
            self.items[m] = x
            self.first = m
        self.size += 1
        if self.size == len(self.items):
            self.resize(len(self.items) * 2)

    def addLast(self, x):
        if self.size == 0:
            self.items[self.last] = x
        else:
            m = (self.last + 1) % len(self.items)
            self.items[m] = x
            self.last = m
        self.size += 1
        if self.size == len(self.items):
            self.resize(len(self.items) * 2)

    def printDeque(self):
        print(list(self))

    def removeFirst(self):
        if self.isEmpty():
            raise IndexError('Cannot remove from empty deque.')
        x = self.items[self.first]
        self.items[self.first] = None
        self.first = (self.first + 1) % len(self.items)
        self.size -= 1
        if (self.size <= len(self.items) // 4 and
            len(self.items) > 8):
            self.resize(len(self.items) // 2)
        return x

    def removeLast(self):
        if self.isEmpty():
            raise IndexError('Cannot remove from empty deque.')
        x = self.items[self.last]
        self.items[self.last] = None
        self.last = (self.last - 1) % len(self.items)
        self.size -= 1
        if (self.size <= len(self.items) // 4 and
            len(self.items) > 8):
            self.resize(len(self.items) // 2)
        return x

    def get(self, i):
        if i >= self.size or i < -self.size:
            raise IndexError('Index magnitude larger than deque size.')
        else:
            if i >= 0:
                m = (self.first + i) % len(self.items)
            if i < 0:
                m = (self.last + i + 1) % len(self.items)
            return self.items[m]

    def __iter__(self):

        class itemIterator:

            def __init__(self, items, first, size):
                self.items = items
                self.first = first
                self.size = size
                self.position = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.position == self.size:
                    raise StopIteration
                m = (self.first + self.position) % len(self.items)
                returnItem = self.items[m]
                self.position += 1
                return returnItem

        return itemIterator(self.items, self.first, self.size)



import unittest
class InsertionsTest(unittest.TestCase):

    def test1(self):
        L = AList()
        n = 10 ** 6
        for i in range(n):
            L.addLast(i)
        for i in range(n):
            L.addLast(L.get(i))


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    unittest.main()
