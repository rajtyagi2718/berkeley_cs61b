
class ArrayMap:

    def __init__(self):
        self.keys = [None] * 8
        self.values = [None] * 8
        self.size = 0

    def __iter__(self):

        class KeyIterator:

            def __init__(self, keys, size):
                self.keys = keys
                self.size = size
                self.next = 0

            def __next__(self):
                if self.next == self.size:
                    raise StopIteration
                else:
                    returnVal = self.keys[self.next]
                    self.next += 1
                    return returnVal

            def __iter__(self):
                return self

        return KeyIterator(self.keys, self.size)

    # def __iter__(self): # self.keys contains None items too
    #     return iter(self.keys)

    # def __iter__(self):
    #     self.next = 0
    #     return self
    #
    # def __next__(self):
    #     if self.next == self.size:
    #         raise StopIteration
    #     else:
    #         returnVal = self.keys[self.next]
    #         self.next += 1
    #         return returnVal

    def resize(self, capacity):
        newKeys = [None] * capacity
        newValues = [None] * capacity
        newKeys[0 : self.size] = self.keys[0 : self.size]
        newValues[0 : self.size] = self.values[0 : self.size]
        self.keys = newKeys
        self.values = newValues

    def keyIndex(self, key):
        """ Returns the index of the given key if it exists,
            -1 otherwise. """
        for i in range(self.size):
            if self.keys[i] == key:
                return i
        return -1

    def containsKey(self, key):
        """ Check to see if arraymap contains the key."""
        index = self.keyIndex(key)
        return index > -1

    def put(self, key, value):
        """ Associate key with value."""
        index = self.keyIndex(key)
        if index == -1:
            if self.size == len(self.keys):
                self.resize(self.size * 2)
            self.keys[self.size] = key
            self.values[self.size] = value
            self.size += 1
        else:
            self.values[index] = value

    def remove(self, key, value=None):
        i = self.keyIndex(key)
        returnVal = self.values[i]
        if value and returnVal != value:
            return None
        self.size -= 1
        if i != 0:
            self.keys[i] = self.keys[self.size]
            self.values[i] = self.values[self.size]
        self.keys[self.size] = None
        self.values[self.size] = None
        if self.size <= len(self.keys) // 4:
            self.resize(len(self.keys) // 2)
        return returnVal

    def get(self, key):
        """ Returns value, assuming key exists..."""
        index = self.keyIndex(key)
        if index > -1:
            return self.values[index]
        else:
            return None

    def maxKey(self):
        if self.size == 0:
            return None
        largest_item = (self.keys[0], self.values[0])
        for i in range(1, self.size):
            if self.values[i] > largest_item[1]:
                largest_item = (self.keys[i], self.values[i])
        return largest_item[0]

    # def size(self):
        # """ Returns number of keys."""
        # pass

    # def keys(self):
        # """ Returns list of keys."""
        # keyList = list()
        # for i in range(self.size):
        #      keylist.append(self.keys[i])
        # return keylist


###########
## TESTS ##
###########

import unittest
class InsertionsTest(unittest.TestCase):

    def test1(self):
        a = ArrayMap()
        lst = ['a', 'b', 'c']
        for i in range(3):
            a.put(lst[i], i)
        for x in a:
            print(x)

if __name__ == '__main__':
    #import doctest
    #doctest.testmod()

    unittest.main()
