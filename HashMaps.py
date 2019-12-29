# A hash table-backed Map implementation.
# Provides amortized constant time access
# to elements via get(), remove(), and put()
# in the best case.

class ArrayMap:

    def __init__(self, length=8):
        self.keys = [None] * length
        self.values = [None] * length
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

class HashMap:

    DEFAULT_LENGTH = 16 # number of buckets
    MAX_LF = 0.75

    def __init__(self, length=None):
        if not length:
            length = (HashMap.DEFAULT_LENGTH)
        self.buckets = [ArrayMap() for _ in range(length)]
        self.size = 0 #number of items

    def loadFactor(self):
        return self.size / len(self.buckets)

    def clear(self):
    #  Removes all of the mappings from this map.
        self.buckets = [ArrayMap() for _ in range(len(self.buckets))]
        self.size = 0

    def hash(self, key):
    # Computes the hash function of the given key. Consists of
    # computing the hashcode, followed by modding by the number
    # of buckets.
        if key == None:
            return 0
        numBuckets = len(self.buckets)
        return hash(key) % numBuckets

    def get(self, key):
        bucketNum = self.hash(key)
        return self.buckets[bucketNum].get(key)

    def put(self, key, value):
        bucketNum = self.hash(key)
        bucketSize = self.buckets[bucketNum].size
        self.buckets[bucketNum].put(key, value)
        self.size += (self.buckets[bucketNum].size - bucketSize)
        if self.loadFactor() > HashMap.MAX_LF:
            self.resize(len(self.buckets) * 2)

    def resize(self, capacity):
        newHashMap = HashMap(capacity)
        for b in self.buckets:
            for i in range(b.size):
                key, value = b.keys[i], b.values[i]
                newHashMap.put(key, value)
        self.buckets = newHashMap.buckets

    def keySet(self):
        returnSet = []
        for bucket in self.buckets:
            returnSet += [key for key in bucket]

    def remove(self, key, value=None):
        bucketNum = self.hash(key)
        returnVal = self.buckets[bucketNum].remove(key, value)
        if returnVal != None:
            self.size -= 1
        # if (self.loadFactor() > 0 and
        #     self.loadFactor() < HashMap.MIN_LF):
        #     self.resize(self.size // 4)
        return returnVal

    def __iter__(self):

        class KeyIterator:

            def __init__(self, buckets, size):
                self.buckets = buckets
                self.size = size
                self.bucketNum = 0
                self.bucketItr = self.buckets[self.bucketNum].__iter__()
                self.next = 0

            def incrementItr(self):
                self.bucketNum += 1
                self.bucketItr = self.buckets[self.bucketNum].__iter__()

            def __next__(self):
                if self.next == self.size:
                    raise StopIteration
                try:
                    returnVal = self.bucketItr.__next__()
                    self.next += 1
                    return returnVal
                except StopIteration:
                    self.incrementItr()
                    return self.__next__()

            def __iter__(self):
                return self

        return KeyIterator(self.buckets, self.size)

        #return itertools.chain.from_iterable(self.buckets)
