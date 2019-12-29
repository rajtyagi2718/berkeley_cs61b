

class WeightedQuickUnionDSWithPathCompression:

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
