class Node:

    def __init__(self, key, value, left=None, right=None, size=1):
        assert ((left is None or isinstance(left, Node)) and
                (right is None or isinstance(right,Node)))
        self.key = key
        self.value = value
        self.left = left
        self.right = right
        self.size = size

class BST:

    def __init__(self, root=None):
        self.root = root

    def size(self):
        if not self.root:
            return 0
        result = self.root.size
        if self.root.left:
            result += self.root.left.size()
        if self.root.right:
            result += self.root.right.size()
        return result

    def find(self, key):
        if not self.root:
            return None
        if key == self.root.key:
            return self.root.value
        if key < self.root.key:
            if not self.root.left:
                return None
            return self.root.left.find(key)
        if key > self.root.key:
            if not self.root.right:
                return None
            return self.root.right.find(key)

    def insert(self, key, value):
        if not self.root:
            self.root = Node(key, value)
        elif key == self.root.key:
            self.root.value = key
        elif key < self.root.key:
            if not self.root.left:
                self.root.left = BST()
            self.root.left.insert(key, value)
        else: # key > self.root.key:
            if not self.root.right:
                self.root.right = BST()
            self.root.right.insert(key, value)

    def delete(self, key):
        node = self.root
        if node.key == key:
            if node.left:
                

        while node.key != key:
            if node.key < key:
                node = node.left
            else:
                node = node.right




    @property
    def key(self):
        return self.root.key

    @property
    def value(self):
        return self.root.value

    @property
    def left(self):
        return self.root.left

    @property
    def right(self):
        return self.root.right
