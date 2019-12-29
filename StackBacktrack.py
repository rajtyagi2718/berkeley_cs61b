# rewrite breadth first search more clearly
# list copying takes time
# tree?

class Maze:

    def __init__(self, grid, start, end):
        self.grid = grid
        self.start = start
        self.end = end
        self.dead = []
        self.prev = []

    def isEnd(self, place):
        return place == self.end

    def isNotDead(self, place):
        return place not in self.dead

    def isOnGrid(self, place):
        row, col = place
        numRows = len(grid)
        numCols = len(grid[0])
        return (0 <= row and row < numRows and
                0 <= col and col < numCols)

    def isNotBlocked(self, place):
        row, col = place
        return grid[row][col] == '.'

    def isGoodToMove(self, place):
        return (self.isOnGrid(place) and self.isNotBlocked(place) and
                self.isNotDead(place))

    def listNextMoves(self, place):
        result = []
        for step in Steps.listMethods():
            dir = step(place)
            while self.isGoodToMove(dir):
                result.append(dir)
                dir = step(dir)
        return result

    def addDead(self, place):
        self.dead.append(place)

    def addPrev(self, place):
        self.prev.append(place)


class Steps:

    def listMethods():
        return [Steps.left, Steps.right, Steps.up, Steps.down]

    def left(place):
        x, y = place
        return x-1, y

    def right(place):
        x, y = place
        return x+1, y

    def up(place):
        x, y = place
        return x, y+1

    def down(place):
        x, y = place
        return x, y-1

def backFind(lst, item, stop):
        position = -1
        while lst[position] != stop:
            if lst[position] == item:
                return True
            position -= 1
        return False

grid = [['.','.','.'],['.','X','.'],['.','X','.']]
start = (2, 0)
end = (2, 2)
M = Maze(grid, start, end)

def findEndPath(grid, start, end):
    pass


def minimumMoves(grid, startX, startY, goalX, goalY):
    n = len(grid)
    start = (startX, startY)
    end = (goalX, goalY)

    if s == g:
        return 0

    def isNotPrevious(x, prevPositions):
        return x not in prevPositions

    def isLegalPosition(x):
        rowBounds = x[0] >= 0 and x[0] < n
        columnBounds = x[1] >= 0 and x[1] < n
        if rowBounds and columnBounds:
            notBlocked = grid[x[0]][x[1]] == '.'
            return notBlocked
        return False

    def nextMoves(x, prevPositions):
        f = lambda x: (isNotPrevious(x, prevPositions) and
                       isLegalPosition(x))
        n1 = lambda y: (y[0]-1, y[1])
        s1 = lambda y: (y[0]+1, y[1])
        w1 = lambda y: (y[0], y[1]-1)
        e1 = lambda y: (y[0], y[1]+1)
        moves = [n1, s1, w1, e1]
        result = []
        for m in moves:
            y = m(x)
            while f(y):
                result.append(y)
                y = m(y)
        return result

    pathLengths = []

    def helper(currPosition, prevPositions):
        # print('curr=', currPosition)
        prevPositions.append(currPosition)
        # print('prev:', prevPositions)
        nextPositions = nextMoves(currPosition, prevPositions)
        # print('next:', nextPositions)
        if nextPositions == []:
            # print('not good', prevPositions)
            return []
        if g in nextPositions:
            pathLengths.append(len(prevPositions))
            # print('good', prevPositions)
            return
        return [[x, list(prevPositions)] for x in nextPositions]

    currPosition = s
    prevPositions = []
    stack = [[currPosition, prevPositions]]
    while pathLengths == []:
        newStack = []
        for x in stack:
            helperX = helper(*x)
            if helperX == None:
                break
            newStack.extend(helperX)
        stack = newStack

    return pathLengths[0]
