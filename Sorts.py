######################
##SORTING ALGORITHMS##
######################

# Note: implement quick_sort_random_pivot
# Note: start heap_sort_in_place, merge_sort_in_place
# Note: check stability i.e. if equal items stay in relative order

##############
## Preamble ##
##############

#def argmax(lst):
#    return max([i for in range(len(lst))], key=lambda i: lst[i])

def argmax(lst):
    return max(enumerate(lst), key=lambda x: x[1])[0]

def argmin(lst):
    return min(enumerate(lst), key=lambda x: x[1])[0]

def enumerate_max(lst):
    return max(enumerate(lst), key=lambda x: x[1])

def swapIndices(lst, i, j):
    lst[i], lst[j] = lst[j], lst[i]

###############
## Selection ## in_place
###############

# Find min, move to front, find next min,

# TIME: Theta(n**2)
# SPACE: Theta(1) naive, Theta(n) constructive

# Note: truly in place: argmax def in place, not slice
# Note: can check if max_i == bd to determine swap or not

def selection_sort_min_in_place(lst):
    for bd in range(len(lst)-1):
        min_i = bd + argmin(lst[bd : ])
        lst[min_i], lst[bd] = lst[bd], lst[min_i]
    return lst

def selection_sort_min(lst):
    aux = []
    while lst:
        min_i = argmin(lst)
        aux.append(lst.pop(min_i))
    return aux

def selection_sort_max_in_place(lst):
    for bd in range(len(lst) - 1, 0, -1):
        max_i = argmax(lst[ : bd + 1])
        lst[max_i], lst[bd] = lst[bd], lst[max_i]
    return lst

def selection_sort_max(lst):
    aux = []
    while lst:
        max_i = argmax(lst)
        aux.append(lst.pop(max_i))
    aux = aux[ : : -1]
    return aux

def selection_sort_max2(lst):
    bd = len(lst)
    while bd > 0:
        max_i = argmax(lst[ : bd])
        lst.insert(bd - 1, lst.pop(max_i))
        bd -= 1
    return lst


##########
## Heap ##
##########

# Heapify list, find remove place extreme into new list

# TIME: Theta(n log n)
# SPACE: theta(1), Theta(n)

import heapq

def heap_sort_min(lst):
    lst = list(lst)
    heapq.heapify(lst)
    aux = []
    while lst:
        aux.append(heapq.heappop(lst))
    return aux

def heap_sort_in_place(lst):
    pass


###########
## Merge ##
###########

# recursively sort each half, merge ordered halves

# TIME: Theta(nlogn)
# SPACE: Theta(n)

def merge_sort(lst):
    if len(lst) < 2:
        return lst
    l1 = len(lst) // 2
    l2 = len(lst) - l1
    if len(lst) > 1:
        h1 = merge_sort(lst[ : l1])
        h2 = merge_sort(lst[l1 : ])
    i, j = 0, 0
    aux = []
    while i < l1 and j < l2:
        if h1[i] < h2[j]:
            aux.append(h1[i])
            i += 1
        else:
            aux.append(h2[j])
            j += 1
    if i < l1:
        aux.extend(h1[i : ])
    elif j < l2:
        aux.extend(h2[j : ])
    return aux

def merge_sort_in_place(lst):
    pass


###############
## Insertion ## in_place
###############

# sort beginning of list, insert rest of items 1 by 1, swap in place

# TIME: theta(n), Theta(n**2)
# SPACE: theta(1)

# BEST USE: almost sorted arrays (num inv = Theta(n)),
#           small array (< 15)


def insertion_sort_in_place(lst):
    for i in range(1, len(lst)):
        for j in range(i, 0, -1):
            if lst[j-1] > lst[j]:
                lst[j-1], lst[j] = lst[j], lst[j-1]
            else:
                break
    return lst

def insertion_sort(lst):
    aux = [lst[0]]
    for i in range(1, len(lst)):
        for j in range(len(aux) - 1, -1, -1):
            if lst[i] >= aux[j]:
                aux.insert(j+1, lst[i])
                break
            elif j == 0:
                aux.insert(0, lst[i])
    return aux

def insertion_sort_shells(lst, strides=[]):
    if not lst: return lst
    if not strides:
        k = 1
        while 2**k <= len(lst):
            strides.append(2**k - 1)
            k += 1
    while strides:
        s = strides.pop()
        if s >= len(lst): continue
        for i in range(s, len(lst)):
            for j in range(i, s-1, -s):
                if lst[j - s] > lst[j]:
                    lst[j-s], lst[j] = lst[j], lst[j-s]
                else:
                    break
    return lst


###########
## Quick ## in_place
###########

# recursively choose pivot, swap <=,>= to left, right

# TIME: theta(n**2), Theta(nlogn), avg: nlogn
# SPACE: Theta(logn)

# BEST USE: array NOT almost sorted, nor mostly duplicates
# Imporovements: random pivot

def quick_sort_in_place(lst, left=None, right=None):
    if not left:
        left = 0
    if not right:
        right = len(lst)
    p, l = left, right
    while p + 1 < l:
        if lst[p] > lst[p + 1]:
            lst[p], lst[p + 1] = lst[p + 1], lst[p]
            p += 1
        else:
            lst.insert(right - 1, lst.pop(p + 1))
            # pop shortens list, insertion at right - 1
            l -= 1
    if p - left > 1:
        quick_sort_in_place(lst, left, p)
    if right - (p + 1) > 1:
        quick_sort_in_place(lst, p + 1, right)
    return lst

def quick_sort(lst):
    p, l = 0, len(lst)
    while p + 1 < l:
        if lst[p] > lst[p + 1]:
            lst[p], lst[p + 1] = lst[p + 1], lst[p]
            p += 1
        else:
            lst.insert(len(lst), lst.pop(p + 1))
            l -= 1
    left, right = lst[ : p], lst[p + 1 : ]
    if len(left) > 1:
        left = quick_sort_in_place(left)
    if len(right) > 1:
        right = quick_sort_in_place(right)
    return left + [lst[p]] + right



###########
## Tests ##
###########

import random
lst0 = [random.randint(0, 9) for _ in range(10)]
lst1 = [i for i in range(10)]; random.shuffle(lst1)

def test(i):
    if i == 0:
        print('TESTS')
    a = list(lst0)
    b = list(lst1)
    c = sorted(lst0)
    d = sorted(lst1)
    if i == 1: # selection_sort_min_in_place
        a = list(lst0)
        print('selection_sort_min_in_place')
        b = selection_sort_min_in_place(a)
        print(c == b)
    if i == 2: # selection_sort_min
        a = list(lst0)
        print('selection_sort_min')
        b = selection_sort_min(a)
        print(c == b)
    if i == 3: # selection_sort_max_in_place
        a = list(lst0)
        print('selection_sort_max_in_place')
        b = selection_sort_max_in_place(a)
        print(c == b)
    if i == 4: # selection_sort_max
        a = list(lst0)
        print('selection_sort_max')
        b = selection_sort_max(a)
        print(c == b)
    if i == 5: # selection_sort_max2
        a = list(lst0)
        print('selection_sort_max2')
        b = selection_sort_max2(a)
        print(c == b)
    if i == 6: # merge_sort
        print('merge_sort')
        a = merge_sort(lst0)
        b = merge_sort(lst1)
        print(a == c and b == d)
    if i == 7: # merge_sort_in_place
        pass

    if i == 8: # insertion_sort_in_place
        print('insertion_sort_in_place')
        a = insertion_sort_in_place(list(lst0))
        b = insertion_sort_in_place(list(lst1))
        print(a == c and b == d)
        #print(lst0, a, c, sep='\n')
        #print(lst1, b, d, sep='\n')
    if i == 9: # insertion_sort
        print('insertion_sort')
        a = insertion_sort(list(lst0))
        b = insertion_sort(list(lst1))
        print(a == c and b == d)
    if i == 10: # heap_sort_min
        print('heap_sort_min');
        print(c == heap_sort_min(a) and d == heap_sort_min(b))
    if i == 11: # insertion_sort_shells
        print('insertion_sort_shells')
        print(insertion_sort_shells(a) == c and
               insertion_sort_shells(b) == d)
        #e = list('SORTEXAMPLE')
        #insertion_sort_shells(e)
    if i == 12: # quick_sort_in_place
        print(quick_sort_in_place(a) == c and
              quick_sort_in_place(b) == d)
    if i == 13: # quick_sort
        print(quick_sort(a) == c and
              quick_sort(b) == d)

for i in range(13, 14):
    test(i)
