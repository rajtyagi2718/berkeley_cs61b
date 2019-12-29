def listPrimesV5(n):
    if n <= 1:
        return []
    if n == 2:
        return [2]
    if n == 3:
        return [2, 3]
    results = []
    m = math.floor(math.sqrt(n))
    # print('m= ', m)
    bools = [True for _ in range(3, m+1, 2)]
    stop = (m - 1) // 2
    for index in range(len(bools)):
        if bools[index]:
            p = 3 + 2*index
            #for q in range(p**2, m+1, 2*p):
            start = (p**2 - 3) // 2
            incr = p
            for j in range(start, stop, incr):
                #qIndex = (q - 3) // 2
                bools[j] = False
    # result does not contain 2 for now
    results.append([3 + 2*i for i in range(len(bools)) if bools[i]])
    # print('init primes=', results)
    for k in range(1, m):
        # print('k=', k)
        beg = k*m + 1 + k*m % 2 #beg is always odd
        end = (k+1) * m
        # print('beg, end:', beg, end)
        bools = [True for _ in range(beg, end+1, 2)]
        stop = len(bools) #(end - (beg-1 % 2)) // 2...
        # print('stop=', stop, bools)
        for result in results:
            for p in result:
                # print('checking ', p)
                if p**2 > end:
                    break
                start = (p**2 - beg) // 2
                if start < 0:
                    start %= p
                # print('start=', start)
                incr = p
                for j in range(start, stop, incr):
                    # print('remove ', beg + 2*j)
                    bools[j] = False
        results.append([beg + 2*i for i in range(len(bools)) if bools[i]])
        # print('primes so far ', results)
    beg = m**2 + 1 + (m**2) % 2
    end = n
    # print('last seg', beg, end)
    bools = [True for _ in range(beg, end+1, 2)]
    stop = len(bools)
    for result in results:
        for p in result:
            # print('checking ', p)
            if p**2 > end:
                break
            start = (p**2 - beg) // 2
            if start < 0:
                start %= p
            # print('start=', start)
            incr = p
            for j in range(start, stop, incr):
                # print('remove ', beg + 2*j)
                bools[j] = False
    results.append([beg + 2*i for i in range(len(bools)) if bools[i]])
    return [2] + [p for r in results for p in r]


def listPrimesV3(n): #index conversion ideal?
#boolean list of odds, cross off odd prime multiples
    if n <= 1:
        return []
    result = [2]
    bools = [True for _ in range(3, n+1, 2)]
    for index in range(len(bools)):
        if bools[index]:
            p = 3 + 2*index
            if p**2 > n:
                break
            #for q in range(p**2, n+1, 2*p):
            start = (p**2 - 3) // 2
            stop = (n - 1) // 2
            incr = p
            for j in range(start, stop, incr):
                #qIndex = (q - 3) // 2
                bools[j] = False
    return result + [3 + 2*i for i in range(len(bools)) if bools[i]]

n = 1000
for i in range(n):
    a, b = listPrimesV5(i), listPrimesV3(i)
    if a != b:
        print(i, a, b, sep='\n')
        break
    if i == n-1:
        print('good up to ', n-1)
