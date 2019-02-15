import math


def sum_half(arr):
    new_arr = []
    for i in range(int(len(arr)/2)):
        value = sum(arr[i*2:i*2+2])
        new_arr.append(value)
    return new_arr

def prefix_sum_build_tree(arr):
    divides = int(math.log(len(arr), 2)) + 1
    size = 2**divides
    new_arr = arr + [0 for i in range(size-len(arr))]
    
    sums = []
    
    for i in range(divides+1):
        sums.append(new_arr)
        new_arr = sum_half(new_arr)
        
        size = int(size / 2)
    tree = []
    #print(sums)
    index = 0

    sums_ = sums
    sums = []
    for i in reversed(sums_):
        sums += i

    sgeo = lambda a1, r, n: int((a1 / (1-r)) * (1-r**(n+1)))
    for i in range(0, divides+1, 1):
        for v in sums_[len(sums_)-1-i]:
            where = int(math.log(index+1, 2))
            pwhere = where - 1
            S = sgeo(1, 2, where)
            pS = sgeo(1, 2, pwhere)
            #print(S, pS)
            #print(index, S)
            l = (index*2+1)*3
            r = (index*2+2)*3
            aindx = (S-(index+1))+pS
            aindx = index
            #print(aindx, S)
            V = sums[aindx]
            #print(v, V)#, S, pS, aindx)
            entry = [l, r, V]
            tree += entry

            index += 1
   # print(tree)
    return tree, divides

def get_prefix_sum2(i, tree, d):
    tind = 0
    total = tree[2]
    bitness = d
    b = bin(i)[2:]
    filled = (bitness-len(b)) * "0" + b
    

    for n in range(d): # correct number of iterations
        c1 = tree[tind+0]
        c2 = tree[tind+1]
        rsum = tree[c2+2]
        if filled[n] == "0":
            tind = c1
            total -= rsum
        else:
            tind = c2

    return total

def get_prefix_sum(i_, tree, d):
    tind = 0
    total = tree[2]
    bits = d

    i = i_ - 1
    if i == -1:
        total = 0
    
    for n in range(d): # correct number of iterations
        shifted = (2**bits) >> (n+1)
        #print(bits, n, shifted)
        direction = i & shifted
        #print(direction)
        c1 = tree[tind+0]
        c2 = tree[tind+1]
        rsum = tree[c2+2]
        if direction == 0:
            tind = c1
            total -= rsum
        else:
            tind = c2
    
    return total

def prefix_sum_parallel(arr):
    tree, d = prefix_sum_build_tree(arr)
    #print(tree)
    prefix_sums = []
    for i in range(len(arr)):
        prefix_sums.append(get_prefix_sum(i, tree, d))

    return prefix_sums


import random
test_arr1 = [random.randint(0, 100) for i in range(1000)]
test_arr2 = [i for i in range(256)]
test_arr3 = [i for i in range(64)]

#ps = prefix_sum_parallel(test_arr1)
#prefix_sum_parallel(test_arr2)
result = prefix_sum_parallel(test_arr3)
