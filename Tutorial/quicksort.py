def quicksort(a):
    if len(a) <= 1:
        return a
    pivot = a[len(a)//2]
    left = [x for x in a if x < pivot]
    middle = [x for x in a if x == pivot]
    right = [x for x in a if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print(quicksort([3,6,8,10,1,2,1]))

x = 3
print(type(x))
print(x)
print(x + 1)
print(x -1)
print(x * 2)
print(x ** 2)
x += 1
print(x)
x *= 2
print(x)
y=2.5
print(type(y))
print(y, y + 1, y * 2, y ** 2)