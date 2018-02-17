from math import sqrt

def prime1(n):
    for r in range(2, int(sqrt(n))+1):
        if n % r == 0:
            return False
    return True
print(prime1(23))

def primes(n):
    x = []
    for num in range(2, n):
        status = True
        for i in range(2, num):
            if num % i == 0:
                status = False
        if status:
            x.append(num)
    print(x)

primes(25)
