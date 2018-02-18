from random import randint

def isprime3(n):
    if n<=1:
        return False
    elif n == 2 or n==3:
        return True
    else:
        for i in range(0,10):
           x = randint(2,n-2)
           if pow(x,n,n)!=x:
                return False
        return True
# print(isprime3(560))