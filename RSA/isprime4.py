import random as rand
def isprime4(n):
    if n<=1:
        return False
    if n==2 or n==3:
        return True
    if n%2==0:
        return False
    r,s=0,n-1
    while s%2==0:
        r+=1
        s//=2
    for i in range(10):
        a=rand.randrange(2,n-1)
        x=pow(a,s,n)
        if x==1 or x==n-1:
            continue
        for j in range(r-1):
            x=pow(x,2,n)
            if x==n-1:
                break
        else:
            return False
    return True
# print(isprime4(563))