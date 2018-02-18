from isprime4 import isprime4

def nextprime(n):
    if(isprime4(n) is True):
        n+=1
    while(isprime4(n) is False):
       n+=1
    print(n)

nextprime(17**81)

