#def cube(n):
#    return n**3
#
#ints=list(range(2,26))
#for r in range(0,24):
#    pr=ints[r]
#    ints=[n for n in ints if n%pr>0 or n==pr]
# 
#    ints= list(range(2,26))
#for r in range (0,4):
#    pr=ints[r]
#    ints=[n for n in ints if n%pr>0 or n==pr]
#    print(ints)
#    
#
#    from math import exp
#    x=1.0
#    for n in range(20):
#        x=exp(-x)
#    print(x)
#        
#    a=1
#    b=1
#    for n in range(3,101):
#        temp=b
#        b=a+b
#        a=temp
#    print(b)
    
#def fibonacci(r):
#        if r==1 or r==2:
#            return 1
#        else:
#            a,b=1,1
#            for i in range(3,r+1):
#                a,b=b,a+b
#            return b
#        
#    def fibonacci_list(r):
#        if r==1: 
#            return [1]
#        elif r==2:
#            return [1,1]
#        else:
#            a,b=1,1
#            fib_list=[1,1]
#            for i in range(3,r+1):
#                a,b=b,a+b
#                #append b to fib_list
#                fib_list.append(b)
#                
#            return fib_list
#        
#def hailsyone(a, max_iterations):
#        #initialization
#        alist=[a]
#        #loop max_iterations times
#        for n in range(max_iterations):
#            #test if a evenn
#            if a%2==0:
#                #halve a
#                a=a//2
#            else:
#                a=3*a+1
#                #append latest
#                alist.append(a)
#                #a break if a is 1
#                if a==1:
#                    break
#                #return final value of alist
#                return alist
            
def hailstone1(a):
        from itertools import count
        #initialization
        alist=[a]
        #loop max_iterations times
        for n in count():
            #test if a evenn
            if a % 2 == 0:
                #halve a
                a=a//2
            else:
                a=3*a+1
                #append latest
            alist.append(a)
            #a break if a is 1
            if  a==1:
                break
            #return final value of alist
            return alist

'''def isprime2(n):
    return n in primes(n)

def isprime1(n):
    from math import sqrt
    for r in range(2,int(sqrt(n)+1)):
        if n%r==0:
            return False
    return True

def primes(n):
    plist=[]
    for i in range(1,n):
        if isprime1(i)==True:
            plist.append(i)
    print(plist)
    
def fermat_test(n,a):
    return (pow(a,n,n)==a)


def ehcf(a,b):
    from math import *
    p1=1,q1=0,h1=a
    p2=1,q2=0,h2=b
    while h2>0:
        r=h1//h2
        p3=p1-r*p2
        q3=q1-r*q2
        h3=h1-r*h2
    return p1,q1,h1
'''

        
    
    

   

    