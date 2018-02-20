def ehcf(a,b):
    p1,q1,h1=1,0,a
    p2,q2,h2=0,1,b
    while h2>0:
        r=h1//h2
        p3=p1-r*p2
        q3=q1-r*q2
        h3=h1-r*h2
        p1,q1,h1,p2,q2,h2=p2,q2,h2,p3,q3,h3
    return h1
#print(ehcf(12,8))

def multiple(n,e):
    p1,q1,h1=ehcf(n,e)
    print(p1,q1,h1)
    i = 0
    while q1+i*n<0:
        i+=1
    print("d and positive d are", q1,q1+i*n)
#print(multiple(2**36,3**10))
