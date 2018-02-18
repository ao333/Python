from isprime3 import isprime3
from isprime4 import isprime4

def carmichael():
    for x in range(2,3001):
        if isprime3(x) != isprime4(x):
            print(x)

carmichael()