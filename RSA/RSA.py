l1 = [ord(char)-32 for char in 'My name is Bond']
pt = [ord(char)-32 for char in 'yuhan.du17@imperial.ac.uk']
ptj = int(''.join(map(str, pt)))
print(ptj)

n = 5101840198838930959955074732856532015727277325559096180034945208454221257076367307470390454365875514870472254402420766727874928909352737261630117040011773047026455965372829
e = 32226577655179845269168208085215800138699
ct = 5970250928760092818098248851210017441366824848327954848109890158004502344881171312289600564419830756330192780962428935964749587942161616007397776717958977313099202366638372

n1 = 9655020917106901547888376921602009
e1 = 43

n2 = 2867
e2 = 7
ct2 = [694, 1960, 2612, 1134, 579, 2515, 836, 1960, 2165, 2050, 1981, 2360, 2394, 2749, 2229, 1321, 2360, 1134, 2767, 2515, 1134, 2752, 2515, 1960, 1051]
d = 1183

def encrypt(pt,n,e):
    x = (pt**e) % n
    return x
3print(encrypt(ptj,n,e))

def decrypt(ct,n,d):
    decrypted=[]
    for i in range(0, len(ct)):
        x = (ct[i]**d) % n
        decrypted.append(x)
    return decrypted
#print(decrypt(ct,n2,d))

def get_factors(n):

    def largest_prime_factor(n):
        i = 2
        while i * i <= n:
            if n % i:
                i += 1
            else:
                n //= i
        return n

    p = largest_prime_factor(n)
    q = n/p

    return p, q
#print(get_factors(n))

def get_key(n,e):

    p, q = get_factors(n)
    phi = (p - 1) * (q - 1)

    def egcd(a, b):
        if a == 0:
            return (b, 0, 1)
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)

    def modinv(a, m):
        g, x, y = egcd(a, m)
        if g != 1:
            raise Exception('No modular inverse')
        return x % m

    d = modinv(e, phi)

    return d
#print(get_key())