from ehcf import ehcf

pt = [ord(char)-32 for char in 'yuhan.du17@imperial.ac.uk']
print(pt)

#n = 5101840198838930959955074732856532015727277325559096180034945208454221257076367307470390454365875514870472254402420766727874928909352737261630117040011773047026455965372829
#e = 32226577655179845269168208085215800138699

p = 2074722246773485207821695222107608587480996474721117292752992589912196684750549658310084416732550077
#print(len(str(p)))
q = 1814159566819970307982681716822107016038920170504391457462563485198126916735167260215619523429714031
#print(len(str(q)))
n1 = p*q
print(n1)
#print(len(str(n1)))
e1 = 5202642720986189087034837832337828472969800910926501361967872059486045713145450116712488685004691423

def encrypt(pt,n,e):
    encrypted=[]
    for i in range(0, len(pt)):
        x = pow(pt[i], e, n)
        encrypted.append(x)
    return encrypted
print(encrypt(pt,n1,e1))

def decrypt(ct,n,d):
    decrypted=[]
    for i in range(0, len(ct)):
        y = pow(ct[i], d, n)
        decrypted.append(y)
    return decrypted
print(decrypt(encrypt(pt,n1,e1),n1,2437207784167187545862535959217917140428694671270056207539470920166570567627458925891397723862323179972793235758751030070984299020854935107972176789675916427431293001549390526274145218622956820724167))

def pick_e(p,q,e):
    phi = (p - 1) * (q - 1)
    if(ehcf(e, phi)) == 1 and e < phi:
        return phi
#print(pick_e(p,q,5202642720986189087034837832337828472969800910926501361967872059486045713145450116712488685004691423))

def get_d(e,phi):

    def egcd(a, b):
        if a == 0:
            return (b, 0, 1)
        g, y, x = egcd(b % a, a)
        return g, x - (b // a) * y, y

    def modinv(a, m):
        g, x, y = egcd(a, m)
        if g != 1:
            raise Exception('No modular inverse')
        return x % m

    d = modinv(e, phi)
    return d
#print(get_d(5202642720986189087034837832337828472969800910926501361967872059486045713145450116712488685004691423,3763877212478341464594602454228914655621760776946038991070349541916262235020064903144856319081877635112108098207762744348203440569158223173390858053058282491617069260844234598866790315077273334766280))

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

    def modinv(a, m):
        g, x, y = egcd(a, m)
        if g != 1:
            raise Exception('No modular inverse')
        return x % m

    d = modinv(e, phi)

    return d
#print(get_key(n1,e1))