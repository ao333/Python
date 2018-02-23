import numpy as np
import math

'''
x = np.arange(3)
print(x)
w = np.arange(3*3).reshape(3,3)
print(w)
f = np.dot(w,x)
print(f)
sum = np.sum(w, axis=0)
print(sum)

a = 0*0+1*1+2*2
b = 0*3+1*4+2*5
c = 0*6+1*7+2*8
print(a,b,c)

x = np.arange(3)
print(x)
w = np.arange(3*3).reshape(3,3)
print(w)
f = np.dot(w,x)
print(f)

x = np.array([56,231,24,2])
N = x.shape[0]
X = x.reshape(N, -1)
print(X)

w = np.array([[0.2,-0.5,0.1,2],[1.5,1.3,2.1,0],[0,0.25,0.2,-0.3]]).reshape(3,4)
print(w)

b = np.array([1.1,3.2,-1.2])

f = w.dot(X) + b
print(f)

cat = [3.2,5.1,-1.7]
catprob = np.exp(3.2)
print(catprob)

w = [2,-3,-3]
x = [-1, -2]

# forward pass
dot = w[0]*x[0] + w[1]*x[1] + w[2]
print(dot)
f = 1.0 / (1 + math.exp(-dot)) # sigmoid function
print(f)

# backward pass
ddot = (1 - f) * f # gradient on dot variable, using the sigmoid gradient derivation
print(ddot)
dx = [w[0] * ddot, w[1] * ddot] # backprop into x
print(dx)
dw = [x[0] * ddot, x[1] * ddot, 1.0 * ddot] # backprop into w

W = np.random.randn(5, 6)
X = np.random.randn(6, 3)
D = W.dot(X)
print(W)
print(X)
print(D)

dD = np.random.randn(*D.shape) # same shape as D
dW = dD.dot(X.T) #.T gives the transpose of the matrix
dX = W.T.dot(dD)
print(dD)
print(dW)
print(dX)

def graph(x,y):
    print(((2*x-y)**2)*(x-2*y))
print(graph(5,5))
print(graph(5,-5))
print(graph(-5,5))
print(graph(-5,-5))
'''