import tkinter as tk
from tkinter import Canvas
import random
from math import sqrt

def create_circle(x,y,rad,c):
    x0=x-rad
    y0=y-rad
    x1=x+rad
    y1=y+rad
    return c.create_oval(x0,y0,x1,y1)
root = tk.Tk()
# width x height + x_offset + y_offset:
root.geometry("600x600+300+300")
canvas=Canvas(root)
class obstacle:
    def __init__(self,pos,size):
            self.pos = pos
            self.size = size
o1 = obstacle([1,2],[100,60])
o2 = obstacle ([100,200],[50,50])
canvas.create_rectangle(o1.pos[0],o1.pos[1],o1.size[0],o1.size[1])
canvas.create_rectangle(o2.pos[0],o2.pos[1],o2.size[0],o2.size[1])
create_circle(100,100,50,canvas)
canvas.pack(fill="both",expand=1)



class Point:
    def __init__(self,x_init,y_init):
        self.x = x_init
        self.y = y_init

    def shift(self, x, y):
        self.x += x
        self.y += y

    def __repr__(self):
        return "".join(["Point(", str(self.x), ",", str(self.y), ")"])

p1 = Point(133, 133)
p2 = Point(111, 222)


def distance(a, b):
    return sqrt((a.x-b.x)**2+(a.y-b.y)**2)

print(p1.x, p1.y, p2.x, p2.y)
print(distance(p1,p2))

p2.shift(2,3)

print(p2)
def draw_robot(self,pos,rad,dir):
    tk.Canvas.create_circle(x,y,rad,fill="blue")
root.mainloop()

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 8)

# intial parameters
n_iter = 50
sz = (n_iter,) # size of array
x = -0.37727 # truth value (typo in example at top of p. 13 calls this z)
z = np.random.normal(x,0.1,size=sz) # observations (normal about x, sigma=0.1)

Q = 1e-5 # process variance

# allocate space for arrays
xhat=np.zeros(sz)      # a posteri estimate of x
P=np.zeros(sz)         # a posteri error estimate
xhatminus=np.zeros(sz) # a priori estimate of x
Pminus=np.zeros(sz)    # a priori error estimate
K=np.zeros(sz)         # gain or blending factor

R = 0.1**2 # estimate of measurement variance, change to see effect

# intial guesses
xhat[0] = 0.0
P[0] = 1.0

for k in range(1,n_iter):
    # time update
    xhatminus[k] = xhat[k-1]
    Pminus[k] = P[k-1]+Q

    # measurement update
    K[k] = Pminus[k]/( Pminus[k]+R )
    xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
    P[k] = (1-K[k])*Pminus[k]

plt.figure()
plt.plot(z,'k+',label='noisy measurements')
plt.plot(xhat,'b-',label='a posteri estimate')
plt.axhline(x,color='g',label='truth value')
plt.legend()
plt.title('Estimate vs. iteration step', fontweight='bold')
plt.xlabel('Iteration')
plt.ylabel('Voltage')

plt.figure()
valid_iter = range(1,n_iter) # Pminus not valid at step 0
plt.plot(valid_iter,Pminus[valid_iter],label='a priori error estimate')
plt.title('Estimated $\it{\mathbf{a \ priori}}$ error vs. iteration step', fontweight='bold')
plt.xlabel('Iteration')
plt.ylabel('$(Voltage)^2$')
plt.setp(plt.gca(),'ylim',[0,.01])
plt.show()