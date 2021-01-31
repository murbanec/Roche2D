
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 10:37:04 2021

@author: martin urbanec
"""

#calculates trajectory of small mass positioned close to L4 Lagrange point
#creates gif as output

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter  


DistanceJ = 778570000000. # m JUPITER FROM SUN
G = 6.67259*10**-11
Jupiter_mass = 1.8982*10**27 # kg
Sun_mass = 1.989*10**30 # kg



M1=Sun_mass
M2=Jupiter_mass
M2=M1/30.
a=DistanceJ
Ang_vel=math.sqrt(G*(M1+M2)/(a**3)) #FROM KEPLER LAW
P=2.*math.pi/Ang_vel #Period
#center of mass is located at [0,0] massive object (Sun) is located at -r1, secondary object (Jupiter) is located at +r2
r2=M1*a/(M1+M2) 
r1=M2*a/(M1+M2)



# Calculations are done in corotating frame
#  s1, s2 are distances from sources of gravity (Sun, Jupiter)
def pot(x,y):
    r=math.sqrt(x*x + y*y)    
    if x==0: 
        if y>0:
            theta=math.pi/2.
        if y<0:
            theta=math.pi/2.
    if x>0:
        theta=math.atan(abs(y)/x)
    else:
        theta=math.pi-math.atan(abs(y)/x)
    s1=math.sqrt(r1*r1 + r*r + 2.*r1*r*math.cos(theta))
    s2=math.sqrt(r2*r2 + r*r - 2.*r2*r*math.cos(theta))
    result = -G*(M1/s1 + M2/s2) -1.*Ang_vel*Ang_vel*r*r/2. 
    return result

#Force per unit mass (acceleration) in x direction 
# ax = \partial pot(x,y) / \partial x - 2 \Omega \times v 
# in our case \Omega=(0,0,\Omega) and v=(vx,vy,0)
# second term is corresponding to Coriolis force


def ax(x,y,vx,vy):
    dx=a/1000.
 #   result=-(pot(x+dx,y) -pot(x-dx,y))/(2.*dx)    + 2.* Ang_vel*vy 
    result=-(-pot(x+2.*dx,y) + 8.*pot(x+dx,y) - 8.*pot(x-dx,y) + pot(x-2.*dx,y))/(12.*dx)    + 2.* Ang_vel*vy 
    return result

def ay(x,y,vx,vy):
    dy=a/1000.
#    result=-( pot(x,y+dy)-pot(x,y-dy))/(dy*2.) - 2.* Ang_vel*vx
    result=-(-pot(x,y+2.*dy) + 8.*pot(x,y+dy) - 8.*pot(x,y-dy) + pot(x,y-2*dy))/(dy*12.) - 2.* Ang_vel*vx 
    return result




pot2=np.vectorize(pot)



#TRAJECTORY OF ASTEROID CLOSE STARTING CLOSE TO L4 in rest with respecting to the rotating frame

x0=a/2.-r1
y0=math.sqrt(3)*a/2.
x0=1.005*x0
y0=1.005*y0
vx0=0.
vy0=0.
steps=300000
#initialize arrays
x= np.linspace(0, 10, steps) 
y= np.linspace(0, 10, steps) 
vx=np.linspace(0, 10, steps) 
vy=np.linspace(0, 10, steps) 
t= np.linspace(0, 10, steps) 
x[0]=x0
vx[0]=vx0
y[0]=y0
vy[0]=vy0
t[0]=0.
i=0

timescale = math.sqrt((a*a)**1.5 / G/(M1+M2))
dt=timescale/1000.

for i in range (1,steps):
    t[i]=(t[i-1]+dt)
    Kx1=dt*ax(x[i-1],y[i-1],vx[i-1],vy[i-1])
    Kx2=dt*ax(x[i-1],y[i-1],vx[i-1]+Kx1/2.,vy[i-1])
    Kx3=dt*ax(x[i-1],y[i-1],vx[i-1]+Kx2/2.,vy[i-1])
    Kx4=dt*ax(x[i-1],y[i-1],vx[i-1]+Kx3,vy[i-1])
    vx[i]=vx[i-1] + Kx1/6. + Kx2/3. + Kx3/3. + Kx4/6.

    Ky1=dt*ay(x[i-1],y[i-1],vx[i-1],vy[i-1])
    Ky2=dt*ay(x[i-1],y[i-1],vx[i-1],vy[i-1]+Ky1/2.)
    Ky3=dt*ay(x[i-1],y[i-1],vx[i-1],vy[i-1]+Ky2/2.)
    Ky4=dt*ay(x[i-1],y[i-1],vx[i-1],vy[i-1]+Ky3)
    vy[i]=vy[i-1] + Ky1/6. + Ky2/3. + Ky3/3. + Ky4/6.
 
     
    
    x[i]=x[i-1] + (vx[i-1]+vx[i])*dt/2. #taking the average of velocities
    y[i]=y[i-1] + (vy[i-1]+vy[i])*dt/2. 
    dt=timescale/1000.



#LAGRANGE POINTS
#L3, L1 and L2 points are lying on x-axis (left to right) for small values of alpha=M2/(M1+M2) the positions can are given analytically (to first order in alpha)
#
    
alpha=M2/(M1+M2)

L1X=a*(1.-(alpha/3.)**(1./3.))
L1Y=0.
P1=pot(L1X,L1Y)

L2X=a*(1.+(alpha/3.)**(1./3.))
L2Y=0.
P2=pot(L2X,L2Y)

L3X=-a*(1. + 5.*alpha/12)
L3Y=0.
P3=pot(L3X,L3Y)

L4X=a/2.-r1
L4Y=math.sqrt(3)*a/2.
P4=pot2(L4X,L4Y)



P0=pot(x0,y0)


steps=301
xx= np.arange(-2*a, 2.*a,a/steps)
yy= np.arange(-1.5*a, 1.5*a,a/steps)
X, Y = np.meshgrid(xx, yy)
Z1=pot2(X,Y)

 

fig, ax = plt.subplots()
ax.set_aspect('equal','box')


ln1, = plt.plot([],[], 'k+')  
ln2, = plt.plot([], [], 'm*')  

XXX,YYY=[],[]

def init():  
    ax.set_xlim(-1.25,1.25)  
    ax.set_ylim(-1.25,1.25)  
    ax.contour(X/a, Y/a, Z1,levels=[P1,P2,P3,P0],colors=('r', 'green', 'blue', 'm'))
  
def update(i):  
    ln1.set_data(x[1000*i]/a, y[1000*i]/a)  



 
zed= np.arange(60)
ani = FuncAnimation(fig, update, np.arange(300), init_func=init)  
plt.show()

writer = PillowWriter(fps=25)  
ani.save("Animation.gif", writer=writer) 



    
