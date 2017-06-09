
# coding: utf-8

# In[280]:

import random
import numpy as np
from numpy import *
import scipy.sparse as SP
m=512;n=1024;
A=np.mat(random.normal(0,1,(m,n)))
u=SP.random(n,1,density=0.1)
b=A*u
uArr=u.toarray()
def InfNmBallProj(Arr, R):
    ProjArr=Arr
    PosMask=Arr>R
    NegMask=Arr<-R
    ProjArr[PosMask]=R
    ProjArr[NegMask]=-R
    return ProjArr


# In[281]:

EPS=1E-14*n
MaxIter=10000
x0=np.zeros((n,1))


# In[282]:

import time
mu=1.1
strt = time.clock()
for i in range(1):
    
    xcur=x0
    lamdcur=mu*(A*x0-b)
    scur=np.zeros((n,1))
    AAinv=(A*A.T).I
fin=time.clock()
print "Precompute time(On Average):",(fin-strt),"s"
itr_new=[]
x_new=[]
u_new=[]
import numpy.linalg as la
for j in range(MaxIter): 
    
    snxt=InfNmBallProj(A.T*lamdcur+mu*xcur,1)
    lamdnxt=AAinv*(A*snxt+mu*(b-A*xcur))# 交替优化时 先优化哪个 用哪个结果会不一样
    delx=A.T*lamdnxt-snxt
    xnxt=xcur+delx/mu#/2
    stepres=la.norm(delx)
    itr_new.append(j)
    x_new.append(la.norm(xnxt,ord=1))
    u_new.append(la.norm(uArr,ord=1))
    if stepres<EPS:
        print "Iteration step no: ",j+1
        print "Optimized 1 norm",la.norm(xnxt,ord=1)
        print "Constraint Residue",la.norm(A*xnxt-b,ord=1)
        #return xnxt;
        break
    elif j==MaxIter-1:
        print "Warning! Unfinished Optimization"
        print "Iteration step no: ",j+1
        print "Optimized 1 norm",la.norm(xnxt,ord=1)
        print "Constraint Residue",la.norm(A*xnxt-b,ord=1)
        #return xnxt;
    elif j%100==0:
        dobj=float(b.T*lamdnxt)
        pobj=la.norm(xnxt,ord=1)
        dconsres=la.norm(A.T*lamdnxt-snxt,ord=2)
        pconsres=la.norm(A*xnxt-b,ord=2)
        print "# ",j,"Dual Obj",dobj,"Primal Obj",pobj,"dpGap",pobj-dobj,"Dual Constraint Res:",dconsres,"Primal Constraint Res:",pconsres
    xcur=xnxt
    scur=snxt
    lamdcur=lamdnxt
fin2=time.clock()
print("Total time: ",fin2-strt,"s")


# In[283]:

la.norm(uArr,ord=1)


# In[286]:

import matplotlib.pyplot as plt
#plt.plot(itr,x_,'b')
#plt.plot(itr,u_)
plt.plot(itr_new,x_new,'r')
#plt.plot(itr_alm,x_alm,'g')


# In[287]:

plt.show()


# In[243]:

plt.plot(xnxt)
plt.show()


# In[257]:


"""
Created on Sun Apr  9 13:23:13 2017

@author: Compute Binxu Wang
"""
from numpy import *
import numpy as np
import numpy.linalg as la
import scipy.sparse as SP
import matplotlib.pyplot as plt
import time
#data1=mat(zeros((3,3)))
##创建一个3*3的零矩阵，矩阵这里zeros函数的参数是一个tuple类型(3,3)
#data2=mat(ones((2,4)))
##创建一个2*4的1矩阵，默认是浮点型的数据，如果需要时int类型，可以使用dtype=int
#data3=mat(random.rand(2,2))
##这里的random模块使用的是numpy中的random模块，random.rand(2,2)创建的是一个二维数组，需要将其转换成#matrix
#data4=mat(random.randint(10,size=(3,3)))
##生成一个3*3的0-10之间的随机整数矩阵，如果需要指定下界则可以多加一个参数
#data5=mat(random.randint(2,8,size=(2,5)))
##产生一个2-8之间的随机整数矩
#data6=mat(eye(2,2,dtype=int))
##产生一个2*2的对角矩阵
#
#a1=[1,2,3]
#a2=mat(diag(a1))
#data1.T
#==============================================================================
# 
#==============================================================================
#
#m=20;n=40;
#A=np.mat(random.normal(0,1,(m,n)))
#u=SP.random(n,1,density=0.1)
#print(u)
#b=A*u
#print(b)
def Shrink(Arr,Thresh): # 传入参数Arr为np.arr形式数组, 暂时不支持np.matrix形式或者list等
    if Thresh<0 :
        print("Negative Threshold Error");
        return False
    else:
        ShkArr=zeros(Arr.shape)
        PosMask=Arr>Thresh
        NegMask=Arr<-Thresh
        #CenMask=~(PosMask | NegMask)
        ShkArr[PosMask]=Arr[PosMask]-Thresh
        ShkArr[NegMask]=Arr[NegMask]+Thresh
        return ShkArr
    
#==============================================================================
# 1-norm basis pursuit problem Proximal Gradient + Augmented Lagrangian
#==============================================================================
MaxIter=200;
MaxPGIter=500;
EPSPG=1E-15*n
EPS=1E-8*m
itr_alm=[]
x_alm=[]
#IsPGDecs=True;
mu0=2
tau0=0.2

x0=np.zeros((n,1));


lamd=mu0*(A*x0-b)#np.zeros((m,1));#np.random.normal(0,1,(m,1));## 并未想好初值
xcur=x0;
AtA=A.T*A
Atb=A.T*b
StepN=MaxIter
ConsResCurv=[0]*MaxIter
mu=mu0
for j in range(MaxIter): 
    ycur=xcur
    print("#",j," Augm Lagran Step \t")
    # Improv=[0]*MaxPGIter
    # Rang=MaxPGIter
    tau=tau0
    gradcur=A.T*lamd+mu*(AtA*ycur-Atb);#np.zeros(n,1)
    strt=time.clock()
    for i in range(MaxPGIter): 
        ytmp =array(ycur-tau*gradcur)
        ynxt =Shrink(ytmp, tau)
        dely=ynxt-ycur
        delg=array(mu*AtA*dely)
        gradcur=gradcur+delg
        ResNm=vdot(dely,dely)
        #vdot(dely,delg)
        tau=ResNm/vdot(dely,delg)
        # Improv[i]=la.norm(ynxt,ord=1)-la.norm(ycur,ord=1)
        ycur=ynxt
        if (ResNm<EPSPG) | (i==MaxPGIter-1):# Residue 范数不太大
            fin=time.clock()
            print("ProxGrad Iter num: ",i+1,'\t',"Time: ", fin-strt,"s")
            break
        else:
            # print(ynxt)
#            if i%100==0:
#                print(i,"infty norm",max(abs(ycur)))
            # input("")
            continue
    #plt.plot(range(Rang),Improv[0:Rang])
    xnxt=ycur
    ConscRes=A*xnxt-b# 此处若写成xcur则收敛不能, 会震荡很久
    ResNrm=la.norm(ConscRes)
    ConsResCurv[j]=ResNrm
    lamd=lamd+mu*ConscRes
    #mu=max(mu,mu0,0.05*n/ResNrm)
    if ConsResCurv[j]<EPS:
        StepN=j
        #print(xnxt)
        print("Optimized 1 norm",la.norm(xnxt,ord=1))
        print("Constraint Residue",la.norm(A*xnxt-b,ord=1))
        break
        #return xnxt;
    else:
        ObjImpr=la.norm(xnxt,ord=1)-la.norm(xcur,ord=1)
        xcur=xnxt
        print("Penalty mu ",mu," Objective Improve",ObjImpr," Residue of constraint: ",ResNrm)
    itr_alm.append(j)
    x_alm.append(la.norm(xnxt,ord=1))
plt.figure()    
plt.plot(ConsResCurv[0:StepN])
plt.figure()
plt.plot(xcur)




# In[245]:

plt.show()


# In[256]:

x_alm


# In[253]:

np.shape(itr_alm)


# In[ ]:



