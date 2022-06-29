'''-----------------imports-------------------'''
import matplotlib.pyplot as plt
import numpy as np
import random as ran
from copy import *
from openpyxl import Workbook
from datetime import datetime

#C:/Users/Propriétaire/Documents/TIPE/base de données/Heart_Disease_Prediction.csv
#C:/Users/Propriétaire/Documents/TIPE/python/saves

'''----------------truc excel------------------'''

dt = datetime.now()

wb = Workbook()

'''---------------var globales----------------'''

reussitestr = np.genfromtxt("C:/Users/Propriétaire/Documents/TIPE/base de données/Heart_Disease_Prediction.csv", delimiter=",",skip_header=1,dtype=str)[:,-1]
reussite= reussitestr == 'Presence' #transformation des str en bool

data = np.genfromtxt("C:/Users/Propriétaire/Documents/TIPE/base de données/Heart_Disease_Prediction.csv", delimiter=",",skip_header=1)[:,0:-1]
#tableau ou les indices des lignes sont les patients, et les indices des colonnes les features (infos sur le patient)

'''-------------------classes-----------------'''

class noeud():
    def __init__(self,entropie=None,gauche=None,droite=None,gain=None,valeur=None,condi=None,feat=None):
        self.entropie=entropie
        self.gauche=gauche
        self.droite=droite
        self.gain=gain
        self.condi=condi
        self.feat=feat

        self.valeur=valeur #si feuille
    def hauteur(self):
        if self.valeur==None:
            return max(self.gauche.hauteur(),self.droite.hauteur())+1
        else:
            return 0

class arbre():
    def __init__(self,racine=None,lId=[],precision=0):
        self.racine=racine
        self.lId=lId
        self.precision=precision

    def reponse(self,ligne):
        return parcour(ligne,self.racine)

    def accuracy(self,dat,reus):#different de la precision de la presentation, ici c'est la probabilité que l'algorithme retourne un resultat correct
        n=len(reus) #possible car reus est a 1 dimension
        acc=0
        for i in range(n):
            if self.reponse(ligne=dat[i])==reus[i]:
                acc+=1
        acc/=n
        return acc
    def hauteur(self):
        return self.racine.hauteur()

class foret():
    def __init__(self,tree=[]):
        self.tree=tree
    def vote(self,ligne,poid):
        n=len(self.tree)
        oui=0
        m=0
        for i in range(n):
            x=(self.tree[i].precision)
            if poid:
                if self.tree[i].reponse(ligne):
                    oui+=x-0.5
                m+=x-0.5
            else :
                if self.tree[i].reponse(ligne):
                    oui+=1
                m+=1
        return oui/m
    def reponse(self,ligne,poid):
        return self.vote(ligne,poid)>=0.5

    def accuracy(self,dat,reus,poid=True): #different de la precision de la presentation, ici c'est la probabilité que l'algorithme retourne un resultat correct
        n=len(reus) #possible car reus est a 1 dimension
        acc=0
        for i in range(n):
            if self.reponse(dat[i],poid)==reus[i]:
                acc+=1
        acc/=n
        return acc

'''----------------functions-----------------'''

def parcour(l,nod): #ou l est une ligne (les constantes d'un patient) et nod un noeud
    if nod.valeur==None :
        if l[nod.feat]<=nod.condi:
            return parcour(l,nod.gauche)
        else :
            return parcour(l,nod.droite)
    else :
        return nod.valeur
#parcour recursivement un DTC (decision tree classifier)



def entrop(l,reus): #ou l est une liste d'indices et reuss une liste de bool
    n=len(l)
    nbvrai=0
    for i in range(n):
        if reus[l[i]]:
            nbvrai+=1
    if nbvrai==0 or nbvrai==n:
        return 0
    else :
        p1=nbvrai/n
        p2=1-p1
        return (-p1*np.log2(p1))-(p2*np.log2(p2))

def infogain(prev,l1,l2,reus):
    en1,en2=entrop(l1,reus),entrop(l2,reus)
    n1,n2=len(l1),len(l2)
    return prev-((n1/(n1+n2))*en1)-((n2/(n1+n2))*en2)

#-----------------------version naive-----------------------------

def aprentissageDTC(dat,reus,lId,tmax): #reus[i] correspond a la presence ou non de maladie du patient n°i, dat[i] est l'ensemble des infos du patient i
    l,c=np.shape(dat)

    prevent=entrop([i for i in range(l)],reus)
    if len(reus)==0: #normalement impossible, mais au cas ou, sert a eviter que le if qui suit renvoit une erreur
        return noeud(valeur="rien")
    if prevent==0:
        return noeud(valeur=reus[0])

    if tmax==0 :
        return noeud(valeur=best(reus))

    cond=None
    col=None
    bestig=0
    gau,dro=[],[]
    for i in range(l):
        for j in range(c):
            x0=dat[i,j]
            oui,non=[],[] #arbres fils theoriques

            for k in range(l):
                if dat[k,j]<=x0:
                    oui.append(k)
                else :
                    non.append(k)
            ig=infogain(prevent,oui,non,reus)
            if ig>bestig:
                bestig=ig
                cond=x0
                col=j
                gau,dro=oui,non

    if col==None : #ca veut dire qu'il n'y a pas de condition pour lesquelles le gain d'info serait maximal, dans ce cas, on prend la majorité
        return noeud(valeur=best(reus))

    datg,datd=dat[[i for i in gau]],dat[[i for i in dro]]
    reusg,reusd=reus[[i for i in gau]],reus[[i for i in dro]]
    a=noeud(entropie=prevent,gauche=aprentissageDTC(datg,reusg,lId,tmax-1),droite=aprentissageDTC(datd,reusd,lId,tmax-1),gain=bestig,condi=cond,feat=lId[col])
    return a
#apprentissage d'un arbre DTC
#on test toutes les features de tous les patients (for i et for j)et on regarde pour chacune d'entre elles qui sont les patients qui vont a gauche et a droite (for k).la configuration apportant le meilleur gain d'information est gardé (bestig)
#complexité totale : O(l^2 *c *tmax). peut surement etre amelioré, mais pas important car c'est un apprentissage

#-------------------------------------------------------------

def aprentissageDTCopti(dat,reus,lId,tmax): #reus[i] correspond a la presence ou non de maladie du patient n°i, dat[i] est l'ensemble des infos du patient i
    l,c=np.shape(dat)

    prevent=entrop([i for i in range(l)],reus)
    if len(reus)==0: #normalement impossible, mais au cas ou, sert a eviter que le if qui suit renvoit une erreur
        return noeud(valeur="rien")
    if prevent==0:
        return noeud(valeur=reus[0])

    if tmax==0 :
        return noeud(valeur=best(reus))

    cond=None
    col=None
    bestig=0
    gau,dro=[],[]
    ind=tabtrie(dat)

    for j in range(c):
        for i in range(l):
            x0=dat[i,j]

            nbg,nbd=0,0
            for k in range(l):
                if dat[k,j]<=x0:
                    oui.append(k)
                    if reus[k]:
                        nbg+=1
                else :
                    non.append(k)
                    if reus[k]:
                        nbd+=1
            ng,nd=len(oui),len(non) #calcul du gain d'information (legerement plus rapide que si on utilise infogain; la complexité est divisée par 2 normalement)
            if nbg==0 or nbg==ng:
                en1= 0
            else :
                p1g=nbg/ng
                p2g=1-p1g
                en1=(-p1g*np.log2(p1g))-(p2g*np.log2(p2g))

            if nbd==0 or nbd==nd:
                en2= 0
            else :
                p1d=nbd/nd
                p2d=1-p1d
                en2=(-p1d*np.log2(p1d))-(p2d*np.log2(p2d))
            ig=prevent-((ng/l)*en1)-((nd/l)*en2)
            #ig=infogain(prevent,oui,non,reus)
            if ig>bestig:
                bestig=ig
                cond=x0
                col=j
                gau,dro=oui,non

    if col==None : #ca veut dire qu'il n'y a pas de condition pour lesquelles le gain d'info serait maximal, dans ce cas, on prend la majorité
        return noeud(valeur=best(reus))

    datg,datd=dat[[i for i in gau]],dat[[i for i in dro]]
    reusg,reusd=reus[[i for i in gau]],reus[[i for i in dro]]
    a=noeud(entropie=prevent,gauche=aprentissageDTC(datg,reusg,lId,tmax-1),droite=aprentissageDTC(datd,reusd,lId,tmax-1),gain=bestig,condi=cond,feat=lId[col])
    return a

def aprentissageRTF(dat,reus,tmax,datp,reusp,narbres=25,nfeats=4,randomdata=True):
    n,f=np.shape(dat)
    forest=[]
    if randomdata:
        for i in range(narbres):
            ranId=[]#id des lignes
            testId=[]
            for j in range(n):
                ranId.append(ran.randint(0,n-1))
                testId.append(ran.randint(0,n-1))
            rdat,rreus=dat[ranId],reus[ranId]
            tdat,treus=dat[testId],reus[testId]
            fId=ran.sample(range(0,f),nfeats)
            fdat=rdat[:,fId]
            a=arbre(racine=aprentissageDTC(fdat,rreus,fId,tmax),lId=fId)
            a.precision=a.accuracy(datp,reusp)
            forest.append(a)
            # print(str(i+1)+"/"+str(narbres)+" - hauteur="+str(forest[i].hauteur())+" - feats="+str(fId)+" - precision="+str(forest[i].precision))
        return foret(tree=forest)




def best(reus):
    n=len(reus)
    t=0
    for i in range(n):
        if reus[i]:
            t+=1
    if t==n/2 :
        return "rien"
    else :
        return t>n/2


def noeudtolist(nod):
    if nod.valeur==None:
        return [nod.condi,nod.feat,noeudtolist(nod.gauche),noeudtolist(nod.droite)]
    else :
        return nod.valeur
#cette fonction sert juste à mieux visualiser un arbre
def arbretolist(arb):
    return noeudtolist(arb.racine)

def split(dat,reus,part=0.85):
    n=len(reus)
    m=int(n*part)
    lId=ran.sample(range(0,n-1),m)
    antdat,antreus=np.delete(dat,lId,0),np.delete(reus,lId,0)
    return dat[lId],reus[lId],antdat,antreus


#----------------------------------------------------------------


def resultrtf(dat,reus,na=25,nf=4,nb=40,bruit=False): #na: nb d'arbres   nf: nb de features      nb:nb de forets
    l=[]
    lp=[]

    for i in range(nb):
        datg,reusg,datp,reusp=split(dat,reus)
        if bruit:
            datg=bruitdat(datg)
            reusg=bruitreus(reusg)
        print('------------foret : '+str(i)+'/'+str(nb)+' -------------(feats='+str(nf)+')')
        a=aprentissageRTF(datg,reusg,20,datp,reusp,na,nf)
        rp=a.accuracy(datp,reusp)
        r=a.accuracy(datp,reusp,False)
        print('avg='+str(r))
        print('avg p='+str(rp))
        l.append(r)
        lp.append(rp)
    return (l,lp)


#----------------------------------------------------------------


def megaanalysertf(dat,reus,t=5):
    n,f=np.shape(dat)

    sheet = wb.active
    sheet.title = 'data'
    l=[]
    lp=[]
    for i in range(1,f):
        (a,b)=resultrtf(dat,reus,nf=i)
        l.append(a)
        lp.append(b)
    excel = np.array(l+lp)
    np.savetxt('C:/Users/Propriétaire/Documents/TIPE/python/saves/megaanalysertf_' + dt.strftime("%Y%m%d_%I%M%S") + '.xlsx', excel, delimiter=',')



def moyenne(l):
    n=len(l)
    x=0
    for i in range(n):
        x+=l[i]
    return x/n

def moylist(l):
    n=len(l)
    p=[]
    for i in range(n):
        p.append(moyenne(l[i]))
    return p

def bruitdat(dat,fac):
    l,c=np.shape(dat)
    dat2=np.zeros((l,c))
    for i in range(l):
        for j in range(c):
            u=dat[i,j]
            if isinstance(u,int):
                r=ran.random()
                if u<2 and r<fac:
                    dat2[i,j]=u+1-2*u
                else:
                    dat2[i,j]=int(x+fac(1-2*r))
            else:
                dat2[i,j]=x+fac(1-2*r)

    return dat2

def bruitreus(reus,fac=0.05):
    l=len(reus)
    reus2=np.zeros(l)
    for i in range(l):
        r=ran.random()
        if r<fac :
            reus2=not reus[i]
    return reus2


def fusion(t,g,m,d,ind):
    i,j=g,m
    for k in range(g,d):
        if i<m and (j==d or t[i]<=t[j]):
            ind[k]=i
            i+=1
        else:
            ind[k]=j
            j+=1
def trifurec(t,g,d,ind):
    if d>g+1:
        m=(g+d)//2
        trifurec(t,g,m,ind)
        trifurec(t,m,d,ind)
        fusion(t,g,m,d,ind)
def trifusion(t):
    tc=deepcopy(t)
    ind=[0]*len(t)
    trifurec(tc,0,len(t),ind)
    return ind
def tabtrie(dat):
    l,c=np.shape(dat)
    L=[]
    for i in range(c):
        L.append(trifusion(dat[:,i]))
    return np.array(L).T


'''------------------main--------------------'''

# arb=arbre(racine=aprentissageDTC(data,reussite,[i for i in range(13)]))
# larb=arbretolist(arb)
#
# np.save("C:/Users/Propriétaire/Documents/TIPE/python/saves/arbreComplet",[arb])
#
# test=np.load("C:/Users/Propriétaire/Documents/TIPE/python/saves/arbreComplet.npy",allow_pickle=True)[0]
# ltest=arbretolist(test)



# datag,reussiteg,datap,reussitep=split(data,reussite,part=0.85)
# bois=aprentissageRTF(datag,reussiteg,15,datap,reussitep,25,4)
# print(bois.accuracy(datap,reussitep,False))
# print(bois.accuracy(datap,reussitep,True))
# np.save("C:/Users/Propriétaire/Documents/TIPE/python/saves/foret1",[bois])





#print(open("C:/Users/Propriétaire/Documents/TIPE/base de données/Heart_Disease_Prediction.csv").read())