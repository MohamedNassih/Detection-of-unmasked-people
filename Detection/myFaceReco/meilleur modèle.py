from scipy import linalg
import time
import matplotlib.pyplot as plt
from imageio.v2 import imsave, imread
import glob
import numpy as np
pngs = glob.glob('./gallery/*.png')
image = imread(pngs[0])
heigh = len(image[:,0])
length= len(image[0])
print(heigh , length)
imgs = np.array([imread(i).flatten() for i in pngs])
moyenne = np.mean(imgs,0)
phi=imgs-moyenne
covreduit=np.dot(phi, phi.transpose())
valp,vectpreduit=np.linalg.eig(covreduit)
vectp=np.dot(phi.transpose(),vectpreduit)
poids=np.dot(phi,vectp)
#for i in range(len(imgs)):
    #imsave("eigenface/eigenface"+str(i)+".png",vectp[:,i].reshape(heigh,length))
#for k in range(len(imgs)):
    #for i in range(len(imgs)):
        #recon=moyenne+np.dot(poids[k,:i],vectp[:,:i].transpose())
        #imsave("reconst/img_"+str(k)+"_"+str(i)+".png",recon.reshape(heigh,length))
t1=time.time()
image_a_trouver = np.array(imread("a_tester/11.png").flatten())
a_trouver=plt.figure(1)
plt.imshow(image_a_trouver.reshape(heigh,length))
plt.title("A trouver")
phi2=image_a_trouver-moyenne
poids2= np.dot(phi2,vectp)
dist = np.min((poids-poids2)**2,axis=1)
indiceImg = np.argmin(dist)
mindist=np.sqrt(dist[indiceImg])
print("Image de : "+pngs[indiceImg])
found=plt.figure(2)
plt.imshow(imgs[indiceImg].reshape(heigh,length))
plt.title("C'est trouvé")
found.show()
if mindist <=2.0:
    print( "MATCH !")
else:
    print( "NO MATCH !")
t2=time.time()
ex=t2-t1
print(f'le temps d\'éxecution est : {ex:.2}ms')

    
    
    


