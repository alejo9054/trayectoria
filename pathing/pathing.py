import numpy as np
# import imageio
import matplotlib.pyplot as plt
# import pandas as pd
#import sklearn.datasets
import cv2 as cv
# import glob

# import os

# Cargar arreglo NumPy desde un archivo
from numpy import loadtxt
# Cargar Arreglo
punto_inicio = loadtxt('punto_inicio.csv', delimiter=',',dtype=np.int)
# Imprimir el arreglo
print(punto_inicio)
punto_final = loadtxt('punto_final.csv', delimiter=',',dtype=np.int)
# Imprimir el arreglo
print(punto_final)
obstaculos = loadtxt('obstaculos.csv', delimiter=',',dtype=np.int)
# Imprimir el arreglo
print(obstaculos)

class pathing:
    def __init__(self, inicio, fin, obstaculos):
        self.inicio = inicio
        self.fin = fin
        self.obstaculos = obstaculos
        self.path=np.array([inicio])
    #metodos
        
    def generate(self):
        i=True
        while i==True:
        
            maskObs=self.maskObstaculos(self.obstaculos)
        
            maskInicioFin= self.maskLineaInicioFin(self.inicio , self.fin)
        
            inter=self.maskIntersecciones(maskInicioFin,maskObs)
        
            primer_obstaculo , numero_del_obstaculo=self.maskObstaculoMasCercano(inter,self.inicio,self.obstaculos)
        
            linea_I_O=self.maskLineaInicioObstaculo(self.inicio,self.obstaculos,numero_del_obstaculo)
        
            perpendicular=self.maskPerpendicular(self.inicio,self.obstaculos,numero_del_obstaculo)
        
            inter_primer_obs=self.maskIntersecciones(perpendicular,primer_obstaculo)
        
            maskPuntos,puntos_x,puntos_y=self.puntosExtremos(inter_primer_obs,self.obstaculos,numero_del_obstaculo)
        
            punto=self.puntoCercanoFinal(self.fin,puntos_x,puntos_y)
        
            i=self.iterarPuntos(punto,self.fin,self.obstaculos,numero_del_obstaculo)
        
        
        
        return path,self.path
        
        #mask2=linea_I_O+perpendicular
        plt.figure(figsize=(12,12))
        plt.imshow(maskPuntos, cmap='viridis')
#         plt.figure(figsize=(12,12))
#         plt.imshow(mask2, cmap='viridis')
        print('via points')
        print(punto==self.fin)
        
    def seg_color(self,src,bajos, altos):
        mask = cv.inRange(src, bajos, altos)
        return mask

    def dibujarObs(self,mask,x,y,radio):
        #mask=np.zeros((mask.shape[0],mask.shape[1]),dtype=np.uint8)
        #Dibujando un círculos
        cv.circle(mask,(x,y),radio,(150),-1)
        return mask
        #plt.figure(figsize=(12,12))
        #plt.imshow(mask, cmap='viridis')
    def maskObstaculos(self, obstaculos):
        mask = np.zeros((480,640),dtype=np.uint8)
        for i in obstaculos:
            mask = dibujarObs(mask,i[0],i[1],i[2])
            
        return mask
    
    def maskLineaInicioFin(self, punto_inicio, punto_final):
        mask = np.zeros((480,640),dtype=np.uint8)
        cv.line(mask,(punto_inicio[0],punto_inicio[1]),(punto_final[0],punto_final[1]),(255),2)
        return mask
    
    def maskIntersecciones(self, mask1, mask):
        diferencias=cv.subtract(mask1,mask)
        inters = np.array([105], dtype=np.uint8)
        mask = seg_color(diferencias, inters, inters)
        return mask
    
    def maskObstaculoMasCercano(self, intersecciones, punto_inicio,obstaculos):
        ##puntos de las intesecciones
        puntos_y, puntos_x=np.where(intersecciones)
        ##punto de interseccion mas cercano al inicio
        distancia = np.sqrt((puntos_x-punto_inicio[0])**2 + (puntos_y-punto_inicio[1])**2)#distancia euclidiana
        pos=np.where(distancia == min(distancia))
        
        ##punto mas cercano
        #maskInicio=dibujarObs(maskInicio,int(puntos_x[pos]),int(puntos_y[pos]),5)
        
        ##obstaculo al que le pertenece ese punto
        distanciaObs = np.sqrt((obstaculos[:,0]-puntos_x[pos])**2+(obstaculos[:,1]-puntos_y[pos])**2)
        posObs=np.where(distanciaObs == min(distanciaObs))#cual es el obstaculo
        
        #mascara del obstaculo correspondiente
        mask = np.zeros((480,640),dtype=np.uint8)
        obstaculo = obstaculos[posObs].flatten()
        mask= dibujarObs(mask,obstaculo[0],obstaculo[1],obstaculo[2])
                
        return mask, posObs[0]
    
    def maskLineaInicioObstaculo(self,punto_inicio,obstaculos,posObs):
        mask = np.zeros((480,640),dtype=np.uint8)
        cv.line(mask,(punto_inicio[0],punto_inicio[1]),(int(obstaculos[posObs,0]),int(obstaculos[posObs,1])),(255),2)
        return mask
    
    def vector_p(self,punto_inicial_x,punto_inicial_y,punto_final_x,punto_final_y):
        #establecer vector con los puntos
        vector_x=punto_inicial_x-punto_final_x
        vector_y=punto_inicial_y-punto_final_y
        vector=np.array([vector_x,vector_y])
        return vector
    
    def perpendicular_90(self,vector,escala=1):
        theta= np.deg2rad(90)
        matriz=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])*escala
        perpendicular=matriz.dot(vector)
        #perpendicular.dtype=int
        return perpendicular.astype(int)
    
    def perpendicular_270(self,vector,escala=1):
        theta= np.deg2rad(-90)
        matriz=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])*escala
        perpendicular=matriz.dot(vector)
        #perpendicular.dtype=int
        return perpendicular.astype(int)
    
    def dibujar_perpendiculares(self,centroide_obstaculo,perpendicular_90,perpendicular_270,tamaño_imagen=(480,640)):
        mask = np.zeros(tamaño_imagen,dtype=np.uint8)
        cv.line(mask,(centroide_obstaculo[0],centroide_obstaculo[1]),(centroide_obstaculo[0]+perpendicular_90[0],centroide_obstaculo[1]+perpendicular_90[1]),(255),2)
        cv.line(mask,(centroide_obstaculo[0],centroide_obstaculo[1]),(centroide_obstaculo[0]+perpendicular_270[0],centroide_obstaculo[1]+perpendicular_270[1]),(255),2)
        return mask
    
    def maskPerpendicular(self,inicio,obstaculos,numero_del_obstaculo):
        punto_inicial_x=inicio[0]
        punto_inicial_y=inicio[1]
        punto_final_x=int(obstaculos[numero_del_obstaculo,0])
        punto_final_y=int(obstaculos[numero_del_obstaculo,1])
        
        vector = self.vector_p(punto_inicial_x,punto_inicial_y,punto_final_x,punto_final_y)
        
        perp=self.perpendicular_90(vector,10)
        perp2=self.perpendicular_270(vector,10)
        centroide_obstaculo=obstaculos[numero_del_obstaculo,:2].flatten()
        
        mask=self.dibujar_perpendiculares(centroide_obstaculo,perp,perp2)
                
        return mask
    
    def puntosExtremos(self,maskInterseccion,obstaculos,numero_del_obstaculo):
                
        puntos_y, puntos_x=np.where(maskInterseccion)
        
        distancia = np.sqrt((puntos_x-int(obstaculos[numero_del_obstaculo,0]))**2 + (puntos_y-int(obstaculos[numero_del_obstaculo,1]))**2)
        posF=np.where(distancia == max(distancia))
        posF= np.array(posF[0])
        #print(posF[0])
        #print(puntosF_x[posF])
        #print(puntosF_y[posF])
        ##dibujando el puntos lejanos al centroide

        maskInterseccion=dibujarObs(maskInterseccion,int(puntos_x[posF[0]]),int(puntos_y[posF[0]]),5)
        maskInterseccion=dibujarObs(maskInterseccion,int(puntos_x[posF[1]]),int(puntos_y[posF[1]]),5)
        
        puntos_x=puntos_x[posF]
        puntos_y=puntos_y[posF]
        
        return maskInterseccion,puntos_x,puntos_y
    def puntoCercanoFinal(self,punto_final,puntos_x,puntos_y):
        #punto mas cercano a punto final
       
        distancia = np.sqrt((puntos_x-punto_final[0])**2+(puntos_y-punto_final[1])**2)
        numero_punto=np.where(distancia == min(distancia))
        punto_cercano_F=np.array([puntos_x[numero_punto[0]],puntos_y[numero_punto[0]]])
        
        return punto_cercano_F.flatten()
    
    def iterarPuntos(self,punto_cercano_F,punto_final,obstaculos,numero_del_obstaculo):
        if punto_cercano_F[0]==punto_final[0] and punto_cercano_F[1]==punto_final[1]:
            self.path=np.append([self.path,punto_cercano_F])
            i=False
            return i
        else:
            self.inicio=punto_cercano_F
            self.obstaculos=np.delete(obstaculos, numero_del_obstaculo, axis=0)
            self.path=np.append(self.path,punto_cercano_F)
            #path=pathing(self.inicio , self.fin , self.obstaculos)
            #path.generate()
            i=True
            return i
            
        
path=0        
path = pathing(inicio=punto_inicio , fin=punto_final , obstaculos=obstaculos)

x,y=path.generate()
    
