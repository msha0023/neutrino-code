# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
import math
import scipy.integrate as integrate 
from scipy.interpolate import RectBivariateSpline
import kepdump





class Neutrino(object):
    files = ["Spec-BremSi-e", "Spec-BremSi-eb","Spec-BremSi-x","Spec-BremSi-x","Spec-Pair-e","Spec-Pair-eb",
             "Spec-Pair-x","Spec-Pair-x","Spec-Pair-xb","Spec-Pair-xb","Spec-Phot-e","Spec-Phot-eb",
             "Spec-Phot-x","Spec-Phot-x","Spec-Phot-xb","Spec-Phot-xb","Spec-Plas-e","Spec-Plas-e",
             "Spec-Plas-x","Spec-Plas-x","Spec-Plas-x","Spec-Plas-x","Spec-BremSi-xb","Spec-BremSi-xb"]
    fortranFile = 'output.txt'
    photoFortran = "photo.txt"
    pairFortran = "pair.txt"
    bremFortran = "bremsi.txt"
    plasFortran = "plasma.txt"
    recombFortran = "recombination.txt"
    recombTot = "recombTotal.txt"
    labels = ['BremFe e n', 'BremFe anti n e', 'BremFe mu n', 'BremFe tau n','Pair e n', 
              'Pair e anti n', 'Pair mu n', 'Pair tau n','Pair mu anti n', 'Pair tau anti n','Photo e n', 
              'Photo anti e n','Photo mu n', 'Photo tau n','Photo mu anti-n',
              'Photo tau anti n', 'Plasma e n', 'Plasma anti e n','Plasma mu n','Plasma anti mu n', 'Plasma tau n',
              'Plasma anti tau n','BremSi mu anti n','BremSi tau anti n']


    def __init__(self):
        self.e = []
        for fn,lb in zip(self.files, self.labels): #loops to read all files and associated labels
            t,d,e = self.readfileEnergy(fn)
            self.e.append(e)
        self.e = np.array(self.e)
        self.t = t
        self.d = d
        self.n = len(self.files)

        f = self.readFortranFile(self.fortranFile)
        self.f = f
        
        photof = self.readFortranFile(self.photoFortran)
        self.photof = photof
        pairf = self.readFortranFile(self.pairFortran)
        self.pairf = pairf
        bremf = self.readFortranFile(self.bremFortran)
        self.bremf = bremf
        plasf = self.readFortranFile(self.plasFortran)
        self.plasf = plasf
        
        recombf = self.readFortranFile(self.recombFortran)
        self.recombf = recombf
        
        retot = self.readFortranFile(self.recombTot)
        self.retot = retot
        
        self.E = 1e-4 * 10**(0.02 * np.arange(e.shape[2]))
        
        mass, density, logd, logt, z2a, factor = self.keplerdata()
        self.mass = mass
        self.density = density
        self.logd = logd
        self.logt = logt  
        self.z2a = z2a
        self.factor = factor
        
    @staticmethod         
    def keplerdata():
        d = kepdump.load('/home/megha/kepler/s15/s15#cign')
        Z = np.array([x.Z for x in d.abu.ions[:,0]])
        z2a = np.sum(d.ppn[:,1:-1] * (Z[:-1,np.newaxis]**2), axis=0)
        factor = z2a /((14**2)/28) #finding the scaling factor 
        print(factor)
        temp = d.tn[1:-1]
        logt = np.log10(temp) #taking the log of the temp to make it in similar form as the gang's temp.
        print(logt) #Kelvin
        density = d.dn[1:-1] #g/cm^3
        logd = np.log10(density*0.5)
        print(logd)
        mass = d.xm[1:-1] #gram
        print(mass)
        return mass, density, logd, logt, z2a, factor
        
        
        
    @staticmethod    
    def readfileEnergy(file_name): #opens file to read energy values
        data = np.loadtxt(file_name)
        y = np.reshape(data,(41,101,-1))
        e = y[:,:,2:] #extrats the psi(Energy) values  
        t = y[:,0,0] #Extracts the log(temperature) values 
        d = y[0,:,1] #extracts the log(density) values

        return t, d, e
    @staticmethod      
    def readFortranFile(file_name):
        f =  np.loadtxt(file_name)
        
        return f
    
    
    def interpolationfunction(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        BremtotArea = [] # a list of 41 temperature values, containiting sub-lists of length 101 ( energy for 101 density values )
        OtherTotalArea = []
        for i in range(len(self.t)): #Loop for finding all area values for all temperatures at different densities
            
            data = self.e[:, i].copy()
            data *= self.E 
            area = np.sum((data[:,:,1:] + data[:,:,:-1]) * (self.E[1:] -self.E[:-1]), axis=-1) * 0.5
            area *= 1.60218e-6 #in ergs
            add=0
            for i in [0,1,2,3,22,23]: #adding  brem process for same temp
                add = area[i] + add #units on y axis as ergs/cm^3 
                    
            BremtotArea.append(add)
            
            add1 = 0
            for k in [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]: #adding phot, plas, pair
                add1 = area[k]+add1
            OtherTotalArea.append(add1)                
        print(self.logt.shape)
        print(self.logd.shape)
        print(len(OtherTotalArea))
        x = np.array(BremtotArea).reshape(41, 101) #reshaping the brem total area for all temperatures and denisty
        y = np.array(OtherTotalArea).reshape(41, 101)
        print(x.shape)
        totArea = x+y #adding brem and other processes contribution
        #print(totArea[0])
        splBrem = RectBivariateSpline(np.array(self.t), np.array(self.d), x, kx=1, ky=1) #interpolating the brem data
        splOther = RectBivariateSpline(np.array(self.t), np.array(self.d), y, kx=1, ky=1) #interpolating the other processes data
        spl = splBrem(self.t, self.d) + splOther(self.t,self.d) #adding the interpolation output for gang's data
        # getting output for brem process for kepler code using the brem interpolation
        brem = splBrem.ev(self.logt, self.logd)*self.factor #multiplied with factor to scale the each zone brem energy output. 
        
        other = splOther.ev(self.logt, self.logd) #geting the output for other processes for kepler code using the other interpolation
        totalLoss = brem + other #adding the interpolation output for all processes for kepler code
        print(len(totalLoss))
        print(totalLoss.shape)
        
        
        ax.plot(self.logd, totalLoss, label = "Kepler for each zone")
        #ax.plot(self.d, spl[20], label = "interpolate ") 
        #ax.plot(self.d, totArea[20], label = "python ") 
        ax.set_title(rf"Total Energy loss Kepler")
        ax.set_ylabel("Energy[erg/cm$^3$]")
        ax.set_yscale('log')

        ax.legend(loc='best')
        ax.set_xlabel("log(density/mu e)[g/cm$^3$]")
        fig.show()
        fig.savefig("Area pair.pdf") 
    
        
    def energyDEn(self, tf=True, ti=0): #for plotting 1987 fig 4
        #print(self.e.shape)
            
        fig = plt.figure()
        ax = fig.add_subplot(111)
       
        data = self.e[:, ti].copy() #no of neutrinos/MeV/cm^3
        print(data.shape)
        #This calculated the no of neutrinos/cm^3
        areaNo = np.sum((data[:,:,1:] + data[:,:,:-1]) * (self.E[1:] -self.E[:-1]), axis=-1) * 0.5 #no of neutrinos/cm^3
        # print(data[:,:,:-1].shape)
        add2 = []#list of cm^-3 for all 4 processes. 
        bremsi2 = (areaNo[0]+areaNo[1]+areaNo[2]+areaNo[3]+areaNo[22]+areaNo[23])
        add2.append(bremsi2)
        pair2 = (areaNo[4]+areaNo[5]+areaNo[6]+areaNo[7]+areaNo[8]+areaNo[9])
        add2.append(pair2)
        phot2 = (areaNo[10]+areaNo[11]+areaNo[12]+areaNo[13]+areaNo[14]+areaNo[15])
        add2.append(phot2)
        plas2 = (areaNo[16]+areaNo[17]+areaNo[18]+areaNo[19]+areaNo[20]+areaNo[21]) 
        add2.append(plas2)
        #print(pair)
        
        data1 = self.e[:, ti].copy()
        data1 *= self.E                
        area = np.sum((data1[:,:,1:] + data1[:,:,:-1]) * (self.E[1:] -self.E[:-1]), axis=-1) * 0.5 #this calculates the no of neutrinos*MeV/cm^3

        #area *= 1.60218e-6
        add=[]
        bremsi = (area[0]+area[1]+area[2]+area[3]+area[22]+area[23]) *1.60218e-6
        add.append(bremsi)
        '''print("BremSi")
        print(bremsi)'''
        pair = (area[4]+area[5]+area[6]+area[7]+area[8]+area[9]) 
        add.append(pair)
        phot = (area[10]+area[11]+area[12]+area[13]+area[14]+area[15])
        add.append(phot)
        plas = (area[16]+area[17]+area[18]+area[19]+area[20]+area[21])
        add.append(plas)
        x = pair+plas+phot
        xn = x*1.60218e-6
        print(xn)        
        #p.array(add)
        #np.array(add2)
        energy = np.array(add)/np.array(add2) #energy for y axis as MeV
        energy1=[]
        for i in energy:
            x = i*(10**3)
            energy1.append(x)
        lb = ["BremSi", "Pair", "Phot", "Plas"]   
        for i in range(len(energy)): #loops over the add file and plots the enegy sums 
            ax.plot(10**(self.d), energy1[i], label = lb[i])
        
            
        ax.set_yscale('log')
        ax.set_xscale('log')
        #plt.ylim(1e1, 1e3)
        
        ax.set_title(rf"Total Energy for log(T/K) = {self.t[ti]:5.2f}")
        ax.set_ylabel("Energy[KeV]")
        ax.legend(loc='best')
        ax.set_xlabel("density/mu e[g/cm$^3$]")
        fig.show()
        fig.savefig("Total energy plot .pdf")
        return xn
        
    def plot_all_Types(self,tf = True, ti=0): #this code plots pair, plasma, brem and phot separately. 
        data1 = self.e[:, ti].copy()
        data1 *= self.E              
        area= np.sum((data1[:,:,1:] + data1[:,:,:-1]) * (self.E[1:] -self.E[:-1]), axis=-1) * 0.5
        
        area *= 1.60218e-6
        fig = plt.figure()
        ax = fig.add_subplot(111)      
 
        #print(bremc)
        
        bremsi = (area[0]+area[1]+area[2]+area[3]+area[22]+area[23])
        breml = []
        for i in range(len(bremsi)):
            if bremsi[i]==0:
                breml.append(bremsi[i])
            else:
                x = math.log10(bremsi[i])
                breml.append(x)
        ax.plot(self.d, breml, label = "bremSi ")
        #ax.plot(self.d, np.log(bremsi), label = "bremC ")
        
        pair = (area[4]+area[5]+area[6]+area[7]+area[8]+area[9]) #this adds all the energies for pair files
        #pairl = []
        #for i in range(len(pair)):
        #    if pair[i]==0:
        #        pairl.append(pair[i])
        #    else:
        #        x = math.log10(pair[i])
        #        pairl.append(x)
        pairl = np.log10(np.maximum(pair,1.e-99))         
                
             
        ax.plot(self.d, pairl, label = "pair")
        phot = (area[10]+area[11]+area[12]+area[13]+area[14]+area[15])#this adds all the energies for phot files
        photl = []
        for i in range(len(phot)):
            if phot[i]==0:
                photl.append(phot[i])
            else:
                x = math.log10(phot[i])
                photl.append(x)
        ax.plot(self.d, photl, label = "photo ")
        plas = (area[16]+area[17]+area[18]+area[19]+area[20]+area[21])#this adds all the energies for plasma files
        plasl = []
        for i in range(len(plas)):
            if plas[i]==0:
                plasl.append(plas[i])
            else:
                x = math.log10(plas[i])
                plasl.append(x)
        ax.plot(self.d, plasl, label = "plas ")        
        ax.set_title(rf"Total Energy for log(T/K) using Python = {self.t[ti]:5.2f}")
        ax.set_ylabel("log(Energy)[erg/cm$^3$]")
        #ax.set_yscale('log')
        plt.xticks(np.arange(min(self.d), 16))
        plt.ylim(15, 55)
        
        '''ax.plot(self.d, np.log(self.bremf), label = "Fortran brem Si",linestyle = '--') 
        ax.plot(self.d, np.log(self.photof), label = "Fortran photo",linestyle = '--')
        ax.plot(self.d, np.log(self.plasf), label = "Fortran plas",linestyle = '--')  
        ax.plot(self.d, np.log(self.pairf), label = "Fortran pair",linestyle = '--') '''
        ax.legend(loc='best')
        ax.set_xlabel("log(density/mu e)[g/cm$^3$]")
        fig.show()
        fig.savefig("Area pair.pdf") 
        
        
    def plot_Brem(self, tf = True, ti=0): #plots both fortran and python brem on one plot

        data1 = self.e[:, ti].copy()
        data1 *= self.E                
        area = np.sum((data1[:,:,1:] + data1[:,:,:-1]) * (self.E[1:] -self.E[:-1]), axis=-1) * 0.5
        
        print(area.shape)
        area *= 1.60218e-6
        fig = plt.figure()
        ax = fig.add_subplot(111)
        bremsi = (area[0]+area[1]+area[2]+area[3]+area[22]+area[23])
        print(bremsi)#this adds all the energies for bremSi files
        bremc = []
        for i in bremsi:
            x =  i*((6**2)*28)/((14**2)*12)
            bremc.append(x)
        #print(bremc)
        ax.plot(self.d, bremsi, label = "Python brem Si",color = 'blue')
        ax.plot(self.d, bremc, label = "Python brem C", color='red')
        #ax.plot(self.d, self.bremf, label = "Fortran brem C",color="green")
        ax.set_title(rf"Total Energy for log(T/K) = {self.t[ti]:5.2f}")
        ax.set_ylabel("Energy[erg/cm$^3$]")
        ax.legend(loc='best')
        ax.set_yscale('log')
        ax.set_xlabel("log(density/mu e)[g/cm$^3$]")
        fig.show()
        fig.savefig("Area brem.pdf")   
        
    def plot_Pair(self, tf = True, ti=0): 
        
        data1 = self.e[:, ti].copy()
        data1 *= self.E              
        area = np.sum((data1[:,:,1:] + data1[:,:,:-1]) * (self.E[1:] -self.E[:-1]), axis=-1) * 0.5
        area *= 1.60218e-6
        fig = plt.figure()
        ax = fig.add_subplot(111)
        pair = (area[4]+area[5]+area[6]+area[7]+area[8]+area[9]) #this adds all the energies for pair files
        ax.plot(self.d, pair, label = "Python pair contribution")
        ax.plot(self.d, self.pairf, label = "Fortran pair")
        ax.set_title(rf"Total Energy for log(T/K) = {self.t[ti]:5.2f}")
        ax.set_ylabel("Energy[erg/cm$^3$]")

        ax.legend(loc='best')
        ax.set_xlabel("log(density/mu e)[g/cm$^3$]")
        fig.show()
        fig.savefig("Area pair.pdf") 
        
    def plot_Phot(self, tf = True, ti=0):     
         
        data1 = self.e[:, ti].copy()
        data1 *= self.E                
        area = np.sum((data1[:,:,1:] + data1[:,:,:-1]) * (self.E[1:] -self.E[:-1]), axis=-1) * 0.5
        area *= 1.60218e-6
        fig = plt.figure()
        ax = fig.add_subplot(111)
        phot = (area[10]+area[11]+area[12]+area[13]+area[14]+area[15])#this adds all the energies for phot files
        ax.plot(self.d, phot, label = "Python photo contribution")
        ax.plot(self.d, self.photof, label = "Fortran photo")
        ax.set_title(rf"Total Energy for log(T/K) = {self.t[ti]:5.2f}")
        ax.set_ylabel("Energy[erg/cm$^3$]")
        ax.legend(loc='best')
        ax.set_yscale('log')
        ax.set_xlabel("log(density/mu e)[g/cm$^3$]")
        fig.show()
        fig.savefig("Area phot.pdf")   
        
    def plot_Plas(self, tf = True, ti=0): 
         
        data1 = self.e[:, ti].copy()
        data1 *= self.E                
        area = np.sum((data1[:,:,1:] + data1[:,:,:-1]) * (self.E[1:] -self.E[:-1]), axis=-1) * 0.5
        area *= 1.60218e-6
        fig = plt.figure()
        ax = fig.add_subplot(111)       
        plas = (area[16]+area[17]+area[18]+area[19]+area[20]+area[21])#this adds all the energies for plasma files
        ax.plot(self.d, plas, label = "Python plas contribution")
        ax.plot(self.d, self.plasf, label = "Fortran plas")
        ax.set_title(rf"Total Energy for log(T/K) = {self.t[ti]:5.2f}")
        ax.set_ylabel("Energy[erg/cm$^3$]")
        ax.legend(loc='best')
        ax.set_yscale('log')
        ax.set_xlabel("log(density/mu e)[g/cm$^3$]")
        
        fig.show()
        fig.savefig("Area plasma.pdf")        
    def plot_Recomb(self, ti=0):
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        '''data1 = self.e[:, ti].copy()
        data1 *= self.E              
        area = np.sum((data1[:,:,1:] + data1[:,:,:-1]) * (self.E[1:] -self.E[:-1]), axis=-1) * 0.5
        area *= 1.60218e-6
        bremsi = (area[0]+area[1]+area[2]+area[3]+area[22]+area[23])
        ax.plot(self.d, bremsi, label = "bremSi ",linestyle = '--')
        pair = (area[4]+area[5]+area[6]+area[7]+area[8]+area[9]) #this adds all the energies for pair files
        ax.plot(self.d, pair, label = "pair",linestyle = '--')
        phot = (area[10]+area[11]+area[12]+area[13]+area[14]+area[15])#this adds all the energies for phot files
        ax.plot(self.d, phot, label = "photo ",linestyle = '--')
        plas = (area[16]+area[17]+area[18]+area[19]+area[20]+area[21])#this adds all the energies for plasma files
        ax.plot(self.d, plas, label = "plas ",linestyle = '--')  '''
        ax.plot(self.d, np.log(self.bremf), label = "Fortran brem Si",linestyle = '--') 
        ax.plot(self.d, np.log(self.photof), label = "Fortran photo",linestyle = '--')
        ax.plot(self.d, np.log(self.plasf), label = "Fortran plas",linestyle = '--')  
        ax.plot(self.d, np.log(self.pairf), label = "Fortran pair",linestyle = '--')       
        '''ax.plot(self.d, self.recombf, label = "Recombination process")
        ax.plot(self.d,self.f , label = "Sum of different neutrinos energies fortran (without recomb process)", color="black",linestyle = "--")
        ax.plot(self.d,self.retot , label = "Sum of different neutrinos energies fortran (with recomb process)", color="red")        
        ax.set_title(rf"Total Energy for log(T/K) = {self.t[ti]:5.2f}")'''
        ax.set_ylabel("Energy[erg/cm$^3$")
        ax.set_xlabel("log(density/mu e)[g/cm$^3$]")
        plt.xticks(np.arange(min(self.d)-1, max(self.d)+1, 1.0))
        plt.ylim(-25, 15)
        ax.legend(loc='best')
        fig.show()
        fig.savefig("Area phot.pdf")
        
        
    def plot_total(self, di=0):

            
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #hbar = 
        m = 9.1e-28 #grams
        

        averageTot = []
        for ti in range(len(self.t)):
               
            
            data1 = self.e[:, ti].copy()
            data1 *= self.E   #cm^-3             
            area = np.sum((data1[:,:,1:] + data1[:,:,:-1]) * (self.E[1:] -self.E[:-1]), axis=-1) * 0.5 #units MeV/cm^3
            add = 0
            for i in range(len(area)):
                tot = add + area[i]
                
            data = self.e[:, ti].copy() #MeV^-1 cm^-3
            areaNo = np.sum((data[:,:,1:] + data[:,:,:-1]) * (self.E[1:] -self.E[:-1]), axis=-1) * 0.5 #units cm^-3
            addNo = 0
            for i in range(len(areaNo)):
                number = addNo + areaNo[i]
            energy = np.array(tot[di])/np.array(number[di]) #Calculating the average energy #unit = MeV
                
            
            energy *= 10**3
            averageTot.append(energy) 
            #e = (hbar**2/(2*m))*((3*(3.14**2)*number[di])
                
            ax.plot(self.t, averageTot, label = rf"at log(density/mu e) = {self.d[di]:5.2f}")
            ax.set_yscale('log')
            
        
            ax.set_title("Total Energy all temperatures for all flavours of neutrinos ")
            ax.set_ylabel("Energy[KeV]")
            ax.legend(loc='best')
            ax.set_xlabel("log(Temp) [K]")
            fig.show()
            fig.savefig("Area plots  .pdf")
 
    def plot_electron_n(self, di=0):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        averageTote = []
        averageTotm = []
        averageTott = []
        for ti in range(len(self.t)):
            data1 = self.e[:, ti].copy()
            data1 *= self.E   #cm^-3             
            area = np.sum((data1[:,:,1:] + data1[:,:,:-1]) * (self.E[1:] -self.E[:-1]), axis=-1) * 0.5
            

            electron = (area[0]+area[1]+area[4]+area[5]+area[10]+area[11]+area[16]+area[17])#this is for adding MeV/cm^3 for all processes that produce electron neutrino and anti-neutrino
            muon = (area[2]+area[6]+area[8]+area[12]+area[14]+area[18]+area[20]+area[22])
            tau = (area[3]+area[7]+area[9]+area[13]+area[15]+area[18]+area[21]+area[23])
            #print(electron.shape)
            data = self.e[:, ti].copy() #MeV^-1 cm^-3
            areaNo = np.sum((data[:,:,1:] + data[:,:,:-1]) * (self.E[1:] -self.E[:-1]), axis=-1) * 0.5
            
            numbere = areaNo[0]+areaNo[1]+areaNo[4]+areaNo[5]+areaNo[10]+areaNo[11]+areaNo[16]+areaNo[17]
            numberm = (areaNo[2]+areaNo[6]+areaNo[8]+areaNo[12]+areaNo[14]+areaNo[18]+areaNo[20]+areaNo[22])
            numbert = (areaNo[3]+areaNo[7]+areaNo[9]+areaNo[13]+areaNo[15]+areaNo[18]+areaNo[21]+areaNo[23])
            energye = np.array(electron[di])/np.array(numbere[di]) #Calculating the average energy for a particular density
            energym = np.array(muon[di])/np.array(numberm[di])
            energyt = np.array(tau[di])/np.array(numbert[di])
            
           
           
            energye *= 10**3
            energym *= 10**3
            energyt *= 10**3
            averageTote.append(energye)
            averageTotm.append(energym)
            averageTott.append(energyt)
    
        ax.plot(self.t, averageTote, label = "Python electron neutrino contribution")
        ax.plot(self.t, averageTotm, label = "Python muon neutrino contribution")
        ax.plot(self.t, averageTott, label = "Python tau neutrino contribution")
        ax.set_title(rf"Total energy at log(density/mu e) = {self.d[di]:5.2f}")
        ax.set_ylabel("Energy [KeV]")
        ax.legend(loc='best')
        ax.set_yscale('log')
        ax.set_xlabel("log(temp)[K]")
        fig.show()
        fig.savefig("Area phot.pdf")     
        

        
    def columb_param(self):
        density = []
        for i in range(len(self.d)):#prints density values. 
            x = 10**(self.d[i]+0.3)
            density.append(x)
        
        a = 28
        z = 14
        k = 1.3807*10**(-16)
        t = 10**9
        e = 4.8*10**(-10)
        u = 1.66054e-24
        d=[]
        for i in density:
            x = ((3*a*u)/(4*3.14*i))**(1/3)
            d.append(x)
        
        g = []
        for i in d:
            gamma = ((z**2*e**2)/(i*k*t))
            g.append(gamma)
        a1 = 12
        z1 = 6
        d1 =[]
        for i in density:
            x = ((3*a1*u)/(4*3.14*i))**(1/3)
            d1.append(x)
        g1 = []
        for i in d1:
            gamma = ((z1**2*e**2)/(i*k*t))
            g1.append(gamma)            
        
        fig = plt.figure()
        ax = fig.add_subplot(111)       
        ax.plot(self.d, g, label = "gamma factor for silicon")
        ax.plot(self.d, g1, label = "gamma factor for carbon")
        ax.axhline(y=1, color='r', linestyle="--", label="gamma = 1" )
        ax.axhline(y=178, color='b', linestyle = '--', label= "gamma = 178")
        ax.set_title("Value of gamma factor for brem process at 10$^9$ K")
        ax.set_ylabel("Gamma")
        ax.set_yscale('log')
        ax.legend(loc='best')
        ax.set_xlabel("log(density/mu e)(g/cm$^3$)")
        fig.show()
        fig.savefig("gammaFactor.pdf")    
        
        

       
    def sum_Area_Total(self, tf=True):
        totArea = []
        for i in range(len(self.t)): #Loop for finding all area values for all temperatures at different densities
            print(i)
            data = self.e[:, i].copy()
            

            if tf == True: #This is for neutrino flux Vs neutrino energy
                data *= self.E 
                
            
                
            area = np.sum((data[:,:,1:] + data[:,:,:-1]) * (self.E[1:] -self.E[:-1]), axis=-1) * 0.5
               
            area *= 1.60218e-6 #in ergs
            add=0
            for j in range(self.n): #adding  energy value for the same density
                add = area[j] + add #units on y axis as ergs/cm^3 
                    
            totArea.append(add)
        
        density = []
        for i in range(len(self.d)):#prints density values. 
            x = 10**(self.d[i]+0.3)
            density.append(x)
            print("Density is")
        print(density)
        print("Total energy")
        print(totArea[20])
       
    
    def plot_Fun(self, tf = True, ti=0, di=0):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(self.n):
            plotdata = self.e[i, ti, di].copy()
            if tf == True: #This is for neutrino flux Vs neutrino energy
                plotdata *= self.E
            ax.plot(self.E, plotdata, label = self.labels[i])
            

        if tf == True:
            title = rf"Figure1: Energy spectrum, log(T/K) = {self.t[ti]:5.2f}, log($\rho$ (g/cm$^3$)) = {self.d[di]:5.2f}"
            ylab = 'Flux of neutrinos [number of neutrinos/s/cm$^3$]'        
        else:
            title = rf"Figure1: Energy distribution, log(T/K) = {self.t[ti]:5.2f}, log($\rho$ (g/cm$^3$)) = {self.d[di]:5.2f}"
            ylab = 'phi(Energy)[ number of neutrinos/ergs/s/cm$^3$]'
        
        ax.set_title(title)
        ax.legend(loc='best')
        ax.set_ylabel(ylab)
        ax.set_xlabel('neutrino energy (MeV)')
        #ax.set_xscale('log')
        fig.tight_layout()
        fig.show()
        fig.savefig("Flux Vs Energy of neutrinos.pdf")
        
        self.ax = ax
        self.fig = fig
    
        

if __name__ == "__main__":        
    #plot_Fun(files, labels, True)
    #plot_Fun(files, labels, False)
    Neutrino().plot_Fun(True, 20, 10)

    #for temp in range(len(t)):
     #   for den in range(len(d)):
      #      pass

