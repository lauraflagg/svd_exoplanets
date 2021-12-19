import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 13)
from matplotlib.backends.backend_pdf import PdfPages

from astropy.time import Time

from astropy.io import fits, ascii
import os
import spectres, argparse
import importlib

import sys
sys.path.append('../../toimport')
sys.path.append('C:/Users/laura/programming/python/toimport')

from stats_lf import xcor, chisqwithshift
import astropy.constants
sys.path.append('../../directdetectionprograms/combine-and-xcor')
import astro_lf, comb_xcor, v_curve
from readwrite_lf import backupandwrite


import pickle, pandas
from PyAstronomy import pyasl
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
 


def getphase(mjd):
    return ((mjd-day0)/per) % 1.0

def getplanetv(mjd,e=0):
    if e>0.03:
        return comb_xcor.planetrvshift(mjd,planet_pars2,day0,code='vcurve')       
    return comb_xcor.planetrvshift(mjd,planet_pars,day0,code='sine')

def intransit(mjd): 
    pha=getphase(mjd)    
    td_perc=td_days/per
    td_perc_half=td_perc/2
    #print(pha,td_perc_half)
    if pha<td_perc_half or pha>(1-td_perc_half):
        return True
    else:
        return False
# In[8]:

#class ObservationSet:
#    def __init__(self,mjds,):

class Instrument:
    def __init__(self,name):
        self.name=name
        if name=='GRACES':
            totalorders=33
            wl_low=4100.
            wl_high=10000.
            disp=2.15
            lat=19.82396
            long=-155.46984
            height=4213
            npix=None
            leftedgecut=0
            
            rightedgecut=0
            orderstoplotinphase=[15,11,6,3]
            orderstoplotasresids=[11]
            wlunit='AA'
            wl_air_or_vac='air'
            printsecs=['allwls']
        if name=='McD_107in_echelle':
            totalorders=55
            wl_low=4100.
            wl_high=10000.
            disp=2.15
            lat=30.6714
            long=-104.022
            height=2070
            npix=None
            leftedgecut=0
            
            rightedgecut=0
            orderstoplotinphase=[18,19,25]
            orderstoplotasresids=[18,19]
            wlunit='AA'
            wl_air_or_vac='air'
            printsecs=['allwls','hawls','hewls']
        if name[0:6]=='IGRINS':
            totalorders=53
            wl_low=1.42
            wl_high=2.52
            disp=2.0

            npix=2048
            leftedgecut=200
            
            rightedgecut=100
            orderstoplotinphase=[6,10,11]
            orderstoplotasresids=[6]
            if name=='IGRINS_GS':
                lat=-30.2407
                long=-70.7366
                height=2715        
            elif name=='IGRINS_DCT' or name=='IGRINS_LDT':
                lat=30.6798
                long=-104.0248
                height=2076  
            name='IGRINS'
            wlunit='microns'
            wl_air_or_vac='vac'
            printsecs=['kwls','hwls']
        if name=='CARMENES' or name=='Carmenes' or name=='carmenes':
            name='CARMENES'
            totalorders=61
            wl_low=5290.
            wl_high=10600.
            disp=1.6
            lat=37.2236
            long=-2.54625
            height=2168
            npix=4096
            leftedgecut=0
            
            rightedgecut=0
            orderstoplotinphase=[25,36,38,47]
            orderstoplotasresids=[38]
            wl_air_or_vac='vac'
            wlunit='AA'
            printsecs=['allwls']
        
        self.totalorders=totalorders
        self.npix=npix
        self.wl_low=wl_low
        self.wl_high=wl_high
        self.disp=disp
        self.long=long
        self.lat=lat
        self.height=height
        self.leftedgecit=leftedgecut
        self.rightedgecut=rightedgecut
        self.orderstoplotinphase=orderstoplotinphase
        self.orderstoplotasresids=orderstoplotasresids
        self.teles = EarthLocation.from_geodetic(lat=self.lat*u.deg, lon=self.long*u.deg, height=self.height*u.m)
        self.wlunit=wlunit
        self.wl_air_or_vac=wl_air_or_vac
        self.printsecs=printsecs
        
    def getvbary(self,mjd,ra,dec):
        #barycorrection
        sc = SkyCoord(ra=ra*u.deg, dec=dec *u.deg)    
        barycorr = sc.radial_velocity_correction(obstime=Time(mjd,format='mjd'), location=self.teles)  
        return barycorr.to(u.km/u.s).value
    
    def jd_to_hjd(self,mjd,ra,dec):
        sc = SkyCoord(ra=ra*u.deg, dec=dec *u.deg)  
        times = Time(mjd, format='mjd',scale='utc', location=self.teles) 
        return mjd+times.light_travel_time(sc).value
    

    def getheaderinfo(self,date,folds,sncut):
        fns=[]
        ts=[]
        minszs=[]
        
        if self.name=='GRACES':        
            for fn in folds:
                if fn.endswith('.fits'):        
                    with fits.open('../data/reduced/'+date+'/'+fn) as f:
                        hdr=f[0].header
                        test=f[0].data        
                        tmid=float(hdr['MJDATE'])+float(hdr['EXPTIME'])*.5/(60*60*24)
                        #print(tmid,float(hdr['HJDUTC'])-2400000.5 )
                        #print(getvbary(tmid,self),hdr['HRV'])
                        snrs=float(hdr['SNR32'][:3]) #CHANGE WHEN SWITCHING OBJECT
                        #print(tmid,snrs)
        
                    if snrs>sncut:            
                        fns.append(fn)
                        ts.append(float(tmid))  
                        minszs.append(np.min(test[1]))  
        elif self.name=='McD_107in_echelle':        
            for fn in folds:
                if fn.endswith('.ech') and fn[:3]!='wid':        
                    with fits.open('../data/reduced/'+date+'/'+fn) as f:
                        hdr=f[0].header
                        test=f[1].data        
                        tmid=Time(hdr['DATE-OBS']+'T'+hdr['UT'], format='isot', scale='utc').to_value('mjd')+float(hdr['EXPTIME'])*.5/(60*60*24)
                        #print(tmid,float(hdr['HJDUTC'])-2400000.5 )
                        #print(getvbary(tmid,self),hdr['HRV'])
                        snrs=np.median(test[0][1][20]) #CHANGE WHEN SWITCHING OBJECT
                        #print(tmid,snrs)
        
                    if snrs>sncut and hdr['IMAGETYP'][0:4]!='comp'and hdr['THARLAMP'][0:3]=='OFF':            
                        fns.append(fn)
                        ts.append(float(tmid))  
                        minszs.append(2048)  
        elif self.name=='Carmenes' or self.name=='CARMENES':
            for fn in folds:
                if fn.endswith('A.fits'):
        
                    with fits.open('../data/'+date+'/'+fn) as f:
                        hdr=f[0].header        
                        tmid=tmid=float(hdr['MJD-OBS'])+float(hdr['EXPTIME'])*.5/(60*60*24)

                        snrs=hdr['HIERARCH CARACAL FOX SNR 35']
        
                    if snrs>sncut: 
                        fns.append(fn)
                        ts.append(float(tmid))   
        elif self.name=='IGRINS':
            for fn in folds:
                if fn.endswith('spec.fits') and fn.startswith('SDCK'):
        
                    code=fn[5:18]
                    with fits.open('../data/'+date+'/'+fn[:-9]+'sn.fits') as f:
                        hdr=f[0].header
                        date_begin=hdr['DATE-OBS'] #in UTC
                        date_end=hdr['DATE-END']
                        #time_MJD=hdr['MJD-OBS']
        
                        t1=Time(date_begin[0:10]+'T'+date_begin[11:],format='isot',scale='utc')
                        t2=Time(date_end[0:10]+'T'+date_end[11:],format='isot',scale='utc')
        
        
        
        
                        dtemp=f[0].data
                        snrs=np.sum(np.nan_to_num(dtemp[6]))
        
                    if snrs>sncut:                    
        
        
        
                        fns.append(fn[5:18])
                        ts.append(float(0.5*(t1.mjd+t2.mjd)))            

        return fns,ts,minszs
    
    def getdata(self,date,all_fns,sim=False,vsysshift=0,template='',scale=1):
        data=np.zeros((len(all_fns),self.totalorders,self.sz))
        uncs0=np.zeros((len(all_fns),self.totalorders,self.sz))
        wl=np.zeros((len(all_fns),self.totalorders,self.sz))
        time_MJD=np.zeros(len(all_fns))
        intransit_list=[]
        if self.name=='GRACES':
            orders = list(np.arange(33)+23)
            
            for i,item in enumerate(all_fns):
                #print(i,item) 

                with fits.open('../data/reduced/'+date+'/'+item) as f:
                    
                    test=f[0].data
                    for o in orders:
                        
                        loc=o-np.min(orders)
                        choose=np.where(test[0]==o)
                        wls_o=test[5][choose]
                        flux_o=test[10][choose]
                        unc_o=test[11][choose]
    
                        
                        more=len(wls_o)-self.sz
                        if more<2:
                            start=0
                        elif more<4:
                            start=1
                        elif more<6:
                            start=2
                        elif more<8:
                            start=3
                        elif more<8:
                            start=4
                        else:
                            start=5
                        
                        wl[i,loc]=wls_o[start:(start+self.sz)]*10
                        data[i,loc]=flux_o[start:(start+self.sz)]
                        uncs0[i,loc]=unc_o[start:(start+self.sz)]
                    

                    hdr=f[0].header
                    tmid=float(hdr['MJDATE'])+float(hdr['EXPTIME'])*.5/(60*60*24)
                    time_MJD[i]=float(tmid)                    
                    intransit_list.append(intransit(float(hdr['HJDUTC'])-2400000.5 ))
                    
    
                    if sim: #DOES NOT WORK YET FOR GRACES
                        intran=intransit_list[i]
                        #print(intran)                    
                        dreturn=np.zeros_like(data[i])
                        if intran:
                            vptot=-getvbary(time_MJD[i])-getplanetv(time_MJD[i])+vsysshift
                            
                            nf_1, wl_1 = pyasl.dopplerShift(template['wl_(A)'].values, template['flux'].values, vptot, edgeHandling='firstlast', fillValue=None)              
                            
                            
                            for o,tempwls in enumerate(f[1].data):
                                dreturn[o]=dtemp[o]*(1-spectres.spectres(tempwls,wl_1/1e4,template['flux'].values,verbose=False)*scale)
                        else:
                            dreturn=dtemp
                        data[i]=dreturn
                    else:
                        tempxxx=1
        elif self.name=='HARPS':
            for i,item in enumerate(all_fns):
                #print(i,item)
    
    
    
                with fits.open('../data/'+date+'/'+item) as f:
                    dtemp=f[0].data
                    hdr=f[0].header
                    wlfile=hdr['HIERARCH ESO DRS CAL TH FILE']
                    with fits.open('../data/'+date+'/'+wlfile) as f2:
                        wl[i]=f2[0].data
                    
        elif self.name=='McD_107in_echelle':
            for i,item in enumerate(all_fns):
                #print(i,item)
    
    
                with fits.open('../data/reduced/'+date+'/'+item) as f:
                    dtemp=f[1].data[0][0]
                    uncs0[i]=f[1].data[0][1]
    
                    data[i]=dtemp
                    wldat=ascii.read('../data/reduced/'+date+'/wls.dat')
                    wl[i]=np.zeros_like(dtemp)
                    for o in range(self.totalorders):
                        wl[i,o]=wldat.columns[o].value
                        
                                     
                    hdr=f[0].header
    
                    tmid=Time(hdr['DATE-OBS']+'T'+hdr['UT'], format='isot', scale='utc').to_value('mjd')+float(hdr['EXPTIME'])*.5/(60*60*24)
                    time_MJD[i]=float(tmid)
                    intransit_list.append(intransit(time_MJD[i]))
                    
    
                    if sim:
                        intran=intransit_list[i]
                        #print(intran)                    
                        dreturn=np.zeros_like(dtemp)
                        if np.array(intransit_list).any():
                            if intran:
                                vptot=-getvbary(time_MJD[i])-getplanetv(time_MJD[i])+vsysshift
                                
                                nf_1, wl_1 = pyasl.dopplerShift(template['wl_(A)'].values, template['flux'].values, vptot, edgeHandling='firstlast', fillValue=None)              
                                
                                
                                for o,tempwls in enumerate(wl):
                                    dreturn[o]=dtemp[o]*(1-spectres.spectres(tempwls,wl_1/1e4,template['flux'].values,verbose=False)*scale)
                            else:
                                dreturn=dtemp
                        else:
                            vptot=-self.getvbary(time_MJD[i],ra=ra,dec=dec)+getplanetv(time_MJD[i],e=e)+vsysshift
                            #print(-self.getvbary(time_MJD[i],ra=ra,dec=dec),-getplanetv(time_MJD[i],e),vptot)
                            
                            nf_1, wl_1 = pyasl.dopplerShift(template['wl_(A)'].values, template['flux'].values, vptot, edgeHandling='firstlast', fillValue=None)              
                            
                            for o,tempwls in enumerate(wl[0]):
                                if o==180 or o==190:
                                    plt.plot(tempwls,dtemp[o])
                                    plt.plot(tempwls,spectres.spectres(tempwls,wl_1,template['flux'].values,verbose=False,fill=1)*scale)
                                    plt.xlim(6675,6685)
                                    plt.show()
                                dreturn[o]=dtemp[o]+spectres.spectres(tempwls,wl_1,template['flux'].values,verbose=False,fill=1)*scale                            
                            
                        data[i]=dreturn
                    else:
                        data[i]=dtemp            
            
        
        elif self.name=='Carmenes' or self.name=='CARMENES':
            for i,item in enumerate(all_fns):
                #print(i,item)
    
    
    
                with fits.open('../data/'+date+'/'+item) as f:
                    dtemp=f[1].data
                    uncs0[i]=f[2].data
    
                    data[i]=dtemp
                    wl[i]=(f[4].data)
                    hdr=f[0].header
    
                    tmid=tmid=tmid=float(hdr['MJD-OBS'])+float(hdr['EXPTIME'])*.5/(60*60*24)#hdr['HIERARCH CARACAL HJD']-0.5
    
                    time_MJD[i]=float(tmid)
                    vbarys[i]=hdr['HIERARCH CARACAL BERV']
                    intransit_list.append(intransit(time_MJD[i]))
                    
    
                    if sim:
                        intran=intransit_list[i]
                        #print(intran)                    
                        dreturn=np.zeros_like(dtemp)
                        if intran:
                            vptot=-getvbary(time_MJD[i])-getplanetv(time_MJD[i])+vsysshift
                            
                            
                            nf_1, wl_1 = pyasl.dopplerShift(template['wl_(A)'].values, template['flux'].values, vptot, edgeHandling='firstlast', fillValue=None)              
                            
                            
                            for o,tempwls in enumerate(f[1].data):
                                dreturn[o]=dtemp[o]*(1-spectres.spectres(tempwls,wl_1/1e4,template['flux'].values,verbose=False)*scale)
                        else:
                            dreturn=dtemp
                        data[i]=dreturn
                    else:
                        data[i]=dtemp
            
        elif name=='IGRINS':
            for i,item in enumerate(all_fns):
                #print(i,item)
    
    
    
                with fits.open('../data/'+date+'/'+'SDCK_'+item+'.spec.fits') as f:
                    dtemp=f[0].data
    
                    data[i,0:25,]=dtemp[0:25,:]
                    wl[i,0:25,]=(f[1].data)[0:25,:]
                    hdr=f[0].header
                    date_begin=hdr['DATE-OBS'] #in UTC
                    date_end=hdr['DATE-END']
                    #time_MJD=hdr['MJD-OBS']
    
                    t1=Time(date_begin[0:10]+'T'+date_begin[11:],format='isot',scale='utc')
                    t2=Time(date_end[0:10]+'T'+date_end[11:],format='isot',scale='utc')
    
    
                    time_MJD[i]=float(0.5*(t1.mjd+t2.mjd))
                    intransit_list.append(intransit(time_MJD[i]))
    
                    if sim:
                        intran=intransit_list[i]
                        #print(intran)                    
                        dreturn=np.zeros_like(dtemp)
                        if intran:
                            vptot=-getvbary(time_MJD[i])-getplanetv(time_MJD[i])+vsysshift
                            
                            nf_1, wl_1 = pyasl.dopplerShift(template['wl_(A)'].values, template['flux'].values, vptot, edgeHandling='firstlast', fillValue=None)              
                            
                            
                            for o,tempwls in enumerate(f[1].data):
                                dreturn[o]=dtemp[o]*(1-spectres.spectres(tempwls,wl_1/1e4,template['flux'].values,verbose=False)*scale)
                        else:
                            dreturn=dtemp
                        data[i,0:25,]=dreturn[0:25,:]
                    else:
                        data[i,0:25,]=dtemp[0:25,:]
    
    
    
                with fits.open('../data/'+date+'/'+'SDCH_'+item+'.spec.fits') as f:
                    wl[i,25:,]=(f[1].data)[:,:]
                    
                    dtemp=f[0].data
                    dt=f[0].data

                    if sim:
                        dreturn=np.zeros_like(dtemp)
                        if intran:

                            for o,tempwls in enumerate(f[1].data):
                                dreturn[o]=dtemp[o]*(1-spectres.spectres(tempwls,wl_1/1e4,template['flux'].values,verbose=False)*scale)
                        else:
                            dreturn=dtemp
                        data[i,25:,]=dreturn[:,:]
                    else:
                        data[i,25:,]=dtemp[:,:]
            data=data[:,:,self.leftedgecut:-self.rightedgecut]
            uncs0=uncs0[:,:,self.leftedgecut:-self.rightedgecut]
            wl=wl[:,self.leftedgecut:-self.rightedgecut]
            
        return wl,data,uncs0,time_MJD,intransit_list
       
def dosmooth(d_byorder,dates,polyorder=5):
    data_arr_A=np.zeros_like(d_byorder)
    #data_arr1_A=data_arr_A.copy()
    #data_arr=np.zeros(data.shape)
    print('smoothing with poly odrer ',str(polyorder))
    for i,item in enumerate(d_byorder):
        specs=item
        new_specs=np.zeros_like(specs)
        #print(specs.shape)
        for w,wlbin in enumerate(specs):
            coeffs=np.polynomial.polynomial.polyfit(dates,wlbin,polyorder)
            new_specs[w]=wlbin-np.polynomial.polynomial.polyval(dates,coeffs)

        data_arr_A[i]=new_specs          
            
    return data_arr_A

def doPCA(d_byorder,comps=4,wlshift=False,sigcut=3.):
    if comps==0:
        return d_byorder
    data_arr_A=np.zeros_like(d_byorder)
    data_arr1_A=data_arr_A.copy()
    #data_arr=np.zeros(data.shape)
    for i,item in enumerate(d_byorder):


        #print(item.shape)
        if i>=-10:
            u,s1,vh=np.linalg.svd(item,full_matrices=False)  #decompose
            s=s1.copy()
            s1[comps:]=0.
            W1=np.diag(s1)
            #print(i)
            A1=np.dot(u,np.dot(W1,vh))
            #plt.imshow(np.transpose(A1))
            #plt.show()
            #plt.close()

            data_arr1_A[i,]=A1

            s[0:comps]=0.
            W=np.diag(s)
            A=np.dot(u,np.dot(W,vh))

            #print(i)
            #plt.imshow(np.transpose(A))
            #plt.show()
            #plt.close()
            #sigma clipping
            #'''
            sig=np.std(A)
            med=np.median(A)
            loc=np.where(A > sigcut*sig+med)
            A[loc]=0#*0.+20*sig
            loc=np.where(A < -sigcut*sig+med)
            A[loc]=0#*0.+20*sig
            #'''
            #
            data_arr_A[i,]=A
            if i==6:
                print(np.std(A))
    return data_arr_A


# In[9]:

def doall(date,inst,iters=2,comps=4,wlshift=False,plot=True,sncut=570000,dvcut=10,templatefn='',vsysshift=-10.,scale=1,wv=True,sigcut=3.,initnorm=True,smooth=-1):

    sim=False
    if templatefn!='':
        template=pandas.read_csv('templates/'+templatefn)
        sim=True
    else:
        template=''
    
    print(date, 'iters:',iters,' components:',comps)
    folds=os.listdir('../data/reduced/'+date)
    folds.sort()

    fns,ts,minszs=inst.getheaderinfo(folds=folds,sncut=sncut,date=date)

    all_fns = fns
    inst.sz=int(np.min(minszs))
    if inst.npix!=None:
        inst.sz=inst.npix
    #inst.sz=sz    


    print('starts at:',np.min(ts),'ends at',np.max(ts))
    dv=(getplanetv(np.max(ts))-getplanetv(np.min(ts)))
    print('num obs used=',len(all_fns),' |  dv=',dv,'km/s')
    #print(all_fns)
    vbarys=np.zeros(len(all_fns))
    if np.abs(dv)>dvcut:

        wl,data,uncs0,time_MJD,intransit_list=inst.getdata(all_fns=all_fns,date=date,sim=sim,vsysshift=vsysshift,template=template,scale=scale)
        data_temp=data[:,:,:]

        wl_meds=np.median(wl,axis=0)      
        if wlshift:
            fff='shifted_data_'+inst.name+date+'_sncut'+str(int(sncut))+'.pic'
            if os.path.exists(fff):
                print('using saved wl shift data')
                with open(fff, 'rb') as file:
                    data,uncs1=pickle.load(file)
            else:
                print('creating new wl shift data')
                arruse=data
                wluse=wl
                blankarr=np.zeros((arruse.shape[0],arruse.shape[1],len(wl_meds[0])))
                uncs1=np.zeros((arruse.shape[0],arruse.shape[1],len(wl_meds[0])))
                for s, spec in enumerate(arruse):
                    for o,f_o in enumerate(spec):
                        blankarr[s,o],uncs1[s,o]=spectres.spectres(wl_meds[o],wluse[s,o],f_o,spec_errs=uncs0[s,o],verbose=False,fill=1)
                data=blankarr        
                with open(fff, 'wb') as file:
                    pickle.dump((data,uncs1),file,protocol=2)            


        else:

            wl_meds=wl_meds[:,:]
            data=data[:,:,:]        
            uncs1=uncs0


        #print(inst)
        data_byorder0=np.nan_to_num(data.transpose((1,2,0)))
        uncs_byorder=np.nan_to_num(uncs1.transpose((1,2,0)))
        #wl_byorder=np.nan_to_num(wl.transpose((1,2,0)))
        intransit_arr=np.array(intransit_list)

        #"blaze" correct
        if (iters>0 or smooth>-1) and initnorm!=False:
            data_byorder=np.zeros_like(data_byorder0)
            for i,item in enumerate(data_byorder0):
                item0=item.transpose()
                if inst.name[0:3]=='McD' and initnorm=='wide.ech':
                    with fits.open('../data/reduced/'+date+'/wide.ech') as f:
                        item1=f[1].data[0][0]                
                if transiting==1 and intransit_arr.any(): #if transit observations, divide by out of tranitmedian ; else divide by all
                    item1=item0[np.where(intransit_arr==False)]
                else:
                    item1=item0


                med_spec=np.median(item1,axis=0)
                #sd_spec=np.std(item1,ddof=1,axis=0)
                #new_specs0=data_byorder0[i].transpose()/med_spec
                #new_specs0[np.where(new_specs0>(med_spec+50
                temp1=np.nan_to_num(np.transpose(data_byorder0[i].transpose()/med_spec))
                temp2=np.ones_like(temp1)
                for ii,spec in enumerate(temp1.transpose()):
                    #print(len(spec))
                    mv=np.median(spec[300:1200])

                    #temp2[:,ii]=spec/astro_lf.findcontlevel(spec)
                    temp2[:,ii]=np.nan_to_num(spec/mv)


    #            print(avg_spec.shape)
    
                data_byorder[i]=temp2
        else:
            data_byorder=data_byorder0
#            print(np.where(avg_spec==0))

        if smooth>-1:
            data_byorder=dosmooth(data_byorder,time_MJD,smooth)


        for i in range(iters):

            data_byorder=doPCA(data_byorder,comps, wlshift=wlshift,sigcut=sigcut)

        data_arr=data_byorder    

        if plot:
            aspect=np.average(inst.wls)/15.
            #increase aspect to increase the height relative to the width
            phase0=getphase(np.min(time_MJD))
            phase1=getphase(np.max(time_MJD))
            if phase1<phase0:
                phase1=phase1+1
            print('phase extent',phase0,phase1)
            
            sbn=len(inst.orderstoplotasresids)+len(inst.orderstoplotinphase)

            fig_all=plt.figure()            
            
            pn=1
            for i in inst.orderstoplotasresids:
                A=data_arr[i,]
                plt.subplot(sbn,1,pn)
                for item in np.transpose(A):
                    plt.plot(wl_meds[i,],item)
                    #print(item)
                plt.ylabel('residual')
                plt.xlim(wl_meds[i,].min(),wl_meds[i,].max())        
                pn=pn+1
            
            for i in inst.orderstoplotinphase:
                A=data_arr[i,]
                plt.subplot(sbn,1,pn)
                plt.title(date)
                plt.imshow(np.transpose(A),extent=[wl_meds[i,].min(),wl_meds[i,].max(), phase0,phase1],aspect=aspect)
                #title('Tellurics Removed (PCA)')
                #plt.xlabel('$\lambda$ [$\AA$]')
                plt.ylabel('phase')
                pn=pn+1
            plt.xlabel('$\lambda$ [$\AA$]')

            pp.savefig(fig_all)
            plt.close()

        variances=np.std(data_byorder,axis=2)**2
        if wv:
            
            print(data_byorder.shape,variances.shape)
            temp33=data_byorder.transpose((2,0,1))
            arrused=np.transpose(temp33/variances,(1,2,0))
            print(temp33.shape,data_arr.shape)
        else:
            arrused=data_byorder


        wlused=wl_meds
        
         #print(arrused[loc])

        data_1d=np.zeros((len(all_fns),len(inst.wls)))
        uncs_1d=np.zeros((len(all_fns),len(inst.wls)))
        print(data_1d.shape)

        #want to interpolate onto evenly spaced scale in vacuum wavelengths
        #if the instrument outputs in vacuum then we can do this directly
        #if not, we need to convert the desired wls to air.  When we interpolate
        #onto that, we can then substitue in the vacuum wavelengths
        if inst.wl_air_or_vac=='vac':
            wlstouse=inst.wls
        elif inst.wl_air_or_vac=='air':
            wlstouse=pyasl.vactoair2(inst.wls)
              
        for o,o_data in enumerate(arrused):
            #print(o_data.shape,wlused[o].shape)
            o_data_1,u1d=spectres.spectres(wlstouse,wlused[o],o_data.transpose(),spec_errs=uncs_byorder[o].transpose(),fill=0,verbose=False)

            #uncertainties and weighting the edges
            wo1=uncs_1d
            wo2=np.nan_to_num(1./wo1**2,neginf=0,posinf=0) #set the wieght of anything that doesn't have a good weight to 0
            wn1=u1d
            wn2=np.nan_to_num(1./wn1**2,neginf=0,posinf=0)
            wsum=wo2+wn2
            #wsum[np.where(wsum==0)]=1e40
            nd=(data_1d*wo2+o_data_1*wn2)/(wo2+wn2)
            #print(nd.shape)
            data_1d=np.nan_to_num(nd,neginf=0,posinf=0)
            uncs_1d=np.sqrt(uncs_1d**2+u1d**2)
        #print(data_1d[:,loc])




        topick=[data_1d,time_MJD]
        if 1==2:
            if wlshift:
                pickle.dump(topick,open(date+'_wlshift_iters'+str(iters)+'_comps'+str(comps)+'_sncut'+str(int(sncut))+'_dvcut'+str(int(dvcut))+'.pic','wb'),protocol=2)
            else:
                pickle.dump(topick,open(date+'_iters'+str(iters)+'_comps'+str(comps)+'_sncut'+str(int(sncut))+'_dvcut'+str(int(dvcut))+'.pic','wb'),protocol=2)

        return topick
    else:
        print(date,' not used')
        return [] 



def conc_data(date_list,returns):
    for date in date_list:
        if date!=date_list[0]:
            print(data_tog.shape)
            #data_tog=data_tog+returns[date][0]
            data_tog=np.concatenate((data_tog,(returns[date])[0]))
            mjd_tog=np.concatenate((mjd_tog,returns[date][1]))


        else:


            data_tog=returns[date][0]
            mjd_tog=returns[date][1]



    return data_tog,mjd_tog
def print_section(inst,sec,secname,data_tog_final,mjd_tog,dates_used,filecode,savecsv=False):


    num=len(mjd_tog)
    toprint=np.zeros((num+1,len(inst.wls[sec])))
    if inst.wlunit=='AA':
        toprint[0]=inst.wls[sec]/1e4 #to convert to microns
    elif inst.wlunit=='microns':
        toprint[0]=inst.wls[sec]
    for i in range(num):
        toprint[i+1]=data_tog_final[i][sec]
    d=['wavelength_(microns)']
    d=d+list(mjd_tog.astype(np.str))


    fn=filecode+'_tog_'+secname+'.csv'
    print(fn[:-3]+'pic')
    
    if savecsv:
        backupandwrite(fn,d,np.transpose(toprint))
    pickle.dump((d,np.transpose(toprint)),open(fn[:-3]+'pic','wb'),protocol=2)
    

 

# # dev code


# # run code

# In[ ]:

def main(target,instname='GRACES', date_list=['20160202','20160222','20160224','20160225','20160226','20160324'],iters=1,comps=4,comps2=0,do_new=True,wlshift=False,templatefn='',savecsv=False,plot=True,wv=True,sncut=570000.,dvcut=10.,savecode='',vsysshift=-10.,scale=1,normtwice=False,subtwice=False,sigcut=3.,initnorm=True,smooth=-1):
    mdl = importlib.import_module(target+'_pars')
    
    # is there an __all__?  if so respect it
    if "__all__" in mdl.__dict__:
        names = mdl.__dict__["__all__"]
    else:
        # otherwise we import all names that don't begin with _
        names = [x for x in mdl.__dict__ if not x.startswith("_")]
    #print(names)
    # now drag them in
    globals().update({k: getattr(mdl, k) for k in names})    
    
    
    #define file code for outpurs
    if 1==1:
        
        if smooth>-1:
            sos='_smoothorder'+str(smooth)
        else:
            sos=''
        
        if initnorm!=True and initnorm!=False:
            ins=intinorm
        else:
            ins=''
        
        if templatefn!='':
            vsstr='_'+str(vsysshift)
            print('template for sims=',templatefn)
            sstr='_scale'+str(scale)
        else:
            vsstr=''
            sstr=''

            
        if wv:
            wbv='_weightedbyvariance'
        else:
            wbv=''
        if normtwice:
            n2='_normedtwice'
        else:
            n2=''  
        if subtwice:
            s2='_normedtwice'
        else:
            s2=''  
        
        if comps2>0:
            c2='_comps2_'+str(comps2)
        else:
            c2=''
        
        if wlshift:
            fss='shifted_'
        else:
            fss=''
            
        if sigcut!=3:
            scc='sigcut'+str(sigcut)
        else:
            scc=''
        
            
        filecode='../data/PCA_'+target+'_'+fss+'iters'+str(iters)+'_comps'+str(comps)+'_sncut'+str(int(sncut))+'_dvcut'+str(int(dvcut))+savecode+templatefn+vsstr+sstr+wbv+n2+s2+c2+scc+ins+sos
        print(filecode)
    
    inst=Instrument(instname)
    inst.wls=astro_lf.createwlscale(inst.disp,inst.wl_low,inst.wl_high)
    dates_used=[]
    returns={}
    if plot:
        global pp

        fn=filecode+'.pdf' 
        pp = PdfPages(fn)    
    

    if do_new:
        for item in date_list:
            temp_ret=doall(item,inst,iters=iters,comps=comps,wlshift=wlshift,plot=plot,sncut=sncut,dvcut=dvcut,templatefn=templatefn,vsysshift=vsysshift,scale=scale,wv=wv,initnorm=initnorm,smooth=smooth)
            if temp_ret!=[]:
                returns[item]=temp_ret
                dates_used.append(item)
    else:  
        for date in date_list:
            if wlshift:
                ob = open(date+'_wlshift_iters'+str(iters)+'_comps'+str(comps)+'_sncut'+str(int(sncut))+'_dvcut'+str(int(dvcut))+'.pic', "rb")

            else:
                ob = open(date+'_iters'+str(iters)+'_comps'+str(comps)+'_sncut'+str(int(sncut))+'_dvcut'+str(int(dvcut))+'.pic', "rb")
            returns[date]=pickle.load(ob)
            ob.close()
    if plot:
        pp.close()


    data_tog,mjd_tog=conc_data(dates_used,returns)      

    Vbary_tog=np.zeros(len(mjd_tog))
    for i in range(len(mjd_tog)):
        Vbary_tog[i]=inst.getvbary(mjd_tog[i],ra,dec)
    mjd_tog=inst.jd_to_hjd(mjd_tog,ra,dec)


    # In[ ]:
    secdefs={}
    secdefs['allwls']=np.where((inst.wls>np.min(inst.wls)) & (inst.wls<np.max(inst.wls)))
    secdefs['kwls']=np.where((inst.wls>2.0) & (inst.wls<2.45))
    secdefs['hwls']=np.where((inst.wls>1.4) & (inst.wls<1.8))
    secdefs['hawls']=np.where((inst.wls>6550) & (inst.wls<6580))
    secdefs['hewls']=np.where((inst.wls>6660) & (inst.wls<6700))

    data_tog_final=np.zeros_like(data_tog)
    for i,spec in enumerate(data_tog):
        #print(Vbary_tog[i])
        nf_1, wl_1 = pyasl.dopplerShift(inst.wls, spec, Vbary_tog[i], edgeHandling='firstlast', fillValue=None)
        data_tog_final[i]=nf_1
        
    if comps2>0:
        print('PCA-ing again')

        data_tog_final0=doPCA([np.nan_to_num(data_tog_final,neginf=0,posinf=0)],comps=comps2)
        data_tog_final=data_tog_final0[0]
        print(data_tog_final.shape,data_tog_final0.shape)

        print('done')
    
    if normtwice:
        intransit_arr=np.array([intransit(item) for item in mjd_tog])
        med_spec=np.zeros_like(inst.wls)
        med_spec[:]=np.median(data_tog_final[~intransit_arr,:],axis=0)

        data_tog_final=(data_tog_final+1)/(med_spec+1)-1
    
    if subtwice:
        intransit_arr=np.array([intransit(item) for item in mjd_tog])
        med_spec=np.zeros_like(inst.wls)
        med_spec[:]=np.median(data_tog_final[~intransit_arr,:],axis=0)

        data_tog_final=(data_tog_final)-(med_spec)-1    
    

    for item in inst.printsecs:
        print_section(inst=inst,sec=secdefs[item],secname=str(item),data_tog_final=data_tog_final,mjd_tog=mjd_tog,dates_used=dates_used,filecode=filecode,savecsv=savecsv)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('iters',type=int)  
    parser.add_argument('comps',type=int)
    
    parser.add_argument('sncut',type=float)
    parser.add_argument('dvcut',type=float)

    parser.add_argument('--date_list', nargs='+')
    parser.add_argument('-c2','--comps2',type=int,default=0)
    parser.add_argument('-ts','--template_scale', type=float,default=1,help='scale template by this factor')

    parser.add_argument('-dn', '--do_new', action='store_true', help="redo PCA")
    parser.add_argument('-uo', '--use_old', action='store_true', help="don't redo PCA")
    parser.add_argument('-uwv', '--unweight_variance', action='store_true', help="don't weight by variance")
    parser.add_argument('-w', '--wl_shift', action='store_true', help="shift wls before PCA")
    parser.add_argument('-dp', '--dontplot', action='store_true', help="don't plot PCA-ed fluxes")
    parser.add_argument('-sc', '--savecode', default='', help="savecode file name addition")
    parser.add_argument('-tfn', '--templatefn', default='', help="template file name for sim")
    parser.add_argument('-scsv', '--savecsv',  action='store_true', default='', help="save csv in addition to a picke file")
    parser.add_argument('-n2', '--normtwice',  action='store_true', default='', help="divide by median again at end")
    parser.add_argument('-s2', '--subtwice',  action='store_true', default='', help="subtract again at end")
    parser.add_argument('-vs', '--vsys', default=0, help="systemic velocity for simulated data",type=float)
    args = parser.parse_args()

    #print(args.date_list)
    if args.date_list==None:
        date_list=os.listdir('../data/reduced')
    else:
        date_list=args.date_list

    if args.dontplot:
        plot=False
    else:
        plot=True
    
    if args.use_old and ~args.do_new:
        dn=False
    else:
        dn=True
        
    if args.unweight_variance:
        wv=False
    else:
        wv=True
    
        
    print('doing iters=',args.iters,' comps=',args.comps,' comps2=',args.comps2,' datelist=',date_list,' wlshift=',args.wl_shift,' s2ncut=',args.sncut,'dvcut=',args.dvcut,' weighted variance=',wv)
    main(date_list,iters=args.iters,comps=args.comps,comps2=args.comps2,do_new=dn,wlshift=args.wl_shift,plot=plot,sncut=args.sncut,wv=wv,dvcut=args.dvcut,templatefn=args.templatefn,savecode=args.savecode,vsysshift=args.vsys,scale=args.template_scale,savecsv=args.savecsv,normtwice=args.normtwice,subtwice=args.subtwice)

    # 3 6  290000 1  -dn -tfn Flat_Gl486b_H2O_main_iso_MassFrac5.117e-02_MMW3.32_R45k.csv
     # 1 6  290000 1  -dn -tfn Flat_Gl486b_H2O_main_iso_MassFrac5.117e-03_MMW2.61_R45k.csv
     # 

     