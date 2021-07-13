# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 09:33:02 2021

NB - this thing will try to estimate the sensor offset, the cell level fluorescence,
and measure transient parameters all automatically. There is something that
it needs to carry this out:
    
    the image needs to have a bit of the sensor bg (i.e. no cell) on the line 
    recording
    
plan:
do some kind of smoothing (either PURE [overkill] or just a 2-3px gauss blur) 
threshold via something (maybe fit exponential/piecewise linear? + 3sd > base), 
take some time interval before threshold region as baseline for f0,
calc ff0, 
calculate ttp, fdhm, dyssync
calculate whatever else....

might make it a class type layout at some point incase i need to incorporate it
into a GUI. It would also make setting some class variables for use by class 
functions a lot easier.

Seriously yufeng, make this into a class.

I need to be better at pep8

Jun 21 - initial build, first iteration (no computer admin access, can't git push/pull)

@author: yhou021
"""

import numpy as np
import pylab as plt
import pandas as pd
from scipy import ndimage as nd
from skimage import io, filters
import os
import tifffile as tif



def otsu_two_step(line):
    """
    carries out double otsu method where an intial threshold extracts the 
    values within the transients and a second pass aims to extract the cell
    position
    
    can be pretty sketchy if the cell bg is very high or if cell width is tiny
    
    can work with infinite dimensions, but you're weird if you do.
    
    TODO: need to test this
    
    seems to fail on some wave measures
    
    Parameters
    ----------
    line : ndarray[M, N, etc]
        data from a single line of linescan data
    

    Returns
    -------
    mask1 : ndarray[M, N, etc]
        upper region (transients/waves)
    mask2 : ndarray[M, N, etc]
        cell region
    mask3 : ndarray[M, N, etc]
        bg region

    """
    
    thresh1 = filters.threshold_otsu(line)
    
    dset1 = line[line>thresh1]
    dset2 = line[line<thresh1]
    mask1 = line > thresh1
    marea = np.sum(mask1)
    
    """
    thresholding tends to go awry, apply additional layer of otsu if the whole 
    cell gets segmented for some reason
    """
    thresh2 = filters.threshold_otsu(dset2)
    mask2 = np.logical_and(line > thresh2, line < thresh1)
    m2area = np.sum(mask2)
    ratio = marea/m2area
    
    if ratio > 2:    
        thresh1 = filters.threshold_otsu(dset1)
        mask1 = line > thresh1
        
    mask3 = line < thresh2
    
    return mask1, mask2, mask3


def gen_masks(img, gauss_radius = 5, yrad = 5):
    """
    Generates the masks for input line scan.
    Assumes axis 1 as x and axis 0 as t

    Parameters
    ----------
    img : ndarray int, unint, float
        image data from line scans
    gauss_radius : int, optional
        radius for initial gauss filter. The default is 3.

    Returns
    -------
    msk1 : ndarray[M, N]
        upper region (transients/waves)
    msk2 : ndarray[M, N]
        cell region
    msk3 : ndarray[M, N]
        bg region

    """
    
    msk1 = np.zeros(img.shape)
    msk2 = np.zeros(img.shape)
    msk3 = np.zeros(img.shape)
    
    #scipy.ndimage.gaussian_filter1d
    print('here')
    #img = nd.gaussian_filter1d(img, sigma=yrad, axis=0)
    print(img.shape)
    print(gauss_radius)
    #img = nd.gaussian_filter1d(img, sigma=int(gauss_radius), axis=1)
    print('here2')
    img = nd.gaussian_filter(img, sigma=[yrad,int(gauss_radius)])
    #plt.imshow(img)
    
    #for i in range(img.shape[1]):
    #    msk1[:,i], msk2[:,i], msk3[:,i] = otsu_two_step(img[:,i])
    
    msk1, msk2, msk3 = otsu_two_step(img)
    
    return msk1, msk2, msk3, img


def circle_mask(r):
    """
    it just makes a circle mask....
    r^2 = x^2 + y^2

    Parameters
    ----------
    r : int
        radius of circle.

    Returns
    -------
    out : ndarray[M, N]
        output mask

    """
    
    out = np.zeros((2*r+1, 2*r+1))
    grid = np.mgrid[-r:r+1, -r:r+1]
    circle = np.sqrt(grid[0]**2 + grid[1]**2)
    out = circle<r
    
    return out
    

def measure_parameters(img, msk1, msk2, bg, buffer, ftime = 0.002, pix = 0.200):
    """
    carries out the measurement of the main parameters for the transients.
    
    I need more coffee

    Parameters
    ----------
    img : TYPE
        unthresheld image
    msk1 : TYPE
        threshold 1 - transient mask
    msk2 : TYPE
        threshold 2 - cell bg mask
    msk3 : TYPE
        threshold 3 - image bg mask
    ftime : float (OPTIONAL)
        frame time (default 0.02)

    Returns
    -------
    None.

    """
    rstart = np.where(np.max(msk1, axis = 0))[0].min() #tuple output element 0
    rend = np.where(np.max(msk1, axis = 0))[0].max()
    
    measure_region = np.arange(rstart, rend)
    print ("bg:{}".format(bg))
    img = img - bg
    
    """parameters as lists for now, pop into dict after return"""
    
    tstart = []
    bgf0 = []
    ttp = []
    tf50 = []
    td50 = []
    fdhm = []
    pff0 = []
    desync = []
    time_const_decay = [] #not used since transients does not appear to have exponential decay
    max_ror = []
    max_rod = []
    tplot = []
    
    
    for i in measure_region:
        line = img[:, i]
        #print(line)
        lmsk1 = line>filters.threshold_otsu(line)
        
        """basic params"""
        f0 = line[0:int(buffer[0]*0.6)].mean()#takes the initial ~60% frames before transient for bg est
        print (f0)
        fmax = line[lmsk1].max()
        crop_t = line * lmsk1
        peak_ffo = fmax/f0
        halfmax = (fmax-f0)/2 + f0
        duration_halfmax = np.sum(crop_t>halfmax)
        thresh5 = (fmax-f0)*0.05 + f0 # 5% peak as tstart
        
        #print (thresh5)
        
        """time params"""
        startmsk = line > thresh5
        
        linelab, obs = nd.label(startmsk)
        #print (linelab)
        #print (obs)
        sizes = nd.sum(startmsk, linelab, index=np.arange(1, obs+1))
        #print(sizes)
         
        tmax = np.where(line==fmax)[0][0] #np.where outputs a tuple
        #print ('tmax:{}'.format(tmax))
        #try:
        trans = linelab==(np.where(sizes==sizes.max())[0]+1)
        #except:
        #    trans = False
        #print(trans)
        if not np.any(trans):
            continue
        #ttstart = np.where(trans)[0][0]
        
        trans_hlf_mx = crop_t > ((fmax-f0)/2)
        
        
        tinit = np.where(trans)[0].min()
        ttf50 = np.where(trans_hlf_mx)[0].min()
        ttd50 = np.where(trans_hlf_mx)[0].max()
        
        tpror, tprod = peak_rate_of_rise(line/f0, startmsk)
        
        #try:
        ttrans = line/f0
        print(len(ttrans))
        tplot.append(ttrans)
            
        #except:
        #    pass
        
        
        """add stuff to the lists"""
        
        #tstart.append(ttstart*ftime)
        ttp.append((tmax - tinit)*ftime)
        tf50.append((ttf50 - tinit)*ftime)
        td50.append((ttd50 - tmax)*ftime)
        fdhm.append(duration_halfmax*ftime)
        pff0.append(peak_ffo)
        bgf0.append(f0)
        tstart.append(tinit*ftime)
        max_ror.append(tpror/ftime)
        max_rod.append(tprod/ftime)
    desync = np.std(tf50)/(len(measure_region)*pix)
    meantrans = np.array(tplot).mean(axis=0)
    
    """means"""
    print ('ttp:{}'.format(ttp))
    mttp = np.mean(ttp)
    mtf50 = np.mean(tf50)
    mtd50 = np.mean(td50)
    mfdhm = np.mean(fdhm)
    mpff0 = np.mean(pff0)
    mror = np.mean(max_ror)
    mrod = np.mean(max_rod)
    
    """std"""
    sttp = np.std(ttp)
    stf50 = np.std(tf50)
    std50 = np.std(td50)
    sfdhm = np.std(fdhm)
    spff0 = np.std(pff0)
    sror = np.std(max_ror)
    srod = np.std(max_rod)
    

    return (desync, 
            mttp, 
            sttp, 
            mtf50, 
            stf50, 
            mtd50, 
            std50, 
            mfdhm, 
            sfdhm, 
            mpff0, 
            spff0, 
            mror, 
            sror,
            mrod,
            srod, 
            meantrans)


def peak_rate_of_rise(line, mask):
    """
    leaverage ndimage to calculate gaussian smoothed first derivative. 
    Extract max and min for max ror and max rod

    Parameters
    ----------
    line : ndarray[N]
        linescan for measurmeent
    mask : ndarray[N], bool
        mask for specific transient in analysis

    Returns
    -------
    ror : Float
        max rate of rise
    rod : Float
        max rate of decline

    """
    
    derivative = nd.gaussian_filter(line, 3, order=1)
    filtd = derivative * mask
    ror = filtd.max()
    rod = filtd.min()
    
    return ror, rod


def append_params(inp, params, tr, ir):
    """
    christ, i need to do this differently....

    Parameters
    ----------
    inp : Dict
        dictionary for data input.
    params : Tuple
        tuple for new data to be appended.

    Returns
    -------
    inp : Dict
        input with additional values added.

    """
    inp['desync'].append(params[0])
    inp['mttp'].append(params[1])
    inp['sttp'].append(params[2])
    inp['mtf50'].append(params[3])
    inp['stf50'].append(params[4])
    inp['mtd50'].append(params[5])
    inp['std50'].append(params[6])
    inp['mfdhm'].append(params[7])
    inp['sfdhm'].append(params[8])
    inp['mpff0'].append(params[9])
    inp['spff0'].append(params[10])
    inp['mror'].append(params[11])
    inp['sror'].append(params[12])
    inp['mrod'].append(params[13])
    inp['srod'].append(params[14])
    inp['traceref'].append(tr)
    inp['imageref'].append(ir)
    
    return inp
    

def measure_events(img, data, ir, buffer = [20, 200], datadir = None, pix = 0.2):
    """
    NB: data is processed in place, not sure how much a dict 
    enjoys that whole muting thing though.

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    buffer : TYPE, optional
        DESCRIPTION. The default is [20, 200].

    Returns
    -------
    msk1 : TYPE
        DESCRIPTION.
    msk2 : TYPE
        DESCRIPTION.
    msk3 : TYPE
        DESCRIPTION.

    """
    RADIUS = 0.500 #in nm
    
    """
    note: tiffile metadata read is odd, loads half
    """
    
    #print("reading metadata")
    #meta = img.lsm_metadata
    #print("read metadata")
    #pix = meta['VoxelSizeX']*1e6
    #print("vox:{}".format(pix))
    #img = np.zeros(img.asarray()[:, 0, 0, :].shape)
    #img = img.asarray()[:20000, 0, 0, :]
    #print('img converted')
    print(img.shape)
    blurrad = RADIUS/pix
    
    structure = circle_mask(15)
    msk1, msk2, msk3, img = gen_masks(img, gauss_radius=blurrad)
    
    """a series of morphos to fill in holes and join small gaps"""
    print ('starting morphos')
    msk1 = nd.binary_opening(msk1, structure)
    msk1 = nd.binary_closing(msk1, structure)
    msk1 = nd.binary_erosion(msk1, circle_mask(5))
    msk2 = nd.binary_closing(msk2, structure)
    msk3 = nd.binary_erosion(msk3, circle_mask(5))
    
    if datadir:
        io.imsave(datadir + ir + '_img.tif', img)
        io.imsave(datadir + ir + '_m3.tif', msk3.astype(np.uint8))
        io.imsave(datadir + ir + '_m2.tif', msk2.astype(np.uint8))
        io.imsave(datadir + ir + '_m1.tif', msk1.astype(np.uint8))
    
    if np.sum(msk3) > 0:
        bg = np.mean(img[msk3])
    else: bg = 0
    print ('morphos complete')
    
    """label transients, crop out individual roi for analysis with buffer"""
    labels, nobs = nd.label(msk1)
    
    #plt.figure() #quick plot of some transients
    
    print("start processing loop")
    for i in np.arange(1, nobs+1):
        tr = ir + '_' + str(i)
        pxl_loc = np.where(labels==i) #output tuple e1 is length, e2 is width
        cropmax = pxl_loc[0].max() + buffer[1]
        cropmin = pxl_loc[0].min() - buffer[0]
        cropwl = pxl_loc[1].min()
        cropwr = pxl_loc[1].max()
        
            
            
        if (cropmin > 0) and (cropmax < len(msk1)) and (cropwr-cropwl>30):
            timg = img[cropmin:cropmax, cropwl:cropwr]
            tm1 = msk1[cropmin:cropmax, cropwl:cropwr]
            tm2 = msk2[cropmin:cropmax, cropwl:cropwr]
            if datadir:
                io.imsave(datadir + tr + '_img.tif', timg.astype(np.float32))
                io.imsave(datadir + tr + '_m1.tif', tm1.astype(np.uint8))
                
            params = measure_parameters(timg, tm1, tm2, bg, buffer = [20, 200], ftime = 0.00065, pix = pix)
            print(params)
            append_params(data, params, tr, ir)
            t = np.arange(len(params[-1]))*0.00065
            plt.plot(t,params[-1])
            
        else:
            pass
    
    return labels, msk2, msk3, data

def batch_process(indir, outdir, datadir = None):
    
    data = {'desync':[], 
            'mttp':[], 
            'sttp':[], 
            'mtf50':[], 
            'stf50':[], 
            'mtd50':[], 
            'std50':[], 
            'mfdhm':[], 
            'sfdhm':[], 
            'mpff0':[], 
            'spff0':[], 
            'mror':[], 
            'sror':[],
            'mrod':[],
            'srod':[],
            'traceref':[],
            'imageref':[]}
    
    for fname in os.listdir(indir):
        if fname.endswith('.lsm'):
            img = tif.TiffFile(indir + fname)
            meta = img.lsm_metadata
            img.close()
            img = io.imread(indir + fname)
            print("read metadata")
            pix = meta['VoxelSizeX']*1e6
            imgref = fname.split('.lsm')[0]
            m1, m2, m3, data = measure_events(img[0], data, imgref ,buffer = [100, 100], datadir = datadir, pix = pix)

    
    df = pd.DataFrame(data)
    df.to_csv(outdir + "transient_output.csv")
    
    return df

if __name__ == '__main__':
    
    """for the love of god I need to make this into a class"""
    
    data = {'desync':[], 
            'mttp':[], 
            'sttp':[], 
            'mtf50':[], 
            'stf50':[], 
            'mtd50':[], 
            'std50':[], 
            'mfdhm':[], 
            'sfdhm':[], 
            'mpff0':[], 
            'spff0':[], 
            'mror':[], 
            'sror':[],
            'mrod':[],
            'srod':[],
            'traceref':[],
            'imageref':[]}
    
    indir = '/Users/sarahfong/Document/Waves/NIL/Control/15.03_NIL/linescans/'
    outdir = '/Users/sarahfong/Document/Waves/NIL/Control/15.03_NIL/output/'
    datadir = '/Users/sarahfong/Document/Waves/NIL/Control/15.03_NIL/rois/'
    df = batch_process(indir, outdir, datadir)