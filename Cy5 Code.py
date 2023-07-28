import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data_file = input('Enter Data File: \n')

wavelength, NP, Cy5 = np.loadtxt(data_file, skiprows=1, delimiter = ',', unpack=True)

OFFSET = 200
START = 250
END = 600
wavelength_start = START-OFFSET
wavelength_end = END-OFFSET
wavelength_short = wavelength[wavelength_start:wavelength_end]
NP_short = NP[wavelength_start:wavelength_end]
Cy5_short = Cy5[wavelength_start:wavelength_end]

DOS = np.sum((NP_short-Cy5_short)**2)

precision = 1000000

def scale_NP():
    iDOS = DOS
    for i in range(0,precision,1):
        num = i/precision
        NP_scaled = NP_short * num
        new_DOS = np.sum((NP_scaled-Cy5_short)**2)
        if new_DOS < iDOS:
            iDOS = new_DOS
            factor = num
    print('The Scale Factor is ' + str(factor) + '\n '\
          + 'With a DOS of ' + str(iDOS))
    return NP * factor

best_fit = scale_NP()

mpl.rcParams['figure.dpi'] = 1200

#plt.plot(wavelength,NP,marker='x',markersize=0.5,label='NP')
plt.plot(wavelength,Cy5,marker='x',markersize=0.5,label='Anti-biotin IgG-MSNP After Cy5 Wash')
plt.plot(wavelength,best_fit,marker='x',markersize=0.5,label='Anti-biotin IgG-MSNP')
plt.legend()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorbance')
plt.title('Absorbance Curves of ' + data_file[:-4])