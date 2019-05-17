from astropy.io import fits
import pylab as plt
import numpy as np
import scipy.optimize as opt

import tkinter as tk
from tkinter import ttk

from specutils import Spectrum1D
from astropy import units as u
from specutils import SpectralRegion
from specutils.analysis import equivalent_width
from specutils.fitting import fit_generic_continuum
from astropy.modeling import models
from specutils.fitting import fit_lines


wavelengths = np.array([])
pixels = np.array([])
sigma = np.array([])
# img = None
# spec = None
# y_continuum_fitted = None
# intensity_box_pix = None
# intensity = None
# wavelengthEquation = None
# spec_normalized = None






'''
load in the fits image you would like to look at
'''


fits_image = fits.open('alphagem.062.fits') #read the fits file making sure it is in the same directory

fits_image.info()  # look at the contents of the image 

#each entry is an HDU, header data unit, and has information on the size of the image

hdu = fits_image[0]  # take the Primary HDU


## you could also look at the header to get information about RA, DEC, Grating and Tilt
#hdu.header  




''' 
now display the fits image 

if you hover your mouse over each button at the bottom of the plot it will tell you what each does
you will most likely use the square to zoom in to particular areas and the home to retern to the
full view of the image
'''

img = hdu.data

plt.figure(figsize = (6,6)) # create a figure and set the size of the display 

## we will display the image with a color scale similar to what we can manipulate in DS9
# you may change vmin and vmax to alter the stretch of the color scale. closer numbers = fewer features usually
plt.imshow(img, origin = 'lower', vmin = 0, vmax = 8000,cmap='gray')  
plt.xlabel('pixels')
plt.ylabel('pixels')
plt.title('Alpha Gemini 2D Spectrum')
plt.colorbar()
plt.show()





'''
where do you want to draw your box?

use the interactive spectrum above to select where you want your box to start (start) and how 
wide you want it to be (width)
'''

## here we definte the column averaging function
## sometimes a wider box helps to reduce some noise but check what your 1D spectrum looks like at different 
## box widths to make your decision
def column_avg(img,start,width):
    column_sum = 0
    i = start
    
    while i<start+(width):
        column_sum += img[:,i]    #add all of the columns in your chosen box together
        
        i+=1
        
    column_avg = column_sum/(width)    # take the average of all those columns 
    
    return column_avg





'''
using the column_avg function on your 2D image take a slice 

now you may look at your 1D spectrum
'''

## here I have chosen to extract a box that begins at pixel 233 and is 8 pixels wide, which 
## encapsulates the full target spectrum
intensity_box_pix = column_avg(img,233,8)

# the pixels go from 0 to the length of the image youve taken
pixels = np.linspace(0,len(intensity_box_pix),len(intensity_box_pix))

plt.figure()
plt.plot(pixels,intensity_box_pix) # plot your pixels vs the intensity in your extracted box
plt.xlabel('pixels')
plt.ylabel('intensity')
plt.title('Alpha Gemini Uncalibrated 1D Spectrum')










def func(x, a, b, c):
	# used in calibration equation
	return a + b*x + c*x*x

def onclick(event): 
	# registers click event assigning a wavelength to a pixel value associated with a spectral feature
    print('xdata=%f' %(event.xdata))
    popup(str(event.xdata))


def popup(xdata):
	"""
	popup dialog from click event
	"""
	popupwindow = tk.Tk()
	popupwindow.wm_title("Log corresponding wavelength")

	# text labeling the dialog's function
	label = ttk.Label(popupwindow, text="What wavelength corresponds to the pixel " + xdata + "?")
	label.pack(side="top", fill="x", pady=10)

	label = ttk.Label(popupwindow, text="If entering manually, use the format 'label angstromWavelength A'.")
	label.pack(side="top", fill="x", pady=10)

	# dropdown or manual enter box for wavelength
	textfield = ttk.Combobox(popupwindow)
	textfield['values']=['H-alpha 6463 A', 'OIII 5007 A', 'H-beta 4861 A', 'HgNe 6402.0 A', 'HgNe 6334.0 A', 'HgNe 6266.0 A', 'HgNe 6143.0 A', 'HgNe 6096.0 A', 'HgNe 5945.0 A', 'HgNe 5852.0 A', 'HgNe 5791.0 A', 'HgNe 5770.0 A', 'HgNe 5461.0 A']
	textfield.pack()

	# button that stores the wavelength and closes the dialog
	B1 = ttk.Button(popupwindow, text="Add Wavelength", command = lambda: addWavelength(xdata, textfield.get(), popupwindow))
	B1.pack()

def addWavelength(xdata, wavelength, pop):
	"""
	adds the pixel value and corresponding wavelength to lists that will be used in the fitting
	"""
	global wavelengths
	global pixels
	global sigma

	wavelength = wavelength.split()
	wavelength = wavelength[1]

	wavelengths = np.append(wavelengths, float(wavelength))
	pixels = np.append(pixels, float(xdata))
	sigma = np.append(sigma, 1.0)

	pop.destroy()

def column_avg(img,start,width):
    column_sum = 0
    i = start
    
    while i<(start+width):
        column_sum += img[:, i]    #add all of the columns in your chosen box together
        
        i+=1
        
    column_avg = column_sum/(width)    # take the average of all those columns 
    
    return column_avg

def calibration(imgName):
	"""
	main function for generating wavelength calibration equation. 

	Takes name of image used for wavelength calibration.

	Returns coefficients [a,b,c] of f(x)=a + b*x + c*x*x which solves for wavelength from pixels. C should be very small.
	"""
	global img
	hdus = fits.open(imgName)
	img = hdus[0].data

	plt.clf()
	plt.imshow(img, origin = 'lower', vmin = 0, vmax = 10000)
	plt.xlabel('pixels')
	plt.ylabel('pixels')
	plt.colorbar()

	plt.show()

	fig = plt.figure()

	###CHANGE THE STARTING COLUMN AND WIDTH OF THE SAMPLE BOX HERE###
	plt.plot(column_avg(img, 233, 8))

	cid = fig.canvas.mpl_connect('button_press_event', onclick)

	plt.show()

	#sample data for h-alpa spectrum
	global pixels, wavelengths, sigma
	pixels = np.array([71.0, 333.0, 359.0])
	wavelengths = np.array([6463.0, 5007.0, 4861.0])
	sigma = np.array([1.0, 1.0, 1.0])

	#sample data for HgNe spectrum
	# pixeldata = np.array([98.0, 110.0, 121.0, 141.0, 150.0, 175.0, 190.0, 200.0, 205.0, 257.0])
	# wavelengthdata = np.array([6402.0, 6334.0, 6266.0, 6143.0, 6096.0, 5945.0, 5852.0, 5791.0, 5770.0, 5461.0])
	# sigma = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

	# x0 = np.array([0.0, 0.0, 0.0])

	# print(opt.curve_fit(func, pixels, wavelengths, x0, sigma)[0])

	# for i in range(len(wavelengths)):
	# 	print(str(pixels[i]) + " : " + str(wavelengths[i]))

	# wavelengthEquation = opt.curve_fit(func, pixels, wavelengths, x0, sigma)[0] # a,b,c of wavelength calibration equation
calibration('m42spec.078.fits')

def final():
    x0 = np.array([0.0, 0.0, 0.0])
    
    print('pixel   :   wavelength')
    for i in range(len(wavelengths)):
        
        print(str(pixels[i]) + "  :  " + str(wavelengths[i]))
        
    return opt.curve_fit(func, pixels, wavelengths, x0, sigma)[0]
fit_params = final()


'''
we use our selected points to fit a quadratic function to transfer pixel values into wavelengths 
'''
## for a quadratic equation y = ax^2 + bx + c
a = fit_params[0]
b = fit_params[1]
c = fit_params[2]

print('quadratic parameters:','a=',a,'b=',b,'c=',c)

pixels = np.linspace(0,len(intensity_box_pix),len(intensity_box_pix))




'''
we put our pixel values back into our equation with the parameters we extracted 
to get an array of calibrated wavelengths
'''

wavelengths = func(pixels, a, b, c)

## we will use general intensity units from the sliced 2D spectrum
intensity = intensity_box_pix



from specutils import Spectrum1D
from astropy import units as u
from specutils import SpectralRegion
from specutils.analysis import equivalent_width
from specutils.fitting import fit_generic_continuum
from astropy.modeling import models
from specutils.fitting import fit_lines




'''
now that we have our wavelengths and intensities we want to read them into a form that we can
use with specutils

we use the class Spectrum1D which has spectral and flux axes - however note that we must set these values
with arrays of astropy.units.Quantities which are arrays that carry a float AND a unit 

you see we use Angstroms for the wavelength unit, but we have not performed a flux calibration, and will later
normalize the spectrum so we will use Janskys (Jy) the non-SI unit of spectral flux as a space holder for if we 
ever choose to calibrate the flux
'''
spec = Spectrum1D(spectral_axis=wavelengths*u.Angstrom, flux=intensity*u.Jy)


## plot the spectrum we defined as spec
plt.figure()
plt.plot(spec.spectral_axis, spec.flux)    # x axis is wavelength y is intensity
plt.xlabel('wavelength (Å)')
plt.ylabel('flux')
plt.grid(True)
plt.title('1D Spectrum of Alpha Gemini')
plt.show()




'''
since stars have a black body continuum spectrum, we would like to subtract that out to look specifically
at the absorbtion and emission lines

we will use specutils fit_generic_continuum to fit a continuum 
''' 

## use specutils.fit_generic_continuum to build a conintuum function to the data
continuum_fit = fit_generic_continuum(spec)
print(continuum_fit)


## plug the wavelengths into your continuum function
intensity_continuum_fitted = continuum_fit(wavelengths*u.Angstrom)

## Plot the original spectrum and the fitted and the continuum
plt.figure()
plt.plot(spec.spectral_axis, spec.flux, label='spectrum')
plt.plot(spec.spectral_axis, intensity_continuum_fitted, label='continuum')
plt.xlabel('wavelength (Å)')
plt.ylabel('flux')
plt.title('Continuum Fitting')
plt.grid(True)
plt.legend()
plt.show()





'''
subtract the contiuum from your spectrum so that you have a more or less flat spectrum centered around 0

plot that continuum subtracted spectrum
'''
spec_normalized = spec - intensity_continuum_fitted

# Plot the subtracted spectrum
plt.figure()
plt.plot(spec_normalized.spectral_axis, spec_normalized.flux)
plt.title('Continuum Subtracted Spectrum')
plt.xlabel('wavelength (Å)')
plt.ylabel('flux')
plt.grid(True)
plt.show()






'''
we will use specutils find_lines_threshold to pickout the emission and absorbtion lines

you will need to find a region in your spectrum where it looks like noise to set as a "noise_region"
'''

from specutils.manipulation import noise_region_uncertainty

## looking at my plot, it seems as though between 5600 and 5800 A there is just noise
noise_region = SpectralRegion(5600*u.Angstrom, 5800*u.Angstrom) # use SpectralRegion class to definte this noise region

## make a new spectrum where we will look for peaks that attaches some noise uncertainty
peak_spectrum = noise_region_uncertainty(spec_normalized, noise_region)


from specutils.fitting import find_lines_threshold

'''
the noise_factor is the number you would like to multiply the uncertainty by to say "a peak above this threshold
is significant". a higher noise factor means fewer peaks are selected

here I set the noise factor to 1.5* the unceratinty but we can see that it seems a few peaks are missing 
'''
lines = find_lines_threshold(peak_spectrum, noise_factor=1.5)


## this will give you which lines the program finds both emission and absorbtion, the wavelength center of the peak,
## and the array index of the center
emissions = lines[lines['line_type'] == 'emission']
absorbtions = lines[lines['line_type'] == 'absorption'] 

print(emissions )

print(absorbtions )






from astropy.modeling import models
from astropy import units as u

from specutils.spectra import Spectrum1D
from specutils.fitting import fit_lines


all_peak_fits = []
all_gaus_fits = []

for i in range(len(absorbtions)):
    #gaussian = models.Gaussian1D(-100, absorbtions[i][0], 20)

   
    # Fit the spectrum
    gaus_init = models.Gaussian1D(amplitude=-100*u.Jy, mean=absorbtions[i][0], stddev=20*u.Angstrom)
    gaus_fit = fit_lines(spec_normalized, gaus_init, window = 
                         SpectralRegion(absorbtions[i][0]-100*u.Angstrom,absorbtions[i][0]+100*u.Angstrom))
    
    peak_fit = gaus_fit(spec_normalized.spectral_axis)
    all_gaus_fits.append(gaus_fit)
    all_peak_fits.append(peak_fit)

    print('Absorbtion peak:',i,'\n','mean:',gaus_fit.mean[0],'Angstrom', '\n','FWHM',gaus_fit.fwhm)


plt.figure()
plt.plot(spec_normalized.spectral_axis, spec_normalized.flux)
for i in range(len(absorbtions)):
    plt.plot(spec_normalized.spectral_axis, all_peak_fits[i],label='$\lambda$ = %d Å' %all_gaus_fits[i].mean[0])
plt.title('Spectrum with Fits on each Peak')
plt.grid(True)
plt.legend( bbox_to_anchor=(0.97, 0.8),prop={'size': 6} )
plt.show()











# def openFits():
# 	# load in the fits image you would like to look at

# 	fits_image = fits.open('alphagem_062.fits') #read the fits file making sure it is in the same directory

# 	fits_image.info()  # look at the contents of the image 

# 	#each entry is an HDU, header data unit and has information on the size

# 	hdu = fits_image[0]  # take the Primary HDU
# 	#hdu.header  # look at the header to get information about RA, DEC, Grating and Tilt

# 	return hdu


# # def fitsDisplay(hdu):
# # 	global img
# # 	img = hdu.data

# # 	plt.figure(figsize = (6,6)) # create a figure and set the size
# # 	plt.imshow(img, origin = 'lower', vmin = 0, vmax = 8000,cmap='gray')  
# # 	plt.xlabel('pixels')
# # 	plt.ylabel('pixels')
# # 	plt.colorbar()
# # 	plt.show()

# '''
# where do you want to draw your box?

# use the interactive spectrum above to select where you want your box to start (start) and how 
# wide you want it to be (width)
# '''


# def get1DSpectra():
# 	global img
# 	global intensity_box_pix
# 	intensity_box_pix = column_avg(img,233,8)  

# 	plt.figure()
# 	plt.plot(intensity_box_pix)
# 	plt.xlabel('pixels')
# 	plt.ylabel('intensity')

# def wavelengthSolvePlot():
# 	global wavelengths, spec, intensity
# 	wavelengths = np.linspace(0,len(intensity_box_pix[10:]),len(intensity_box_pix[10:]))
# 	intensity = intensity_box_pix[10:]

# 	spec = Spectrum1D(spectral_axis=wavelengths*u.AA, flux=intensity*u.Jy)


# 	# Plot the original spectrum and the fitted and the continuum
# 	plt.figure()
# 	plt.clf()
# 	plt.plot(spec.spectral_axis, spec.flux)
# 	plt.xlabel('wavelength')
# 	plt.ylabel('flux')
# 	plt.grid(True)
# 	plt.show()

# def continuumFit():
# 	global intensity, y_continuum_fitted, spec

# 	# y_continuum = 10 * np.exp(-0.5 * (np.array(wavelengths) - 5.6)**2 / 4.8**2)
# 	y_continuum = np.array(lambda x: wavelengthEquation[0] + wavelengthEquation[1]*x + wavelengthEquation[2]*x*x for x in intensity)
# 	# intensity += y_continuum

# 	g1_fit = fit_generic_continuum(spec)

# 	y_continuum_fitted = g1_fit(wavelengths*u.Angstrom)

# 	# Plot the original spectrum and the fitted and the continuum
# 	plt.figure()
# 	plt.plot(spec.spectral_axis, spec.flux, label='spectrum')
# 	plt.plot(spec.spectral_axis, y_continuum_fitted, label='continuum')
# 	plt.xlabel('pixels')
# 	plt.ylabel('flux')
# 	plt.title('Continuum Fitting')
# 	plt.grid(True)
# 	plt.legend()

# def normalizeSpectra():
# 	global spec, y_continuum_fitted, spec_normalized
# 	spec_normalized = spec - y_continuum_fitted

# 	# Plot the normalized spectrum
# 	plt.figure()
# 	plt.plot(spec_normalized.spectral_axis, spec_normalized.flux)
# 	plt.title('Continuum subtracted')
# 	plt.grid(True)


# from specutils.manipulation import noise_region_uncertainty
# from specutils.fitting import find_lines_threshold

# def noise():
# 	global spec_normalized

# 	noise_region = SpectralRegion(200*u.Angstrom, 300*u.Angstrom)
# 	spectrum = noise_region_uncertainty(spec_normalized, noise_region)


# 	lines = find_lines_threshold(spectrum, noise_factor=3)

# 	print(lines[lines['line_type'] == 'emission'] )

# 	print(lines[lines['line_type'] == 'absorption'] )



# calibration('m42spec.078.fits')
# get1DSpectra()
# wavelengthSolvePlot()
# continuumFit()
# normalizeSpectra()
# noise()