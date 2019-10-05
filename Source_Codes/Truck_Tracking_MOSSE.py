import numpy as np
import cv2 as cv

def toFreqview(Img) :
    
    rows, cols = Img.shape
    m = cv.getOptimalDFTSize( rows )
    n = cv.getOptimalDFTSize( cols )
    padded = cv.copyMakeBorder(Img, 0, m - rows, 0, n - cols, cv.BORDER_CONSTANT, value=[0, 0, 0])
    
    planes = [np.float32(padded), np.zeros(padded.shape, np.float32)]
    complexImg = cv.merge(planes)         # Add to the expanded another plane with zeros
    
    cv.dft(complexImg, complexImg) 
    
    cv.split(complexImg, planes)                  # planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    cv.magnitude(planes[0], planes[1], planes[0]) # planes[0] = magnitude
    magImg = planes[0]
    
    matOfOnes = np.ones(magImg.shape, dtype=magImg.dtype)
    cv.add(matOfOnes, magImg, magImg) #  switch to logarithmic scale
    cv.log(magImg, magImg)
    
    magI_rows, magI_cols = magImg.shape
    # crop the spectrum, if it has an odd number of rows or columns
    magImg = magImg[0:(magI_rows & -2), 0:(magI_cols & -2)]
    cx = int(magI_rows/2)
    cy = int(magI_cols/2)
    q0 = magImg[0:cx, 0:cy]         # Top-Left - Create a ROI per quadrant
    q1 = magImg[cx:cx+cx, 0:cy]     # Top-Right
    q2 = magImg[0:cx, cy:cy+cy]     # Bottom-Left
    q3 = magImg[cx:cx+cx, cy:cy+cy] # Bottom-Right
    tmp = np.copy(q0)               # swap quadrants (Top-Left with Bottom-Right)
    magImg[0:cx, 0:cy] = q3
    magImg[cx:cx + cx, cy:cy + cy] = tmp
    tmp = np.copy(q1)               # swap quadrant (Top-Right with Bottom-Left)
    magImg[cx:cx + cx, 0:cy] = q2
    magImg[0:cx, cy:cy + cy] = tmp
    
    #cv.normalize(magImg, magImg, 0, 1, cv.NORM_MINMAX) # Transform the matrix with float values into a
    
    return magImg

def dftQuadSwap(img) :
    
    img_rows, img_cols = img.shape
    
    cx = int(img_rows / 2)
    cy = int(img_cols / 2)

    q0 = img[0:cx, 0:cy]                # Top-Left - Create a ROI per quadrant
    q1 = img[cx:img_rows, 0:cy]         # Top-Right
    q2 = img[0:cx, cy:img_cols]         # Bottom-Left
    q3 = img[cx:img_rows, cy:img_cols]  # Bottom-Right
    
    tmp = np.copy(q0)                   # swap quadrants (Top-Left with Bottom-Right)
    img[0:cx, 0:cy] = q3
    img[cx:cx + cx, cy:cy + cy] = tmp
    tmp = np.copy(q1)                   # swap quadrant (Top-Right with Bottom-Left)
    img[cx:cx + cx, 0:cy] = q2
    img[0:cx, cy:cy + cy] = tmp
    
    return img

def sobelAVG(Img) :

    Img = cv.GaussianBlur(Img, (3, 3), 0, 0, cv.BORDER_DEFAULT)

    sobelx = cv.Sobel(Img, cv.CV_32F, 1, 0)
    abs_grad_x = cv.convertScaleAbs(sobelx)

    sobely = cv.Sobel(Img, cv.CV_32F, 0, 1);
    abs_grad_y = cv.convertScaleAbs(sobely)

    # Take the Average of the two directions
    sobel = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    return sobel

def complexDivide(G, F) :
   
    top = cv.mulSpectrums(G, F, cv.DFT_COMPLEX_OUTPUT, True) # Top is G*(F conjugate)
    bot = cv.mulSpectrums(F, F, cv.DFT_COMPLEX_OUTPUT, True) # Bot is F*(F conjugate)
    
    # Bottom is strictly real and we should divide real and complex parts by it
    botRe = cv.split(bot)
    botRe[1] = botRe[0].copy()
    bot = cv.merge(botRe)

    # Do the actual division
    H = np.divide(top, bot)
    
    return H

#-----Begin Coding-----

# Load in Video File
cap = cv.VideoCapture('C:/Users/bjperry/Research/Truck_Tracking/Video_1.mp4')
width = 1280
length = 720

while(cap.isOpened()):
    
    # Load in Video one Frame at a Time
    _, Img = cap.read()
    Img = cv.cvtColor(Img, cv.COLOR_BGR2GRAY)
    Img = cv.resize(Img, (width, length))
    
    # Manually Select the Truck
    r = cv.selectROI(Img, False)
    
    # Creates Kernal Gauss Point the size of the Video and places it at
    # the center. Then Converts to Frequency Domain
    cp = np.array([int(r[0]+r[2]/2), int(r[1]+r[3]/2)])
    kernelX = cv.getGaussianKernel(11, 11, cv.CV_32F)
    kernelY = cv.getGaussianKernel(11, 11, cv.CV_32F)
    Gauss = kernelX * np.transpose(kernelY)
    Gauss = cv.normalize(Gauss, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
    Gauss = cv.copyMakeBorder(Gauss, cp[1]-5, length-cp[1]-6, cp[0]-6, width-cp[0]-5, cv.BORDER_CONSTANT, 0)
    location = [np.float32(Gauss), np.zeros(Gauss.shape, np.float32)]
    locationRI = cv.merge(location) 
    cv.dft(locationRI, locationRI)
    
    # Crop the Truck from the Frame
    imCrop = Img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    sobel = sobelAVG(imCrop)
    sobel = cv.copyMakeBorder(sobel, r[1], length-r[1]-r[3], r[0], width-r[0]-r[2], cv.BORDER_CONSTANT, 0)
    
    # Begin to Track the Truck
    sobel = [np.float32(sobel), np.zeros(sobel.shape, np.float32)]
    sobelRI = cv.merge(sobel) 
    cv.dft(sobelRI, sobelRI)

    # Complex Divide both in Frequency Domain
    filterRI = complexDivide(locationRI, sobelRI)
        
    # Transform Exact Filter to Spacial Domain
    filt = np.empty(Gauss.shape)
    filt = cv.idft(filterRI)
    filt = cv.magnitude(filt[:,:,0],filt[:,:,1])
    filt = dftQuadSwap(filt)
     
    # Display cropped image
    filt = cv.normalize(filt,0,1)
    cv.imshow("Image", filt)
    cv.waitKey(0)
   
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
