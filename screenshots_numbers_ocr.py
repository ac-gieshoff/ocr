# -*- coding: utf-8 -*-
"""
ocr for apple segments
author: gies
version: 10.12.2020
this is a demo version. 
"""

#Aim of this script: 
#The idea is to read screenshots that contain tables with numbers, to extract the numbers and to store them in a txt-file.
#There may be several screenshots that belong to one table. These shoudld be saved in the same txt-file. 
#Overlapping rows should be deleted.
#screenshots: Tables in white font on black ground
#Warning: Rows that are cut-off ore often not properly parsed.
# See also: https://nanonets.com/blog/ocr-with-tesseract/

#installation of packages via anaconda prompt (pip install pkg), attention: cv2 is opencv
#from PIL import Image
import pytesseract
import cv2
import numpy as np
import re
import os
import glob
#from os.path import join
#specify where teseract is installed
pytesseract.pytesseract.tesseract_cmd = r'wherever Tesseract is installed'

##a few functions or definitions to improve image quality
# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# schwarz und weiss tauschen
def invert(image):
    image = (255-image)
    return(image)

#dilation
def dilate(image):
    kernel = np.ones((2,2),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((2,2),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 


#thresh method seems to work best in combination with psm 12
#whitelist: should only recognize digits and :
#oem: different engins, but no difference in performance on my testdata
#psm: page segmentation mode. Different performances: 1 and 3 are similar, 11 and 12 are similar and 4 and 6, 7-10 and 2 were empty

#let's get started
#define main folder
wd = whateveryourfolder+'\\segments'

#list all subfolder

#list all sub-subfolders

#read jpg and png files...
jpg = glob.glob(wd + '\\**\\*.jpg', recursive=True)
png = glob.glob(wd + '\\**\\*.png', recursive=True)
PNG = glob.glob(wd + '\\**\\*.PNG', recursive=True)
files = jpg+png+PNG
files = sorted(files)


#loop through folder
for f in files:
    
    #extract directory, i.e. remove last part
    path = os.path.dirname(f)
    #extract name of participant
    basename = os.path.basename(f)
    name = basename.split(sep='_')[0]
    number = re.search('(\d)[^\d]*$', basename).group(1)

    filename=path+'\\'+name+'_sgmt.txt'
    
    #open image
    image = cv2.imread(f)
    
    #convert into greyscale
    gray = get_grayscale(image)
    #invert white and black
    gray = invert(gray)
    #improve contrasts
    thresh = thresholding(gray)
    thresh = erode(thresh)
    
    
    #define configuration
    my_config=r'--psm 12 --oem 3 -c tessedit_char_whitelist=0123456789:' 
    #convert to string and save content in t
    t = pytesseract.image_to_string(thresh, lang='eng', config=my_config)
    #convert t to list (=vector), splits after each space
    t2 = t.split()
    
    #manipulate content: keep only values with ':' in it
    pattern = re.compile(r".*:.*") # Create the regular expression to match
    t3 = [i for i in t2 if pattern.match(i)]
    
    #unfortunately, we may have more than one screenshot per table
    #all screenshots from one table should be joined to one textfile
    
    #check whether there is already a textfile for the same table
    l = glob.glob(filename, recursive =True)
    #if not, we can simply save t3 as a textfile
    if number == '1' or os.path.isfile(filename)==False:
        print("no file")
    #remove first item (=time when screenshot was taken)
   
        t3.pop(0)
    
    #transform to array (=dataframe)
        a = np.array(t3)
    
    #build second column with segments
        l=len(t3)
        segs = np.array(range(1,l+1))
        a2 = np.array(list(zip(a,segs)))
    
    #save as textfile: first open the file 
    #file = open('C:/Users/acgie/Desktop/testfile.txt','w') 
    #write the array a2 to file
      
        with open(filename, "w") as file:
            for line in a2:
                file.write(" ".join(line) + "\n")
    
    #close file
        file.close() 
        
        # if there is already a textfile, we need to append the new data to the existing textfile.
        #however, there is probably an overlap which should be avoided
    else :
        print("file exists")
        t3.pop(0)
        obj = open(path+'\\'+name+'_sgmt.txt', 'r')
        #retrieve information from existing textfile to check for overlap
        result=[]
        time = []
        lines=obj.readlines()
        for x in lines:
            #segments
            result.append(x.split(' ')[1])
            #durations
            time.append(x.split(' ')[0])
            obj.close()

        #we need the index of all matches in time and t3
        reduced = [i for i, item in enumerate(t3) if item in time]
        
        #if there is no overlap, everything should be added
        try:
            #otherwise, delete matches
            #there might be errors in ocr, let's built a new list
            #everything up to the highest index should be removed
            range_reduced = list(range(0,max(reduced)+1))
            
            #delete the matches 
            for e in sorted(range_reduced, reverse = True):  
                del t3[e] 
        except:
            print('no overlap')
            t3 = t3

        # define next segment number in order to obtain consecutive numbering of all segments: 
        #retrieve the highest segment number in the textfile
        result = [int(i) for i in result] 
        max_val = max(result)
        
        #transform to array (=dataframe)
        a = np.array(t3)
    
        #build second column with segments
        l=len(t3)
        #segment numbers from max_val+1 (subsequent number) untill max_val+ length t3
        segs = np.array(range(max_val+1,max_val+l+1))
        a2 = np.array(list(zip(a,segs)))
    
    #save as textfile: first open the file 
    #file = open('C:/Users/acgie/Desktop/testfile.txt','w') 
    #append the array a2 to file
        
        with open(filename, "a") as file:
            for line in a2:
                file.write(" ".join(line) + "\n")