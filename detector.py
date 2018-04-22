import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def morphology_diff(contrast_green, clahe):
    #apply open / closing morphology
    #1st
    open1 = cv2.morphologyEx(contrast_green, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    close1 = cv2.morphologyEx(open1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    #2nd
    open2 = cv2.morphologyEx(close1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    close2 = cv2.morphologyEx(open2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    #3rd
    open3 = cv2.morphologyEx(close2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
    close3 = cv2.morphologyEx(open3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)	
    #make diff between contrast_green & blured vision
    contrast_morph = cv2.subtract(close3, contrast_green)
    return clahe.apply(contrast_morph)

def remove_noise(morph_image):
    ret, thr = cv2.threshold(morph_image,15,255,cv2.THRESH_BINARY)	
    mask = np.ones(morph_image.shape[:2], dtype="uint8") * 255	
    im2, contours, hierarchy = cv2.findContours(thr.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) <= 200:
            cv2.drawContours(mask, [cnt], -1, 0, -1)			
    im = cv2.bitwise_and(morph_image, morph_image, mask=mask)
    ret,fin_thr = cv2.threshold(im,15,255,cv2.THRESH_BINARY_INV)			
    new_img = cv2.erode(fin_thr, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)	
    return new_img

def detect_vessel(org_image):
    copy_org_image = org_image.copy()
    #make split of red green blue colors
    blue, green, red = cv2.split(org_image)
    #create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    contrast_green = clahe.apply(green)
    #get image after morph - blured & clahe
    morph_image = morphology_diff(contrast_green, clahe)
    #remove noise
    clear_image = remove_noise(morph_image)

    return clear_image


if __name__ == "__main__":
    data_catalog = "data"
    files_names = [x for x in os.listdir(data_catalog) if os.path.isfile(os.path.join(data_catalog,x))]
    files_names.sort()
    out_catalog = "out"
    for file_name in files_names:
        out_name = file_name.split('.')[0]
        org_image = cv2.imread(data_catalog + '/' + file_name)
        vessel_image = detect_vessel(org_image)
        cv2.imwrite(out_catalog + '/' + out_name + ".JPG", vessel_image)
        
