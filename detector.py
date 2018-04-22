import os
import cv2
import sys
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

def detect_vessel(org_image):
    copy_org_image = org_image.copy()
    #make split of red green blue colors
    blue, green, red = cv2.split(org_image)
    #create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    contrast_green = clahe.apply(green)

    morph_image = morphology_diff(contrast_green, clahe)
    
    plt.imshow(morph_image)
    plt.show()

    return 0


if __name__ == "__main__":
    data_catalog = "data"
    files_names = [x for x in os.listdir(data_catalog) if os.path.isfile(os.path.join(data_catalog,x))]
    files_names.sort()
    out_catalog = "out"
    for file_name in files_names:
        out_name = file_name.split('.')[0]
        org_image = cv2.imread(data_catalog + '/' + file_name)
        vessel_image = detect_vessel(org_image)
        sys.exit()
        cv2.imwrite(out_catalog + '/' + out_name + ".JPG", vessel_image)
        