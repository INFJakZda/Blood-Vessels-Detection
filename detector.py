import os
import cv2
import sys
import matplotlib.pyplot as plt

def detect_vessel(org_image):
    copy_org_image = org_image.copy()
    blue, green, red = cv2.split(org_image)
    for clip in range(1, 40, 1):
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
        contrast_green = clahe.apply(green)
        cv2.imwrite("out" + '/' + str(clip) + ".JPG", contrast_green)
    

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
        