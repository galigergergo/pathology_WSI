import cv2
import pickle
import numpy as np


img = cv2.imread('test.png')

with open('outline_mask.pkl', 'rb') as handle:
    mask = pickle.load(handle)

dst = cv2.inpaint(img, mask.astype('uint8') * 255, 3, cv2.INPAINT_TELEA)

cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('inpainted.png', img)
