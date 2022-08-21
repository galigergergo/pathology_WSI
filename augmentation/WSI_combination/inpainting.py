import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('./test/test.png')

with open('./test/outline_mask.pkl', 'rb') as handle:
    mask = pickle.load(handle)

plt.imshow(mask)
plt.show()

dst = cv2.inpaint(img, mask.astype('uint8') * 255, 3, cv2.INPAINT_TELEA)

cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('./test/inpainted.png', img)
