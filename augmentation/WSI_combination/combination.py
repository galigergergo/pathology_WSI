from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
from time import sleep


def get_random_mask(nr_masks):
    mask_ind = random.randint(0, nr_masks - 1)
    with open('../../resources/masks/pkl/mask_%d.pkl' % mask_ind,
              'rb') as handle:
        mask = pickle.load(handle)
    with open('../../resources/masks/pkl/outline_%d.pkl' % mask_ind,
              'rb') as handle:
        outline = pickle.load(handle)
    return mask, outline


def generate_random_translation(img_shape, mask_shape):
    return (random.randint(0, img_shape[0] - mask_shape[0]),
            random.randint(0, img_shape[1] - mask_shape[1]),)


def replace_masked_pixels(main, donor, mask, main_transl, donor_transl):
    n, m = mask.shape
    for i in range(n):
        for j in range(m):
            if mask[i, j]:
                main[j + main_transl[1], i + main_transl[0]] = \
                    donor[i + donor_transl[0], j + donor_transl[1]]
    return main


def extend_outline_mask(outline_mask, outline, main_shape, main_transl, dist):
    n = main_shape[0]
    m = main_shape[1]
    for xy in outline:
        for iii in range(-dist, dist + 1):
            for jjj in range(-dist, dist + 1):
                ii = xy[0] + iii + main_transl[0]
                jj = xy[1] + jjj + main_transl[1]
                if not (ii < 0 or jj < 0 or ii >= n or jj >= m):
                    outline_mask[ii, jj] = 1
    return outline_mask


NR_MASKS = 1000


main_img = Image.open('../../resources/WSIs/scan_1.png')
donor_img = Image.open('../../resources/WSIs/scan_2.png')
main_pixels = main_img.load()
donor_pixels = donor_img.load()

outline_mask = np.zeros((main_img.size[0], main_img.size[1]))

for i in range(3):
    mask, outline = get_random_mask(NR_MASKS)

    main_transl = generate_random_translation(main_img.size, mask.shape)
    donor_transl = generate_random_translation(donor_img.size, mask.shape)

    main_pixels = replace_masked_pixels(main_pixels, donor_pixels, mask,
                                        main_transl, donor_transl)

    outline_mask = extend_outline_mask(outline_mask, outline,
                                       main_img.size, main_transl, 6)


main_img.save('./test/test.png', format="png")

with open('./test/outline_mask.pkl', 'wb') as handle:
    pickle.dump(outline_mask, handle, protocol=pickle.HIGHEST_PROTOCOL)
