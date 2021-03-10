import numpy as np


def template_matching(image, mask):
    hx, hy = mask.shape
    xmax, ymax = image.shape

    bestx = 0
    besty = 0
    bestv = np.inf

    if xmax < hx or ymax < hy:
        print('Error: mask is bigger than input image')
    else:
        for x in range(xmax - hx + 1):
            for y in range(ymax - hy + 1):
                rez = np.sum(np.abs(image[x:x + hx, y:y + hy] - mask))
                if rez < bestv:
                    bestv = rez
                    bestx = x
                    besty = y
    return bestx, besty, bestv, hx, hy