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


def find_best_shape(pic, mask):
    results = (0, 0, np.inf, 0, 0)
    best_image = None
    for scale in np.linspace(1.0, max(mask.shape) / min(pic.size), 21)[:-1]:
      row = int(pic.size[0] * scale)
      col = int(pic.size[1] * scale)
      image = np.asarray(pic.resize((row, col)))
      image = image/np.max(image)
      bestx, besty, bestv, hx, hy = template_matching(image, mask)
      print((bestx, besty, bestv, hx, hy))
      if bestv < results[2]:
        results = (bestx, besty, bestv, hx, hy)
        best_image = image
    return results, best_image