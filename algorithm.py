import numpy as np
import cv2


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

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
    if mask is None:
      print('Установите маску')
      return pic, pic
    else:
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

      bestx, besty, bestv, hx, hy = results
      rez = np.ones(best_image.shape)
      rez[bestx:bestx + hx, besty:besty + hy] = 0
      return rez, best_image


def Viola_Jones(pic, mask):
    image = np.asarray(pic)
    rez = np.ones(image.shape)

    faces = face_cascade.detectMultiScale(image)
    for (x,y,w,h) in faces:
        rez[y:y+h, x:x+w] = 10

    eyes = eye_cascade.detectMultiScale(image)
    for (ex,ey,ew,eh) in eyes:
        rez[ey:ey+eh, ex:ex+ew] = 20
    return rez, image


def find_face_symmetries(pic, mask):
  def find_ssymetry(im):
    y = 0
    best_v = np.inf
    h = im.shape[1]//3
    for i in range(h, im.shape[1] - h):
      v = np.sum(np.abs(im[:, i-h:i] - im[:, i:i+h][:, ::-1]))
      if v < best_v:
        best_v = v
        y = i
    return y

  image = np.asarray(pic)
  rez = np.ones(image.shape)

  eyes = eye_cascade.detectMultiScale(image)
  for (ex,ey,ew,eh) in eyes:
      y = find_ssymetry(image[ey:ey+eh, ex:ex+ew])
      rez[ey:ey+eh, ex+y:ex+y+1] = 20
      
  faces = face_cascade.detectMultiScale(image)
  for (ex,ey,ew,eh) in faces:
      y = find_ssymetry(image[ey:ey+eh, ex:ex+ew])
      rez[ey:ey+eh, ex+y:ex+y+1] = 20  

  return rez, image


