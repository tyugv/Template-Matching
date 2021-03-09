import sys
import numpy as np
import random
from PyQt5.QtWidgets import QApplication, QWidget, QSizePolicy, QPushButton
from sklearn.datasets import fetch_olivetti_faces

import matplotlib
matplotlib.use('Qt5Agg')

#from PyQt5 import QtCore, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


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
        rez = np.sum(np.abs(image[x:x+hx, y:y+hy] - mask))
        if rez < bestv:
          bestv = rez
          bestx = x
          besty = y
  return bestx, besty, bestv, hx, hy


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self, data, title=''):
        ax = self.figure.add_subplot(111)
        ax.imshow(data, cmap='gray')
        ax.set_title(title)
        self.draw()

    def plot_with_mask(self, image, mask, title=''):
        bestx, besty, bestv, hx, hy = template_matching(image, mask)
        rez = np.ones(image.shape)
        rez[bestx:bestx+hx, besty:besty+hy] = 0
        ax = self.figure.add_subplot(111)
        ax.imshow(image, cmap='gray')
        ax.imshow(rez, cmap='Blues', alpha=0.5)
        ax.set_title(title)
        self.draw()

class DatasetWindow(QWidget):

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.setWindowTitle('Window with dastaset images')
        self.resize(1000, 500)
        self.move(300, 300)

        data_images = fetch_olivetti_faces()
        self.images = data_images['images']
        self.current_pic = None
        self.current_mask = None

        self.image = PlotCanvas(self, width=5, height=4)
        self.image.move(0,0)

        self.change_pic_button = QPushButton('Сменить\n изображение', self)
        self.change_pic_button.setGeometry(200, 425, 100, 50)

        def change_pic():
            self.current_pic = self.images[np.random.randint(len(self.images))]
            self.image.plot(self.current_pic, 'Изображение')

        self.change_pic_button.clicked.connect(change_pic)

        self.mask = PlotCanvas(self, width=5, height=4)
        self.mask.move(500,0)

        self.change_mask_button = QPushButton('Сменить\n маску', self)
        self.change_mask_button.setGeometry(700, 425, 100, 50)

        def change_mask():
            self.current_mask = self.images[np.random.randint(len(self.images))][10:55, 5:55]
            self.mask.plot(self.current_mask, 'Маска')

        self.change_mask_button.clicked.connect(change_mask)

        self.apply_mask_button = QPushButton('Применить\n маску', self)
        self.apply_mask_button.setGeometry(450, 425, 100, 50)

        def apply_mask():
            self.image.plot_with_mask(self.current_pic, self.current_mask, 'Изображение с маской')

        self.apply_mask_button.clicked.connect(apply_mask)

class Main(QWidget):

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.setWindowTitle('Template Matching')
        self.resize(500, 500)
        self.move(300, 300)

        self.dataset_window = DatasetWindow()

        self.dataset_window_button = QPushButton('Изображения из датасета', self)
        self.dataset_window_button.setGeometry(10, 10, 200, 50)
        def open_dataset_window():
            if self.dataset_window.isHidden():
                self.dataset_window.show()
            else:
                self.dataset_window.hide()

        self.dataset_window_button.clicked.connect(open_dataset_window)

        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Main()
    sys.exit(app.exec_())