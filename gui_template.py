from PyQt5.QtWidgets import QWidget, QSizePolicy, QPushButton, QLabel, QSpinBox, QListWidget
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.datasets import fetch_olivetti_faces
import numpy as np
from PIL import Image

from algorithm import find_best_shape, Viola_Jones, find_face_symmetries

matplotlib.use('Qt5Agg')


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        self.ax = self.figure.add_subplot(111)

    def plot(self, data, title=''):
        self.ax.imshow(data, cmap='gray')
        self.ax.set_title(title)
        self.draw()

    def plot_with_mask(self, image, rez, title=''):
        self.ax.imshow(image, cmap='gray')
        self.ax.imshow(rez, cmap='Blues', alpha=0.5)
        self.ax.set_title(title)
        self.draw()


class ImageMaskWindow(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.resize(1200, 500)
        self.current_pic = None
        self.current_mask = None
        self.current_algorithm = find_best_shape

        data_images = fetch_olivetti_faces()
        self.images = data_images['images']

        self.image = PlotCanvas(self, width=5, height=4)
        self.image.move(0, 0)

        self.mask = PlotCanvas(self, width=5, height=4)
        self.mask.move(500, 0)

        self.show_image_button = QPushButton('Показать\n текущее изображение', self)
        self.show_image_button.setGeometry(250, 425, 130, 50)
        self.show_image_button.clicked.connect(self.change_image)

        self.apply_mask_button = QPushButton('Применить\n алгоритм', self)
        self.apply_mask_button.setGeometry(450, 425, 100, 50)

        def apply_mask():
            if self.current_pic is not None:
                image = Image.fromarray(np.uint8(self.current_pic * 255) , 'L')
                rez, image = self.current_algorithm(image, self.current_mask)
                self.image.plot_with_mask(image, rez, 'Изображение с маской')

        self.apply_mask_button.clicked.connect(apply_mask)

        self.change_mask_button = QPushButton('Сменить\n маску', self)
        self.change_mask_button.setGeometry(720, 425, 100, 50)

        self.label_use_as_mask = QLabel(self)
        self.label_use_as_mask.setText("Укажите координаты \nизображения для маски")
        self.label_use_as_mask.setGeometry(1020, 50, 150, 25)

        self.label_x = QLabel(self)
        self.label_x.setText('X:')
        self.label_x.setStyleSheet("font: 13pt;")
        self.label_x.setGeometry(1020, 100, 150, 25)

        self.label_y = QLabel(self)
        self.label_y.setText('Y:')
        self.label_y.setStyleSheet("font: 13pt;")
        self.label_y.setGeometry(1020, 130, 150, 25)

        self.xlabels = []
        self.ylabels = []
        for i in range(2):
            self.xlabels.append(QSpinBox(self))
            self.xlabels[i].move(1050 + 60*i, 100)
            self.xlabels[i].setRange(0, 500)

            self.ylabels.append(QSpinBox(self))
            self.ylabels[i].move(1050 + 60*i, 130)
            self.ylabels[i].setRange(0, 500)

        self.cut_mask_button = QPushButton('Использовать текущее\n изображение как маску', self)
        self.cut_mask_button.setGeometry(1020, 180, 150, 50)

        self.change_algorithm = QListWidget(self)
        self.change_algorithm.addItems(['Template Matching', 'Viola-Jones', 'Symmetry'])
        self.change_algorithm.setGeometry(1020, 250, 120, 60)
        self.change_algorithm.setCurrentRow(0)

        def slot(item):
            algorithms = {
            'Template Matching': find_best_shape,
            'Viola-Jones': Viola_Jones,
            'Symmetry': find_face_symmetries}

            self.current_algorithm = algorithms[item.text()]

        self.change_algorithm.itemClicked.connect(slot)


        def cut_mask_from_image():
            if self.current_pic is not None:
                image = self.current_pic[self.xlabels[0].value() : self.xlabels[1].value(),\
                 self.ylabels[0].value() : self.ylabels[1].value()]
                self.change_mask(image)

        self.cut_mask_button.clicked.connect(cut_mask_from_image)

    def change_image(self, arg=None, pic=None):
        if pic is not None:
            self.current_pic = pic

        if self.current_pic is not None:
            self.image.plot(self.current_pic, 'Изображение')

    def change_mask(self, mask=None):
        if mask is not None:
            self.current_mask = mask

        if self.current_mask is not None:
            self.mask.plot(self.current_mask, 'Маска')