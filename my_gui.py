from PyQt5.QtWidgets import QWidget, QSizePolicy, QPushButton, QFileDialog
from sklearn.datasets import fetch_olivetti_faces
import numpy as np
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image

from algorithm import find_best_shape

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

    def plot_with_mask(self, image, mask, title=''):
        result, image = find_best_shape(image, mask)
        bestx, besty, bestv, hx, hy = result
        rez = np.ones(image.shape)
        rez[bestx:bestx + hx, besty:besty + hy] = 0
        self.ax.imshow(image, cmap='gray')
        self.ax.imshow(rez, cmap='Blues', alpha=0.5)
        self.ax.set_title(title)
        self.draw()


class ImageMaskWindow(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.resize(1000, 500)
        self.current_pic = None
        self.current_mask = None

        data_images = fetch_olivetti_faces()
        self.images = data_images['images']

        self.image = PlotCanvas(self, width=5, height=4)
        self.image.move(0, 0)

        self.mask = PlotCanvas(self, width=5, height=4)
        self.mask.move(500, 0)

        self.show_image_button = QPushButton('Показать\n текущее изображение', self)
        self.show_image_button.setGeometry(250, 425, 130, 50)
        self.show_image_button.clicked.connect(self.change_image)

        self.apply_mask_button = QPushButton('Применить\n маску', self)
        self.apply_mask_button.setGeometry(450, 425, 100, 50)

        def apply_mask():
            if self.current_pic is not None and self.current_mask is not None:
                pic = Image.fromarray(np.uint8(self.current_pic * 255) , 'L')
                self.image.plot_with_mask(pic, self.current_mask, 'Изображение с маской')

        self.apply_mask_button.clicked.connect(apply_mask)

        self.change_mask_button = QPushButton('Сменить\n маску', self)
        self.change_mask_button.setGeometry(650, 425, 100, 50)

        self.cut_mask_button = QPushButton('Использовать текущее\n изображение как маску', self)
        self.cut_mask_button.setGeometry(770, 425, 150, 50)


    def change_image(self, arg=None, pic=None):
        if pic is not None:
            self.current_pic = pic

        if self.current_pic is not None:
            self.image.plot(self.current_pic, 'Изображение')

    def change_mask(self, mask=None):
        self.current_mask = mask
        if self.current_mask is not None:
            self.mask.plot(self.current_mask, 'Маска')


class DatasetWindow(ImageMaskWindow):

    def __init__(self, parent=None):
        ImageMaskWindow.__init__(self, parent)
        self.setWindowTitle('Window with dastaset images')

        self.change_pic_button = QPushButton('Сменить\n изображение', self)
        self.change_pic_button.setGeometry(100, 425, 100, 50)

        def change_random_image():
            self.change_image(pic=self.images[np.random.randint(len(self.images))])

        self.change_pic_button.clicked.connect(change_random_image)

        def change_random_mask():
            self.change_mask(self.images[np.random.randint(len(self.images))][10:55, 5:55])

        self.change_mask_button.clicked.connect(change_random_mask)



class DownloadWindow(ImageMaskWindow):
    def __init__(self, parent=None):
        ImageMaskWindow.__init__(self, parent)

        self.download_button = QPushButton('Загрузить\n изображение', self)
        self.download_button.setGeometry(100, 425, 100, 50)

        def download_picture():
            filename = QFileDialog.getOpenFileName(self, filter = "Files (*.jpg *.png *.jpeg)")[0]
            print(filename)
            image = Image.open(filename)
            img = np.asarray(image.convert('LA'))[:, :, 0]
            self.change_image(img/np.max(img))

        self.download_button.clicked.connect(download_picture)

        def change_random_mask():
            self.change_mask(self.images[np.random.randint(len(self.images))][10:55, 5:55])

        self.change_mask_button.clicked.connect(change_random_mask)


class Main(QWidget):

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.setWindowTitle('Template Matching')
        self.resize(450, 250)

        self.dataset_window = DatasetWindow()

        self.dataset_window_button = QPushButton('Изображения из датасета', self)
        self.dataset_window_button.setGeometry(120, 30, 220, 50)

        def open_dataset_window():
            if self.dataset_window.isHidden():
                self.dataset_window.show()
            else:
                self.dataset_window.hide()

        self.dataset_window_button.clicked.connect(open_dataset_window)

        self.download_window = DownloadWindow()
        self.download_window_button = QPushButton('Загрузить свои изображения', self)
        self.download_window_button.setGeometry(120, 130, 220, 50)

        def open_download_window():
            if self.download_window.isHidden():
                self.download_window.show()
            else:
                self.download_window.hide()

        self.download_window_button.clicked.connect(open_download_window)

        self.show()
