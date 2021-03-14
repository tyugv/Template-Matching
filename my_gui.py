from PyQt5.QtWidgets import QWidget, QPushButton, QFileDialog
import numpy as np
from PIL import Image
from gui_template import ImageMaskWindow


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

            if np.max(image.size) > 500:
                scale = 500 / np.max(image.size)
                row = int(image.size[0] * scale)
                col = int(image.size[1] * scale)
                image = image.resize((row, col))

            img = np.asarray(image.convert('LA'))[:, :, 0]
            self.change_image(pic = img/np.max(img))

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
