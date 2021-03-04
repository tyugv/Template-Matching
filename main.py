import sys
import numpy as np
import random
from PyQt5.QtWidgets import QApplication, QWidget, QSizePolicy

import matplotlib
matplotlib.use('Qt5Agg')

#from PyQt5 import QtCore, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

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
        self.plot()


    def plot(self):
        data = [random.random() for i in range(25)]
        ax = self.figure.add_subplot(111)
        ax.plot(data, 'r-')
        ax.set_title('PyQt Matplotlib Example')
        self.draw()

class Main(QWidget):

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.setWindowTitle('Template Matching')
        self.resize(500, 500)
        self.move(300, 300)

        m = PlotCanvas(self, width=5, height=4)
        m.move(0,0)

        self.show()


if __name__ == '__main__':

    app = QApplication(sys.argv)

    w = QWidget()
    main = Main()
    sys.exit(app.exec_())