#!/usr/bin/env python3
import sys

import matplotlib
matplotlib.use('QtAgg')

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor
from src.gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(245, 246, 250))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(45, 52, 54))
    palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Text, QColor(45, 52, 54))
    palette.setColor(QPalette.ColorRole.Button, QColor(223, 228, 234))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(45, 52, 54))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(52, 152, 219))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
