from PyQt6.QtWidgets import QMainWindow, QStackedWidget

from src.gui.main_menu import MainMenuWidget
from src.gui.single_neuron import SingleNeuronWidget
from src.gui.neuron_explorer import NeuronExplorerWidget
from src.gui.network_lab import NetworkLabWidget


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Oscillation Explorer")
        self.setGeometry(100, 100, 1200, 900)
        self.setStyleSheet("background-color: #f5f6fa;")

        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)

        self.menu_widget = MainMenuWidget(on_navigate=self.navigate_to)
        self.single_neuron_widget = SingleNeuronWidget(
            on_complete=lambda: self.navigate_to('neuron_explorer'),
            on_menu=lambda: self.navigate_to('menu')
        )
        self.neuron_explorer_widget = NeuronExplorerWidget(
            on_complete=lambda: self.navigate_to('network_lab'),
            on_menu=lambda: self.navigate_to('menu')
        )
        self.network_lab_widget = NetworkLabWidget(
            on_menu=lambda: self.navigate_to('menu')
        )

        self.central_widget.addWidget(self.menu_widget)
        self.central_widget.addWidget(self.single_neuron_widget)
        self.central_widget.addWidget(self.neuron_explorer_widget)
        self.central_widget.addWidget(self.network_lab_widget)

        self.pages = {
            'menu': 0,
            'single_neuron': 1,
            'neuron_explorer': 2,
            'network_lab': 3,
        }

    def navigate_to(self, page):
        self.central_widget.setCurrentIndex(self.pages.get(page, 0))
