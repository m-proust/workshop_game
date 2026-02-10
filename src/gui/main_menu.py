from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class MainMenuWidget(QWidget):

    def __init__(self, on_navigate):
        super().__init__()
        self.on_navigate = on_navigate
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)

        title = QLabel("Neural Oscillation Explorer")
        title.setFont(QFont('Arial', 28, QFont.Weight.Bold))
        title.setStyleSheet("color: #2d3436; margin: 30px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Explore how neurons generate brain rhythms")
        subtitle.setFont(QFont('Arial', 14))
        subtitle.setStyleSheet("color: #636e72; margin-bottom: 30px;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(15)

        menu_items = [
            ('single_neuron', '1. Wake Up the Neuron',
             'Learn the basics: make a neuron fire at a target frequency', '#3498db'),
            ('neuron_explorer', '2. Neuron Explorer',
             'Explore different neuron types: PV, SOM, excitatory...', '#9b59b6'),
            ('network_lab', '3. Network Oscillation Lab',
             'See how networks of neurons create brain rhythms', '#27ae60'),
        ]

        for section_id, title, description, color in menu_items:
            btn_widget = QWidget()
            btn_layout = QVBoxLayout(btn_widget)
            btn_layout.setContentsMargins(0, 0, 0, 0)

            btn = QPushButton(title)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    color: white;
                    padding: 20px 40px;
                    border-radius: 10px;
                    font-size: 16px;
                    font-weight: bold;
                    text-align: left;
                }}
                QPushButton:hover {{
                    background-color: {color}dd;
                }}
            """)
            btn.clicked.connect(lambda checked, s=section_id: self.on_navigate(s))
            btn_layout.addWidget(btn)

            desc = QLabel(description)
            desc.setStyleSheet("color: #636e72; font-size: 12px; margin-left: 10px;")
            btn_layout.addWidget(desc)

            buttons_layout.addWidget(btn_widget)

        layout.addLayout(buttons_layout)
        layout.addStretch()
