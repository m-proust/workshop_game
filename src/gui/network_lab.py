import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider,
    QLabel, QGroupBox, QComboBox, QApplication
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from scipy import signal

from src.networks.scenarios import E_PV_Network, E_SOM_Network, E_PV_SOM_Network
from src.gui.canvas import MplCanvas


class NetworkLabWidget(QWidget):

    def __init__(self, on_menu):
        super().__init__()
        self.on_menu = on_menu
        self.networks = {
            'gamma': E_PV_Network(),
            'theta': E_SOM_Network(),
            'coupled': E_PV_SOM_Network(),
        }
        self.current_network = None
        self.current_network_name = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        header = QHBoxLayout()
        back_btn = QPushButton("← Menu")
        back_btn.clicked.connect(self.on_menu)
        back_btn.setStyleSheet("background-color: #dfe4ea; color: #2d3436; padding: 8px 16px; border-radius: 5px;")
        header.addWidget(back_btn)
        header.addStretch()
        layout.addLayout(header)

        title = QLabel("Part 3: Network Oscillation Lab")
        title.setFont(QFont('Arial', 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #2d3436;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        scenario_layout = QHBoxLayout()
        scenarios = [
            ('gamma', '(E-PV)', '#e74c3c'),
            ('theta', '(E-SOM)', '#2ecc71'),
            ('coupled', '(E-PV-SOM)', '#9b59b6'),
        ]
        self.scenario_btns = {}
        for name, label, color in scenarios:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setStyleSheet(f"QPushButton {{ background-color: {color}; color: white; padding: 12px; border-radius: 5px; font-weight: bold; }} QPushButton:checked {{ border: 3px solid white; }}")
            btn.clicked.connect(lambda checked, n=name: self.select_scenario(n))
            scenario_layout.addWidget(btn)
            self.scenario_btns[name] = btn
        layout.addLayout(scenario_layout)

        self.canvas = MplCanvas(self, width=12, height=6)
        layout.addWidget(self.canvas)

        self.params_box = QGroupBox("Network Parameters")
        self.params_box.setStyleSheet("QGroupBox { color: #2d3436; border: 1px solid #b2bec3; border-radius: 5px; padding: 10px; }")
        self.params_layout = QHBoxLayout(self.params_box)

        self.sliders = {}
        self.slider_labels = {}
        self.slider_containers = {}

        self.scenario_sliders = {
            'gamma': [
                ('drive', 'Drive', 0, 5000, 3300),
                ('e_e', 'E→E', 0, 1000, 130),
                ('e_pv', 'E→PV', 0, 1000, 460),
                ('pv_e', 'PV→E', 0, 1000, 370),
                ('pv_pv', 'PV→PV', 0, 1000, 200),
            ],
            'theta': [
                ('drive', 'Drive', 0, 5000, 2800),
                ('e_e', 'E→E', 0, 1000, 80),
                ('e_som', 'E→SOM', 0, 1000, 250),
                ('som_e', 'SOM→E', 0, 1000, 400),
            ],
            'coupled': [
                ('drive', 'Drive', 0, 5000, 2200),
                ('e_e', 'E→E', 0, 1000, 80),
                ('e_pv', 'E→PV', 0, 1000, 350),
                ('e_som', 'E→SOM', 0, 1000, 680),
                ('pv_e', 'PV→E', 0, 1000, 300),
                ('pv_pv', 'PV→PV', 0, 1000, 200),
                ('som_e', 'SOM→E', 0, 1000, 350),
                ('som_pv', 'SOM→PV', 0, 1000, 200),
            ],
        }

        self.sim_col = QVBoxLayout()
        self.duration_combo = QComboBox()
        self.duration_combo.addItems(["1 sec", "2 sec", "5 sec"])
        self.duration_combo.setCurrentIndex(1)
        self.duration_combo.setStyleSheet("background-color: #dfe4ea; color: #2d3436; padding: 5px;")
        self.sim_col.addWidget(self.duration_combo)

        self.simulate_btn = QPushButton("▶ Simulate")
        self.simulate_btn.clicked.connect(self.run_simulation)
        self.simulate_btn.setStyleSheet("background-color: #27ae60; color: white; padding: 10px; border-radius: 5px; font-weight: bold;")
        self.sim_col.addWidget(self.simulate_btn)

        layout.addWidget(self.params_box)

        self.description_label = QLabel("Select a scenario to begin!")
        self.description_label.setStyleSheet("color: #e67e22; padding: 10px; background-color: #ffeaa7; border-radius: 5px;")
        self.description_label.setWordWrap(True)
        layout.addWidget(self.description_label)

        self.setup_plots()

    def setup_plots(self):
        self.canvas.fig.clear()
        self.canvas.fig.set_facecolor('#f5f6fa')
        gs = self.canvas.fig.add_gridspec(3, 1, height_ratios=[2, 1.5, 1.5], hspace=0.35)

        self.ax_raster = self.canvas.fig.add_subplot(gs[0])
        self.ax_raster.set_facecolor('#ffffff')
        self.ax_raster.set_ylabel('Neuron', color='#2d3436')
        self.ax_raster.set_title('Spike Raster', color='#2d3436', fontweight='bold')
        self.ax_raster.tick_params(colors='#2d3436')

        self.ax_rate = self.canvas.fig.add_subplot(gs[1])
        self.ax_rate.set_facecolor('#ffffff')
        self.ax_rate.set_ylabel('Rate (Hz)', color='#2d3436')
        self.ax_rate.set_title('Population Activity', color='#2d3436', fontweight='bold')
        self.ax_rate.tick_params(colors='#2d3436')

        self.ax_spectrum = self.canvas.fig.add_subplot(gs[2])
        self.ax_spectrum.set_facecolor('#ffffff')
        self.ax_spectrum.set_xlabel('Frequency (Hz)', color='#2d3436')
        self.ax_spectrum.set_ylabel('Power', color='#2d3436')
        self.ax_spectrum.set_title('Power Spectrum', color='#2d3436', fontweight='bold')
        self.ax_spectrum.tick_params(colors='#2d3436')
        self.ax_spectrum.set_xlim(0, 100)

        for low, high, name, color in [(4, 8, 'θ', '#9b59b6'), (30, 80, 'γ', '#e74c3c')]:
            self.ax_spectrum.axvspan(low, high, alpha=0.15, color=color)

        for ax in [self.ax_raster, self.ax_rate, self.ax_spectrum]:
            for spine in ax.spines.values():
                spine.set_color('#b2bec3')

        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def select_scenario(self, name):
        for btn_name, btn in self.scenario_btns.items():
            btn.setChecked(btn_name == name)

        self.current_network_name = name
        self.current_network = self.networks[name]

        self._rebuild_sliders(name)

        self.setup_plots()

    def _rebuild_sliders(self, scenario_name):
        self.params_layout.removeItem(self.sim_col)

        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.layout():
                while item.layout().count():
                    child = item.layout().takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
            elif item.widget():
                item.widget().deleteLater()

        self.sliders.clear()
        self.slider_labels.clear()

        slider_configs = self.scenario_sliders.get(scenario_name, [])

        for param_name, label, min_v, max_v, default in slider_configs:
            col = QVBoxLayout()
            lbl = QLabel(f"{label}: {default/10:.1f}")
            lbl.setStyleSheet("color: #2d3436; font-size: 10px;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.slider_labels[param_name] = lbl
            col.addWidget(lbl)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(min_v, max_v)
            slider.setValue(default)
            slider.valueChanged.connect(lambda v, n=param_name, l=label: self.on_param_changed(n, v, l))
            self.sliders[param_name] = slider
            col.addWidget(slider)
            self.params_layout.addLayout(col)

        self.params_layout.addLayout(self.sim_col)

    def on_param_changed(self, name, value, label=None):
        val = value / 10.0
        if label:
            self.slider_labels[name].setText(f"{label}: {val:.1f}")
        else:
            self.slider_labels[name].setText(f"{name.replace('_', '→').upper()}: {val:.1f}")

    def run_simulation(self):
        if self.current_network is None:
            self.description_label.setText("Select a scenario first!")
            return

        duration_ms = int(self.duration_combo.currentText().split()[0]) * 1000

        params = {}
        for name, slider in self.sliders.items():
            params[name] = slider.value() / 10.0

        setup_params = {}
        drive = params.get('drive', 150)

        if self.current_network_name == 'gamma':
            setup_params['input_drive_E'] = drive
            setup_params['input_drive_PV'] = 50.0
            setup_params['J_EE'] = params.get('e_e', 8)
            setup_params['J_EPV'] = params.get('e_pv', 35)
            setup_params['J_PVE'] = params.get('pv_e', 30)
            setup_params['J_PVPV'] = params.get('pv_pv', 20)

        elif self.current_network_name == 'theta':
            setup_params['input_drive_E'] = drive
            setup_params['input_drive_SOM'] = 0.0
            setup_params['J_EE'] = params.get('e_e', 8)
            setup_params['J_ESOM'] = params.get('e_som', 25)
            setup_params['J_SOME'] = params.get('som_e', 40)

        elif self.current_network_name == 'coupled':
            setup_params['input_drive_E'] = drive
            setup_params['input_drive_PV'] = 50.0
            setup_params['input_drive_SOM'] = 0.0
            setup_params['J_EE'] = params.get('e_e', 8)
            setup_params['J_EPV'] = params.get('e_pv', 35)
            setup_params['J_ESOM'] = params.get('e_som', 25)
            setup_params['J_PVE'] = params.get('pv_e', 30)
            setup_params['J_PVPV'] = params.get('pv_pv', 20)
            setup_params['J_SOME'] = params.get('som_e', 35)
            setup_params['J_SOMPV'] = params.get('som_pv', 20)

        self.current_network.setup(**setup_params)

        self.simulate_btn.setEnabled(False)
        self.simulate_btn.setText("Running...")
        QApplication.processEvents()

        data = self.current_network.run_step(duration_ms=duration_ms)

        self.simulate_btn.setEnabled(True)
        self.simulate_btn.setText("▶ Simulate")

        self.display_results(data, duration_ms)

    def display_results(self, data, duration_ms):
        self.setup_plots()
        colors = {'E': '#3498db', 'PV': '#e74c3c', 'SOM': '#2ecc71'}

        offset = 0
        for group_name, spike_data in data['spikes'].items():
            times, indices = spike_data['times'], spike_data['indices']
            if len(times) > 0:
                self.ax_raster.scatter(times, indices + offset, s=0.5, c=colors.get(group_name, 'white'), label=group_name)
            n = getattr(self.current_network, f'n_{group_name.lower()}', getattr(self.current_network, 'n_exc', 80) if group_name == 'E' else 20)
            offset += n

        self.ax_raster.set_xlim(0, duration_ms)
        self.ax_raster.set_ylim(-1, offset + 1)
        self.ax_raster.legend(loc='upper right', facecolor='#ffffff', labelcolor='#2d3436', fontsize=8)

        if 'E' in data['rates'] and len(data['rates']['E']['rates']) > 10:
            times, rates = data['rates']['E']['times'], data['rates']['E']['rates']
            kernel = np.ones(min(50, len(rates)//10)) / min(50, len(rates)//10)
            rates_smooth = np.convolve(rates, kernel, mode='same') if len(kernel) > 1 else rates
            self.ax_rate.plot(times, rates_smooth, color='#e67e22', linewidth=1)
            self.ax_rate.set_xlim(0, duration_ms)
            self.ax_rate.set_ylim(0, max(rates_smooth) * 1.2 + 10)

        if 'E' in data['rates'] and len(data['rates']['E']['rates']) > 100:
            rates, times = data['rates']['E']['rates'], data['rates']['E']['times']
            dt = np.mean(np.diff(times)) / 1000
            if dt > 0:
                fs = 1 / dt
                nperseg = 100 * min(1024, len(rates) // 2)
                if nperseg > 10:
                    freqs, psd = signal.welch(rates, fs, nperseg=nperseg)

                    mask = freqs <= 100
                    self.ax_spectrum.fill_between(freqs[mask], psd[mask], color='#f39c12', alpha=0.7)
                    if len(psd[mask]) > 0:
                        self.ax_spectrum.set_ylim(0, np.max(psd[mask]) * 1.2 + 0.01)
                        peak_idx = np.argmax(psd[mask])
                        self.ax_spectrum.annotate(f'Peak: {freqs[mask][peak_idx]:.1f} Hz',
                                                   xy=(freqs[mask][peak_idx], psd[mask][peak_idx]),
                                                   xytext=(freqs[mask][peak_idx] + 15, psd[mask][peak_idx] * 0.8),
                                                   color='#2d3436', fontsize=9,
                                                   arrowprops=dict(arrowstyle='->', color='#2d3436'))

        self.canvas.refresh()
