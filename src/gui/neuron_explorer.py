import numpy as np
from collections import deque
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider,
    QLabel, QGroupBox, QStackedWidget, QComboBox, QSizePolicy, QApplication
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

from src.neurons.single_neuron import NeuronExplorer, NEURON_PRESETS
from src.gui.canvas import MplCanvas


class NeuronExplorerWidget(QWidget):

    def __init__(self, on_complete, on_menu):
        super().__init__()
        self.on_complete = on_complete
        self.on_menu = on_menu

        self.simulations = {
            'E': NeuronExplorer(),
            'PV': NeuronExplorer(),
            'SOM': NeuronExplorer()
        }
        self.simulations['E'].set_preset('regular_spiking')
        self.simulations['PV'].set_preset('fast_spiking')
        self.simulations['SOM'].set_preset('low_threshold')

        self.is_running = False
        self.window_ms = 500
        self.voltage_buffers = {k: deque(maxlen=5000) for k in self.simulations}
        self.time_buffers = {k: deque(maxlen=5000) for k in self.simulations}

        self.challenge_mode = False
        self.current_challenge = None
        self.challenges = {
            'tonic': {
                'name': 'Tonic Spiking',
                'description': 'Create sustained regular firing with no adaptation. Constant ISI.',
                'hint': 'Set a=0 (no subthreshold adaptation), moderate b, fast τw...',
                'solution': {
                    'a': 0.0, 'b': 60.0, 'V_r': -55.0, 'tau_w': 30.0,
                    'current': 120.0, 'C': 40.0, 'g_L': 2.0, 'E_L': -70.0,
                    'V_T': -50.0, 'Delta_T': 2.0
                }
            },
            'adapting': {
                'name': 'Adapting Neuron',
                'description': 'Create an ADAPTING pattern: starts fast, gradually slows down over time',
                'hint': 'Set a=0 (no subthreshold adaptation), small b, and slow τw to accumulate adaptation...',
                'solution': {
                    'a': 0.0, 'b': 5.0, 'V_r': -55.0, 'tau_w': 210.0,
                    'current': 120.0, 'C': 40.0, 'g_L': 2.0, 'E_L': -70.0,
                    'V_T': -50.0, 'Delta_T': 2.0
                }
            },
            'bursting': {
                'name': 'Bursting Neuron',
                'description': 'Create RHYTHMIC BURSTS: groups of spikes separated by silent periods',
                'hint': 'Key insight: negative a makes w regenerative! Also raise V_r close to threshold...',
                'solution': {
                    'a': -0.5, 'b': 7.0, 'V_r': -46.0, 'tau_w': 100.0,
                    'current': 120.0, 'C': 10.0, 'g_L': 2.0, 'E_L': -70.0,
                    'V_T': -50.0, 'Delta_T': 2.0
                }
            },
            'irregular': {
                'name': 'Irregular/Chaotic',
                'description': 'Create IRREGULAR firing with variable inter-spike intervals',
                'hint': 'Negative a combined with high V_r creates chaotic dynamics near bifurcation...',
                'solution': {
                    'a': -0.5, 'b': 7.0, 'V_r': -46.0, 'tau_w': 100.0,
                    'current': 120.0, 'C': 20.0, 'g_L': 2.0, 'E_L': -70.0,
                    'V_T': -50.0, 'Delta_T': 2.0
                }
            },
        }

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(5)

        header = QHBoxLayout()
        back_btn = QPushButton("← Menu")
        back_btn.clicked.connect(self.on_menu)
        back_btn.setStyleSheet("background-color: #dfe4ea; color: #2d3436; padding: 8px 16px; border-radius: 5px;")
        header.addWidget(back_btn)
        header.addStretch()

        self.compare_btn = QPushButton("Compare Mode")
        self.compare_btn.setCheckable(True)
        self.compare_btn.setChecked(True)
        self.compare_btn.clicked.connect(lambda: self.set_mode(False))
        self.compare_btn.setStyleSheet("QPushButton { background-color: #9b59b6; color: white; padding: 8px 16px; border-radius: 5px; } QPushButton:checked { border: 2px solid #2d3436; }")
        header.addWidget(self.compare_btn)

        self.challenge_btn = QPushButton("Challenge Mode")
        self.challenge_btn.setCheckable(True)
        self.challenge_btn.clicked.connect(lambda: self.set_mode(True))
        self.challenge_btn.setStyleSheet("QPushButton { background-color: #e67e22; color: white; padding: 8px 16px; border-radius: 5px; } QPushButton:checked { border: 2px solid #2d3436; }")
        header.addWidget(self.challenge_btn)

        layout.addLayout(header)

        title = QLabel("Part 2: Neuron Explorer")
        title.setFont(QFont('Arial', 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #2d3436;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        self.mode_stack = QStackedWidget()

        compare_widget = QWidget()
        compare_layout = QVBoxLayout(compare_widget)
        compare_layout.setContentsMargins(0, 0, 0, 0)

        compare_desc = QLabel("Compare how E (Excitatory), PV (Fast-Spiking), and SOM (Adapting) neurons respond to the same input!")
        compare_desc.setStyleSheet("color: #636e72; font-style: italic; margin: 5px;")
        compare_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        compare_layout.addWidget(compare_desc)

        self.compare_canvas = MplCanvas(self, width=12, height=4)
        compare_layout.addWidget(self.compare_canvas)

        self.mode_stack.addWidget(compare_widget)

        challenge_widget = QWidget()
        challenge_layout = QVBoxLayout(challenge_widget)
        challenge_layout.setContentsMargins(0, 0, 0, 0)

        challenge_selector = QHBoxLayout()
        challenge_label = QLabel("Select Challenge:")
        challenge_label.setStyleSheet("color: #2d3436;")
        challenge_selector.addWidget(challenge_label)

        self.challenge_combo = QComboBox()
        self.challenge_combo.addItem("Make it Burst!", "bursting")
        self.challenge_combo.addItem("Tonic Neuron", "tonic")
        self.challenge_combo.addItem("Adapting Pattern", "adapting")
        self.challenge_combo.currentIndexChanged.connect(self.on_challenge_changed)
        self.challenge_combo.setStyleSheet("background-color: #dfe4ea; color: #2d3436; padding: 8px; border-radius: 5px;")
        challenge_selector.addWidget(self.challenge_combo)
        challenge_selector.addStretch()

        self.solution_btn = QPushButton("Show Solution")
        self.solution_btn.clicked.connect(self.show_solution)
        self.solution_btn.setStyleSheet("background-color: #f39c12; color: white; padding: 8px 16px; border-radius: 5px;")
        challenge_selector.addWidget(self.solution_btn)
        challenge_layout.addLayout(challenge_selector)

        self.challenge_desc = QLabel(self.challenges['bursting']['description'])
        self.challenge_desc.setStyleSheet("color: #e67e22; padding: 10px; background-color: #ffeaa7; border-radius: 5px;")
        self.challenge_desc.setWordWrap(True)
        challenge_layout.addWidget(self.challenge_desc)

        self.challenge_canvas = MplCanvas(self, width=12, height=4)
        challenge_layout.addWidget(self.challenge_canvas)

        self.mode_stack.addWidget(challenge_widget)
        layout.addWidget(self.mode_stack)

        current_box = QGroupBox("Input Current (same for all neurons)")
        current_box.setStyleSheet("QGroupBox { color: #2d3436; border: 1px solid #3498db; border-radius: 5px; padding: 10px; font-weight: bold; }")
        current_layout = QHBoxLayout(current_box)

        self.current_label = QLabel("Current: 0 pA")
        self.current_label.setStyleSheet("color: #2980b9; font-size: 14px; font-weight: bold; min-width: 120px;")
        current_layout.addWidget(self.current_label)

        self.current_slider = QSlider(Qt.Orientation.Horizontal)
        self.current_slider.setRange(0, 5000)
        self.current_slider.setValue(0)
        self.current_slider.valueChanged.connect(self.on_current_changed)
        current_layout.addWidget(self.current_slider)
        layout.addWidget(current_box)

        self.challenge_params_box = QGroupBox("AdEx Parameters (adjust to complete the challenge)")
        self.challenge_params_box.setStyleSheet("QGroupBox { color: #2d3436; border: 1px solid #e67e22; border-radius: 5px; padding: 10px; }")
        params_layout = QHBoxLayout(self.challenge_params_box)

        self.param_sliders = {}
        self.param_labels = {}

        param_configs = [
            ('a', 'a (subthresh)', 0, 80, 20),
            ('b', 'b (spike adapt)', 0, 200, 100),
            ('V_r', 'V_r (reset)', -65, -35, -58),
            ('tau_w', 'τ_w (time)', 1, 40, 12),
            ('C', 'C (capacitance)', 50, 400, 200),
            ('g_L', 'gL (leak)', 5, 30, 10),
            ('E_L', 'E_L (leak rev)', -80, -55, -70),
            ('V_T', 'V_T (threshold)', -55, -40, -50),
            ('Delta_T', 'ΔT (slope)', 5, 50, 20),
        ]

        for name, label, min_v, max_v, default in param_configs:
            col = QVBoxLayout()
            lbl = QLabel(f"{label}")
            lbl.setStyleSheet("color: #2d3436; font-size: 10px;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.param_labels[name] = lbl
            col.addWidget(lbl)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(min_v, max_v)
            slider.setValue(default)
            slider.valueChanged.connect(lambda v, n=name: self.on_challenge_param_changed(n, v))
            self.param_sliders[name] = slider
            col.addWidget(slider)
            params_layout.addLayout(col)

        self.challenge_params_box.hide()
        layout.addWidget(self.challenge_params_box)

        eq_box = QGroupBox("AdEx Neuron Model")
        eq_box.setStyleSheet("QGroupBox { color: #2d3436; border: 1px solid #b2bec3; border-radius: 5px; padding: 5px; }")
        eq_layout = QVBoxLayout(eq_box)

        self.eq_canvas = MplCanvas(self, width=10, height=3.2, dpi=100)
        self.eq_canvas.fig.set_facecolor('#f5f6fa')
        self.render_latex_equations()
        eq_layout.addWidget(self.eq_canvas)

        self.eq_params_label = QLabel()
        self.eq_params_label.setStyleSheet("color: #2980b9; font-family: monospace; font-size: 10px; padding: 5px;")
        self.eq_params_label.setWordWrap(True)
        eq_layout.addWidget(self.eq_params_label)

        layout.addWidget(eq_box)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.toggle_simulation)
        self.start_btn.setStyleSheet("background-color: #27ae60; color: white; padding: 10px 30px; border-radius: 5px; font-weight: bold;")
        btn_layout.addWidget(self.start_btn)

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_simulation)
        self.reset_btn.setStyleSheet("background-color: #e74c3c; color: white; padding: 10px 30px; border-radius: 5px;")
        btn_layout.addWidget(self.reset_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.info_label = QLabel("Adjust the input current and watch how each neuron type responds differently!")
        self.info_label.setStyleSheet("color: #e67e22; padding: 10px; background-color: #ffeaa7; border-radius: 5px;")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        self.continue_btn = QPushButton("Continue to Challenge Mode →")
        self.continue_btn.clicked.connect(self.go_to_challenge_mode)
        self.continue_btn.setStyleSheet("background-color: #9b59b6; color: white; padding: 15px 30px; border-radius: 5px; font-weight: bold; font-size: 14px;")
        layout.addWidget(self.continue_btn)

        self.continue_to_network_btn = QPushButton("Continue to Network Lab →")
        self.continue_to_network_btn.clicked.connect(self.on_complete)
        self.continue_to_network_btn.setStyleSheet("background-color: #3498db; color: white; padding: 15px 30px; border-radius: 5px; font-weight: bold; font-size: 14px;")
        self.continue_to_network_btn.hide()
        layout.addWidget(self.continue_to_network_btn)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)

        self.challenge_sim = NeuronExplorer()
        self.challenge_voltage_buffer = deque(maxlen=5000)
        self.challenge_time_buffer = deque(maxlen=5000)

        self.setup_compare_plots()
        self.setup_challenge_plots()

    def set_mode(self, challenge_mode):
        self.challenge_mode = challenge_mode
        self.compare_btn.setChecked(not challenge_mode)
        self.challenge_btn.setChecked(challenge_mode)

        if challenge_mode:
            self.mode_stack.setCurrentIndex(1)
            self.challenge_params_box.show()
            self.info_label.setText("Adjust the parameters and click Run to see the firing pattern!")
            self.continue_btn.hide()
            self.continue_to_network_btn.show()
            self.start_btn.setText("Run")
            self.timer.stop()
            self.is_running = False
        else:
            self.mode_stack.setCurrentIndex(0)
            self.challenge_params_box.hide()
            self.info_label.setText("Adjust the input current and watch how each neuron type responds differently!")
            self.continue_btn.show()
            self.continue_to_network_btn.hide()
            self.start_btn.setText("Start")

        self.reset_simulation()

    def go_to_challenge_mode(self):
        self.set_mode(True)

    def on_challenge_changed(self, index):
        challenge_key = self.challenge_combo.currentData()
        self.current_challenge = challenge_key
        challenge = self.challenges[challenge_key]
        self.challenge_desc.setText(f"{challenge['description']}\n\nHint: {challenge['hint']}")
        self.reset_simulation()

    def show_solution(self):
        if self.current_challenge is None:
            self.current_challenge = 'bursting'

        solution = self.challenges[self.current_challenge]['solution']

        self.param_sliders['a'].setValue(int(solution['a'] * 10))
        self.param_sliders['b'].setValue(int(solution['b']))
        self.param_sliders['V_r'].setValue(int(solution['V_r']))
        self.param_sliders['C'].setValue(int(solution['C']))
        self.param_sliders['g_L'].setValue(int(solution['g_L']))
        self.param_sliders['E_L'].setValue(int(solution['E_L']))
        self.param_sliders['V_T'].setValue(int(solution['V_T']))
        self.param_sliders['Delta_T'].setValue(int(solution['Delta_T'] * 10))
        self.param_sliders['tau_w'].setValue(int(solution['tau_w'] / 10))
        self.current_slider.setValue(int(solution['current'] * 10))

        self.info_label.setText(f"Solution applied! Watch the {self.challenges[self.current_challenge]['name']} pattern.")
        self.info_label.setStyleSheet("color: #27ae60; padding: 10px; background-color: #d5f5e3; border-radius: 5px;")

    def on_challenge_param_changed(self, name, value):
        if name == 'a':
            actual_value = value / 10.0
            self.challenge_sim.set_parameter('a', actual_value)
            self.param_labels[name].setText(f"a: {actual_value:.1f} nS")
        elif name == 'b':
            actual_value = value
            self.challenge_sim.set_parameter('b', actual_value)
            self.param_labels[name].setText(f"b: {actual_value:.0f} pA")
        elif name == 'V_r':
            self.challenge_sim.set_parameter('V_r', value)
            self.param_labels[name].setText(f"Vr: {value} mV")
        elif name == 'tau_w':
            actual_value = value * 10.0
            self.challenge_sim.set_parameter('tau_w', actual_value)
            self.param_labels[name].setText(f"τw: {actual_value:.0f} ms")
        elif name == 'C':
            self.challenge_sim.set_parameter('C', value)
            self.param_labels[name].setText(f"C: {value} pF")
        elif name == 'g_L':
            self.challenge_sim.set_parameter('g_L', value)
            self.param_labels[name].setText(f"gL: {value} nS")
        elif name == 'E_L':
            self.challenge_sim.set_parameter('E_L', value)
            self.param_labels[name].setText(f"EL: {value} mV")
        elif name == 'V_T':
            self.challenge_sim.set_parameter('V_T', value)
            self.param_labels[name].setText(f"VT: {value} mV")
        elif name == 'Delta_T':
            actual_value = value / 10.0
            self.challenge_sim.set_parameter('Delta_T', actual_value)
            self.param_labels[name].setText(f"ΔT: {actual_value:.1f} mV")

    def render_latex_equations(self):
        self.eq_canvas.fig.clear()
        self.eq_canvas.fig.set_facecolor('#f5f6fa')

        gs = self.eq_canvas.fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.05)

        ax_eq = self.eq_canvas.fig.add_subplot(gs[0, 0])
        ax_eq.set_facecolor('#f5f6fa')
        ax_eq.axis('off')

        equations = (
            r"$C\frac{dV}{dt} = -g_L(V - E_L) + g_L \Delta_T e^{(V - V_T)/\Delta_T} - w + I$"
            "\n\n"
            r"$\tau_w \frac{dw}{dt} = a(V - E_L) - w$"
            "\n\n"
            r"$\mathrm{Spike:}\ V \to V_r,\ w \to w + b$"
        )

        ax_eq.text(
            0.02, 0.98, equations,
            transform=ax_eq.transAxes,
            fontsize=11,
            color='#2d3436',
            ha='left',
            va='top',
            family='serif'
        )

        ax_params = self.eq_canvas.fig.add_subplot(gs[0, 1])
        ax_params.set_facecolor('#f5f6fa')
        ax_params.axis('off')

        colors = {'E': '#3498db', 'PV': '#e74c3c', 'SOM': '#2ecc71'}
        y_positions = [0.85, 0.55, 0.25]

        for idx, (key, sim) in enumerate(self.simulations.items()):
            color = colors[key]
            y = y_positions[idx]
            ax_params.text(0.05, y, f"{key}:", transform=ax_params.transAxes,
                          fontsize=10, color=color, fontweight='bold', va='center')
            param_text = f"a={sim.a:.1f}nS  b={sim.b:.0f}pA  τw={sim.tau_w:.0f}ms"
            ax_params.text(0.25, y, param_text, transform=ax_params.transAxes,
                          fontsize=9, color=color, va='center', family='monospace')

        self.eq_canvas.fig.tight_layout()
        self.eq_canvas.draw()

    def setup_compare_plots(self):
        self.compare_canvas.fig.clear()
        self.compare_canvas.fig.set_facecolor('#f5f6fa')
        gs = self.compare_canvas.fig.add_gridspec(1, 3, wspace=0.25)

        self.compare_axes = {}
        self.voltage_lines = {}

        neuron_info = {
            'E': ('Excitatory (E)', '#3498db', 'Regular spiking\nAdaptation'),
            'PV': ('Fast-Spiking (PV)', '#e74c3c', 'No adaptation\nHigh frequency'),
            'SOM': ('Adapting (SOM)', '#2ecc71', 'Strong adaptation\nLow threshold')
        }

        for idx, (key, (title, color, desc)) in enumerate(neuron_info.items()):
            ax = self.compare_canvas.fig.add_subplot(gs[0, idx])
            ax.set_facecolor('#ffffff')
            ax.set_title(title, color=color, fontweight='bold', fontsize=10)
            ax.set_ylabel('Voltage (mV)' if idx == 0 else '', color='#2d3436', fontsize=9)
            ax.set_xlabel('Time (ms)', color='#2d3436', fontsize=9)
            ax.tick_params(colors='#2d3436', labelsize=8)
            ax.set_ylim(-80, 40)
            ax.set_xlim(0, self.window_ms)

            line, = ax.plot([], [], color=color, linewidth=1)
            self.voltage_lines[key] = line
            self.compare_axes[key] = ax

            ax.text(0.5, 0.02, desc, transform=ax.transAxes, fontsize=7,
                   color='#636e72', ha='center', va='bottom')

            for spine in ax.spines.values():
                spine.set_color('#b2bec3')

        self.compare_canvas.fig.tight_layout()
        self.compare_canvas.draw()

    def setup_challenge_plots(self):
        self.challenge_canvas.setFixedHeight(200)
        self.challenge_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed
        )

        self.challenge_canvas.fig.clear()
        self.challenge_canvas.fig.set_facecolor('#f5f6fa')
        gs = self.challenge_canvas.fig.add_gridspec(1, 1, height_ratios=[1], hspace=0.0)

        self.challenge_ax_voltage = self.challenge_canvas.fig.add_subplot(gs[0])
        self.challenge_ax_voltage.set_facecolor('#ffffff')
        self.challenge_ax_voltage.set_ylabel('Voltage (mV)', color='#2d3436')
        self.challenge_ax_voltage.tick_params(colors='#2d3436')
        self.challenge_ax_voltage.set_ylim(-80, 40)
        self.challenge_ax_voltage.set_xlim(0, self.window_ms)
        self.challenge_voltage_line, = self.challenge_ax_voltage.plot([], [], color='#e67e22', linewidth=1.5)
        for spine in self.challenge_ax_voltage.spines.values():
            spine.set_color('#b2bec3')

        self.challenge_canvas.fig.tight_layout()
        self.challenge_canvas.draw()

    def on_current_changed(self, value):
        current = value / 10.0
        self.current_label.setText(f"Current: {current:.1f}")

        for sim in self.simulations.values():
            sim.set_input_current(current)

        self.challenge_sim.set_input_current(current)

    def toggle_simulation(self):
        if self.challenge_mode:
            self.run_challenge_simulation()
        else:
            if not self.is_running:
                for sim in self.simulations.values():
                    if not sim.is_setup:
                        sim.setup()

                self.is_running = True
                self.start_btn.setText("Pause")
                self.timer.start(50)
            else:
                self.is_running = False
                self.start_btn.setText("Start")
                self.timer.stop()

    def run_challenge_simulation(self):
        self.challenge_sim.reset()
        self.challenge_sim.setup()

        self.start_btn.setEnabled(False)
        self.start_btn.setText("Running...")
        QApplication.processEvents()

        data = self.challenge_sim.run_step(duration_ms=300)

        self.start_btn.setEnabled(True)
        self.start_btn.setText("Run")

        if len(data['time']) == 0:
            return

        times = data['time']
        voltages = data['voltage']

        self.challenge_ax_voltage.clear()
        self.challenge_ax_voltage.set_facecolor('#ffffff')
        self.challenge_ax_voltage.set_ylabel('Voltage (mV)', color='#2d3436')
        self.challenge_ax_voltage.tick_params(colors='#2d3436')
        self.challenge_ax_voltage.set_ylim(-80, 40)
        self.challenge_ax_voltage.set_xlim(0, 300)
        self.challenge_ax_voltage.plot(times, voltages, color='#e67e22', linewidth=1.5)
        for spine in self.challenge_ax_voltage.spines.values():
            spine.set_color('#b2bec3')

        self.challenge_canvas.fig.tight_layout()
        self.challenge_canvas.draw()

        spike_times = data['spike_times']
        n_spikes = len(spike_times)
        if n_spikes > 1:
            isis = np.diff(spike_times)
            mean_isi = np.mean(isis)
            cv_isi = np.std(isis) / mean_isi if mean_isi > 0 else 0
            firing_rate = n_spikes / 0.3

            if cv_isi > 0.5 and np.any(isis < 15):
                pattern = "Bursting detected!"
            elif cv_isi < 0.2:
                pattern = "Regular spiking"
            else:
                pattern = "Irregular spiking"

            self.info_label.setText(
                f"{pattern} | {n_spikes} spikes | {firing_rate:.1f} Hz | CV(ISI)={cv_isi:.2f}"
            )
        elif n_spikes == 1:
            self.info_label.setText("Only 1 spike - increase current or adjust parameters")
        else:
            self.info_label.setText("No spikes - increase current to make the neuron fire")

    def reset_simulation(self):
        self.timer.stop()
        self.is_running = False
        self.start_btn.setText("Start")

        for key in self.voltage_buffers:
            self.voltage_buffers[key].clear()
            self.time_buffers[key].clear()
        self.challenge_voltage_buffer.clear()
        self.challenge_time_buffer.clear()

        for sim in self.simulations.values():
            sim.reset()
        self.challenge_sim.reset()

        self.setup_compare_plots()
        self.setup_challenge_plots()

        self.info_label.setStyleSheet("color: #e67e22; padding: 10px; background-color: #ffeaa7; border-radius: 5px;")

    def update_simulation(self):
        if not self.is_running:
            return

        if self.challenge_mode:
            self.update_challenge_simulation()
        else:
            self.update_compare_simulation()

    def update_compare_simulation(self):
        firing_rates = {}

        for key, sim in self.simulations.items():
            data = sim.run_step(duration_ms=50)
            if len(data['time']) == 0:
                continue

            new_times = data['time']
            new_voltages = data['voltage']

            if len(self.time_buffers[key]) > 0:
                mask = new_times > self.time_buffers[key][-1]
                new_times, new_voltages = new_times[mask], new_voltages[mask]

            for t, v in zip(new_times, new_voltages):
                self.time_buffers[key].append(t)
                self.voltage_buffers[key].append(v)

            current_time = data['time'][-1]
            window_start = max(0, current_time - self.window_ms)
            times_array = np.array(self.time_buffers[key])
            volts_array = np.array(self.voltage_buffers[key])
            display_times = times_array - window_start
            mask = display_times >= 0

            self.voltage_lines[key].set_data(display_times[mask], volts_array[mask])
            firing_rates[key] = data['firing_rate']

        self.compare_canvas.refresh()

        if firing_rates:
            self.info_label.setText(
                f"Firing rates: E={firing_rates.get('E', 0):.1f} Hz | "
                f"PV={firing_rates.get('PV', 0):.1f} Hz | "
                f"SOM={firing_rates.get('SOM', 0):.1f} Hz"
            )

    def update_challenge_simulation(self):
        data = self.challenge_sim.run_step(duration_ms=50)
        if len(data['time']) == 0:
            return

        new_times = data['time']
        new_voltages = data['voltage']

        if len(self.challenge_time_buffer) > 0:
            mask = new_times > self.challenge_time_buffer[-1]
            new_times, new_voltages = new_times[mask], new_voltages[mask]

        for t, v in zip(new_times, new_voltages):
            self.challenge_time_buffer.append(t)
            self.challenge_voltage_buffer.append(v)

        current_time = data['time'][-1]
        window_start = max(0, current_time - self.window_ms)
        times_array = np.array(self.challenge_time_buffer)
        volts_array = np.array(self.challenge_voltage_buffer)
        display_times = times_array - window_start
        mask = display_times >= 0

        self.challenge_voltage_line.set_data(display_times[mask], volts_array[mask])

        self.challenge_canvas.refresh()
        self.info_label.setText(f"Firing rate: {data['firing_rate']:.1f} Hz | Keep adjusting to achieve the target pattern!")
