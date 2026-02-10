import numpy as np
from collections import deque
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

from src.neurons.single_neuron import SingleNeuronSimulation
from src.gui.canvas import MplCanvas


class SingleNeuronWidget(QWidget):

    def __init__(self, on_complete, on_menu):
        super().__init__()
        self.on_complete = on_complete
        self.on_menu = on_menu
        self.simulation = SingleNeuronSimulation(target_frequency=10.0)
        self.is_running = False
        self.success_count = 0
        self.window_ms = 500
        self.voltage_buffer = deque(maxlen=5000)
        self.time_buffer = deque(maxlen=5000)
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

        title = QLabel("Part 1: Wake Up the Neuron!")
        title.setFont(QFont('Arial', 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #2d3436; margin: 10px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        instructions = QLabel(
            "Control the input current to make the neuron fire at exactly 10 Hz.\n"
            "This is the alpha rhythm - associated with relaxed wakefulness!"
        )
        instructions.setStyleSheet("color: #636e72; font-size: 12px;")
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(instructions)

        self.canvas = MplCanvas(self, width=10, height=5)
        layout.addWidget(self.canvas)
        self.setup_plots()

        control_layout = QHBoxLayout()

        slider_layout = QVBoxLayout()
        self.current_label = QLabel("Input Current: 0.0")
        self.current_label.setStyleSheet("color: #2d3436;")
        slider_layout.addWidget(self.current_label)
        self.current_slider = QSlider(Qt.Orientation.Horizontal)
        self.current_slider.setRange(0, 5000)
        self.current_slider.valueChanged.connect(self.on_current_changed)
        slider_layout.addWidget(self.current_slider)
        control_layout.addLayout(slider_layout)

        btn_layout = QVBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.toggle_simulation)
        self.start_btn.setStyleSheet("background-color: #27ae60; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold;")
        btn_layout.addWidget(self.start_btn)

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_simulation)
        self.reset_btn.setStyleSheet("background-color: #e74c3c; color: white; padding: 10px 20px; border-radius: 5px;")
        btn_layout.addWidget(self.reset_btn)
        control_layout.addLayout(btn_layout)
        layout.addLayout(control_layout)

        self.hint_label = QLabel("Adjust the slider and press Start!")
        self.hint_label.setStyleSheet("color: #e67e22; font-size: 14px; padding: 10px; background-color: #ffeaa7; border-radius: 5px;")
        self.hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.hint_label)

        self.continue_btn = QPushButton("Continue to Neuron Explorer →")
        self.continue_btn.clicked.connect(self.on_complete)
        self.continue_btn.setStyleSheet("background-color: #3498db; color: white; padding: 15px; border-radius: 5px; font-weight: bold;")
        self.continue_btn.hide()
        layout.addWidget(self.continue_btn)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)

    def setup_plots(self):
        self.canvas.fig.clear()
        self.canvas.fig.set_facecolor('#f5f6fa')
        gs = self.canvas.fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.3)

        self.ax_voltage = self.canvas.fig.add_subplot(gs[0, 0])
        self.ax_voltage.set_facecolor('#ffffff')
        self.ax_voltage.set_ylabel('Voltage', color='#2d3436')
        self.ax_voltage.set_xlabel('Time (ms)', color='#2d3436')
        self.ax_voltage.tick_params(colors='#2d3436')
        self.ax_voltage.set_ylim(-80, 40)
        self.ax_voltage.set_xlim(0, self.window_ms)
        self.voltage_line, = self.ax_voltage.plot([], [], color='#3498db', linewidth=1)
        for spine in self.ax_voltage.spines.values():
            spine.set_color('#b2bec3')

        self.ax_rate = self.canvas.fig.add_subplot(gs[0, 1])
        self.ax_rate.set_facecolor('#ffffff')
        self.ax_rate.set_xlim(0, 1)
        self.ax_rate.set_ylim(0, 30)
        self.ax_rate.axhline(y=10, color='#2ecc71', linestyle='--', linewidth=2)
        self.ax_rate.set_title('Firing Rate', color='#2d3436')
        self.ax_rate.tick_params(colors='#2d3436')
        for spine in self.ax_rate.spines.values():
            spine.set_color('#b2bec3')

        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def on_current_changed(self, value):
        self.current_label.setText(f"Input Current: {value/10:.1f}")
        self.simulation.set_input_current(value / 10.0)

    def toggle_simulation(self):
        if not self.is_running:
            if not self.simulation.is_setup:
                self.simulation.setup()
            self.is_running = True
            self.start_btn.setText("Pause")
            self.timer.start(50)
        else:
            self.is_running = False
            self.start_btn.setText("Start")
            self.timer.stop()

    def reset_simulation(self):
        self.timer.stop()
        self.is_running = False
        self.start_btn.setText("Start")
        self.voltage_buffer.clear()
        self.time_buffer.clear()
        self.simulation.reset()
        self.setup_plots()
        self.success_count = 0
        self.continue_btn.hide()

    def update_simulation(self):
        if not self.is_running:
            return
        data = self.simulation.run_step(duration_ms=50)
        if len(data['time']) == 0:
            return

        new_times = data['time']
        new_voltages = data['voltage']
        if len(self.time_buffer) > 0:
            mask = new_times > self.time_buffer[-1]
            new_times, new_voltages = new_times[mask], new_voltages[mask]
        for t, v in zip(new_times, new_voltages):
            self.time_buffer.append(t)
            self.voltage_buffer.append(v)

        current_time = data['time'][-1]
        window_start = max(0, current_time - self.window_ms)
        times_array = np.array(self.time_buffer)
        volts_array = np.array(self.voltage_buffer)
        display_times = times_array - window_start
        mask = display_times >= 0
        self.voltage_line.set_data(display_times[mask], volts_array[mask])

        self.ax_rate.clear()
        self.ax_rate.set_facecolor('#ffffff')
        self.ax_rate.set_xlim(0, 1)
        self.ax_rate.set_ylim(0, 30)
        self.ax_rate.axhline(y=10, color='#2ecc71', linestyle='--', linewidth=2)
        bar_color = '#2ecc71' if data['on_target'] else '#e74c3c'
        self.ax_rate.bar(0.5, data['firing_rate'], width=0.4, color=bar_color)
        self.ax_rate.set_title(f"{data['firing_rate']:.1f} Hz", color='#2d3436')
        self.ax_rate.tick_params(colors='#2d3436')
        for spine in self.ax_rate.spines.values():
            spine.set_color('#b2bec3')

        self.canvas.refresh()
        self.hint_label.setText(self.simulation.get_hint(data['firing_rate']))

        if data['on_target']:
            self.success_count += 1
            if self.success_count > 20:
                self.hint_label.setText("Excellent! You have mastered the basics. Let's explore different neuron types!")
                self.hint_label.setStyleSheet("color: #27ae60; padding: 10px; background-color: #d5f5e3; border-radius: 5px;")
                self.continue_btn.show()
        else:
            self.success_count = 0
