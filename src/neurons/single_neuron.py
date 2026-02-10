from brian2 import (
    NeuronGroup, SpikeMonitor, StateMonitor, Network,
    defaultclock, ms, second, Hz, mV, pA, nS, pF, nA,
    start_scope, prefs
)
import numpy as np

prefs.codegen.target = 'numpy'


NEURON_PRESETS = {
    'regular_spiking': {
        'name': 'Regular Spiking (Excitatory)',
        'description': 'Pyramidal neuron.',
        'C': 200.0,    
        'g_L': 10.0,   
        'E_L': -70.0,  
        'V_T': -50.0,   
        'Delta_T': 2.0,  
        'V_r': -58.0,   
        'a': 2.0,      
        'b': 100.0,     
        'tau_w': 120.0, 
    },
    'fast_spiking': {
        'name': 'Fast Spiking (PV Interneuron)',
        'description': 'PV interneuron. No adaptation, can sustain very high firing rates.',
        'C': 150.0,     
        'g_L': 10.0,
        'E_L': -70.0,
        'V_T': -50.0,
        'Delta_T': 0.5, 
        'V_r': -58.0,
        'a': 0.0,       
        'b': 0.0,       
        'tau_w': 10.0,   
    },
    'low_threshold': {
        'name': 'Low-Threshold Spiking (SOM Interneuron)',
        'description': 'SOM interneuron. Strong adaptation, and slow decay.',
        'C': 200.0,
        'g_L': 10.0,
        'E_L': -70.0,
        'V_T': -55.0,    
        'Delta_T': 2.0,
        'V_r': -60.0,
        'a': 4.0,     
        'b': 150.0,    
        'tau_w': 300.0,  
    },
    'tonic': {
        'name': 'Tonic Spiking',
        'description': 'Sustained regular firing with no adaptation. Constant ISI.',
        'C': 200.0,      
        'g_L': 10.0,
        'E_L': -70.0,
        'V_T': -50.0,
        'Delta_T': 2.0,
        'V_r': -55.0,
        'a': 0.0,       
        'b': 60.0,     
        'tau_w': 30.0,  
    },
    'adapting': {
        'name': 'Adapting',
        'description': 'Starts fast, gradually slows down. Classic spike-frequency adaptation.',
        'C': 200.0,
        'g_L': 10.0,
        'E_L': -70.0,
        'V_T': -50.0,
        'Delta_T': 2.0,
        'V_r': -55.0,
        'a': 0.0,        
        'b': 50.0,    
        'tau_w': 100.0,  
    },
    'initial_burst': {
        'name': 'Initial Bursting',#unused, to find better parameters
        'description': 'Fires a burst at stimulus onset, then regular spikes.',
        'C': 100.0,     
        'g_L': 20.0,
        'E_L': -70.0,
        'V_T': -50.0,
        'Delta_T': 2.0,
        'V_r': -51.0,    
        'a': 0.5,    
        'b': 70.0,  
        'tau_w': 100.0,
    },
    'bursting': {
        'name': 'Bursting',
        'description': 'Rhythmic bursts of spikes.',
        'C': 100.0,    
        'g_L': 20.0,
        'E_L': -70.0,
        'V_T': -50.0,
        'Delta_T': 2.0,
        'V_r': -46.0,  
        'a': -0.5,       
        'b': 70.0,       
        'tau_w': 100.0,
    },
    'irregular': {
        'name': 'Irregular Spiking', #unused, to find better parameters
        'description': 'Chaotic-like firing with variable ISIs.',
        'C': 99.0,      
        'g_L': 10.0,
        'E_L': -70.0,
        'V_T': -50.0,
        'Delta_T': 2.0,
        'V_r': -46.0,    
        'a': -0.5,       
        'b': 70.0,
        'tau_w': 100.0,
    },
    'intrinsically_bursting': { #unused, to find better parameters
        'name': 'Intrinsically Bursting',
        'description': 'Layer 5 pyramidal. Fires bursts of 2-4 spikes.',
        'C': 200.0,
        'g_L': 10.0,
        'E_L': -70.0,
        'V_T': -50.0,
        'Delta_T': 2.0,
        'V_r': -46.0,    
        'a': 2.0,
        'b': 40.0,      
        'tau_w': 100.0,
    },

}


class SingleNeuronSimulation:

    def __init__(self, target_frequency=10.0):
        self.target_frequency = target_frequency
        self.current_input = 0.0  
        self.neuron = None
        self.spike_monitor = None
        self.state_monitor = None
        self.network = None
        self.is_setup = False

    def setup(self):
        start_scope()
        defaultclock.dt = 0.1 * ms

        eqs = '''
        dV/dt = (-g_L*(V - E_L) + g_L*Delta_T*exp((V - V_T)/Delta_T) - w + I_ext) / C : volt
        dw/dt = (a*(V - E_L) - w) / tau_w : amp
        I_ext : amp
        C : farad
        g_L : siemens
        E_L : volt
        V_T : volt
        Delta_T : volt
        a : siemens
        tau_w : second
        V_r : volt
        b : amp
        '''

        self.neuron = NeuronGroup(
            1, eqs,
            threshold='V > -20*mV',
            reset='V = V_r; w = w + b',
            method='euler'
        )

        params = NEURON_PRESETS['regular_spiking']
        self.neuron.C = params['C'] * pF
        self.neuron.g_L = params['g_L'] * nS
        self.neuron.E_L = params['E_L'] * mV
        self.neuron.V_T = params['V_T'] * mV
        self.neuron.Delta_T = params['Delta_T'] * mV
        self.neuron.V_r = params['V_r'] * mV
        self.neuron.a = params['a'] * nS
        self.neuron.b = params['b'] * pA
        self.neuron.tau_w = params['tau_w'] * ms

        self.neuron.V = params['E_L'] * mV
        self.neuron.w = 0 * pA
        self.neuron.I_ext = self.current_input * pA

        self.spike_monitor = SpikeMonitor(self.neuron)
        self.state_monitor = StateMonitor(self.neuron, 'V', record=True)

        self.network = Network(self.neuron, self.spike_monitor, self.state_monitor)
        self.is_setup = True

    def set_input_current(self, current):
        self.current_input = current
        if self.neuron is not None:
            self.neuron.I_ext = current * pA

    def run_step(self, duration_ms=100):
        if self.network is None:
            return {
                'spike_times': np.array([]),
                'voltage': np.array([]),
                'time': np.array([]),
                'firing_rate': 0.0,
                'target_frequency': self.target_frequency,
                'on_target': False
            }

        self.network.run(duration_ms * ms)

        spike_times = np.array(self.spike_monitor.t / ms)
        voltage = np.array(self.state_monitor.V[0] / mV)  
        time = np.array(self.state_monitor.t / ms)

        firing_rate = 0.0
        if len(time) > 0:
            recent_window = 500
            recent_spikes = spike_times[spike_times > (time[-1] - recent_window)]
            if len(recent_spikes) > 1:
                firing_rate = len(recent_spikes) / (recent_window / 1000)

        return {
            'spike_times': spike_times,
            'voltage': voltage,
            'time': time,
            'firing_rate': firing_rate,
            'target_frequency': self.target_frequency,
            'on_target': abs(firing_rate - self.target_frequency) < 2.0
        }

    def get_hint(self, firing_rate):
        if firing_rate == 0:
            return "The neuron is silent. Increase the input current to depolarize it!"
        elif firing_rate < self.target_frequency - 2:
            return f"Firing at {firing_rate:.1f} Hz. Need more drive to reach {self.target_frequency} Hz."
        elif firing_rate > self.target_frequency + 2:
            return f"Firing at {firing_rate:.1f} Hz. Too fast! Reduce the input slightly."
        else:
            return f"You got it! The neuron fires at ~{self.target_frequency} Hz."

    def reset(self):
        self.is_setup = False
        self.setup()


class NeuronExplorer:

    def __init__(self):
        self.preset = 'regular_spiking'
        self.params = NEURON_PRESETS[self.preset].copy()

        self.a = self.params['a']     
        self.b = self.params['b']     
        self.V_r = self.params['V_r'] 
        self.tau_w = self.params['tau_w'] 

        self.C = self.params['C']
        self.g_L = self.params['g_L']
        self.E_L = self.params['E_L']
        self.V_T = self.params['V_T']
        self.Delta_T = self.params['Delta_T']

        self.current_input = 0.0 

        self.neuron = None
        self.spike_monitor = None
        self.state_monitor = None
        self.network = None
        self.is_setup = False

    def set_preset(self, preset_name):
        if preset_name in NEURON_PRESETS:
            self.preset = preset_name
            self.params = NEURON_PRESETS[preset_name].copy()
            self.a = self.params['a']
            self.b = self.params['b']
            self.V_r = self.params['V_r']
            self.tau_w = self.params['tau_w']
            self.C = self.params['C']
            self.g_L = self.params['g_L']
            self.E_L = self.params['E_L']
            self.V_T = self.params['V_T']
            self.Delta_T = self.params['Delta_T']

    def get_preset_info(self):
        return NEURON_PRESETS.get(self.preset, {})

    def setup(self):
        start_scope()
        defaultclock.dt = 0.1 * ms

        eqs = '''
        dV/dt = (-g_L*(V - E_L) + g_L*Delta_T*exp((V - V_T)/Delta_T) - w + I_ext) / C : volt
        dw/dt = (a_param*(V - E_L) - w) / tau_w_param : amp
        I_ext : amp
        C : farad
        g_L : siemens
        E_L : volt
        V_T : volt
        Delta_T : volt
        V_r : volt
        a_param : siemens
        b_param : amp
        tau_w_param : second
        '''

        self.neuron = NeuronGroup(
            1, eqs,
            threshold='V > -20*mV',
            reset='V = V_r; w = w + b_param',
            method='euler'
        )

        self.neuron.C = self.C * pF
        self.neuron.g_L = self.g_L * nS
        self.neuron.E_L = self.E_L * mV
        self.neuron.V_T = self.V_T * mV
        self.neuron.Delta_T = self.Delta_T * mV
        self.neuron.V_r = self.V_r * mV
        self.neuron.a_param = self.a * nS
        self.neuron.b_param = self.b * pA
        self.neuron.tau_w_param = self.tau_w * ms

        self.neuron.V = self.E_L * mV
        self.neuron.w = 0 * pA
        self.neuron.I_ext = self.current_input * pA

        self.spike_monitor = SpikeMonitor(self.neuron)
        self.state_monitor = StateMonitor(self.neuron, ['V', 'w'], record=True)

        self.network = Network(self.neuron, self.spike_monitor, self.state_monitor)
        self.is_setup = True

    def set_input_current(self, current):
        self.current_input = current
        if self.neuron is not None:
            self.neuron.I_ext = current * pA

    def set_parameter(self, param_name, value):
        if param_name == 'a':
            self.a = value
            if self.neuron is not None:
                self.neuron.a_param = value * nS
        elif param_name == 'b':
            self.b = value
            if self.neuron is not None:
                self.neuron.b_param = value * pA
        elif param_name == 'V_r':
            self.V_r = value
            if self.neuron is not None:
                self.neuron.V_r = value * mV
        elif param_name == 'tau_w':
            self.tau_w = value
            if self.neuron is not None:
                self.neuron.tau_w_param = value * ms
        elif param_name == 'C':
            self.C = value
            if self.neuron is not None:
                self.neuron.C = value * pF  
        elif param_name == 'g_L':
            self.g_L = value
            if self.neuron is not None:
                self.neuron.g_L = value * nS  
        elif param_name == 'E_L':
            self.E_L = value
            if self.neuron is not None:
                self.neuron.E_L = value * mV  
        elif param_name == 'V_T':
            self.V_T = value
            if self.neuron is not None:
                self.neuron.V_T = value * mV  
        elif param_name == 'Delta_T':
            self.Delta_T = value
            if self.neuron is not None:
                self.neuron.Delta_T = value * mV  


    def run_step(self, duration_ms=100):
        if self.network is None:
            return {
                'spike_times': np.array([]),
                'voltage': np.array([]),
                'recovery': np.array([]),
                'time': np.array([]),
                'firing_rate': 0.0,
            }

        self.network.run(duration_ms * ms)

        spike_times = np.array(self.spike_monitor.t / ms)
        voltage = np.array(self.state_monitor.V[0] / mV)  
        recovery = np.array(self.state_monitor.w[0] / pA) 
        time = np.array(self.state_monitor.t / ms)

        firing_rate = 0.0
        if len(time) > 0:
            recent_window = 500
            recent_spikes = spike_times[spike_times > (time[-1] - recent_window)]
            if len(recent_spikes) > 1:
                firing_rate = len(recent_spikes) / (recent_window / 1000)

        return {
            'spike_times': spike_times,
            'voltage': voltage,
            'recovery': recovery,
            'time': time,
            'firing_rate': firing_rate,
        }

    def reset(self):
        self.is_setup = False
        self.setup()

    def get_parameter_description(self, param):
        descriptions = {
            'a': "Subthreshold adaptation (nS): How adaptation current tracks voltage below threshold. "
                 "Higher = more subthreshold adaptation.",
            'b': "Spike-triggered adaptation (pA): How much adaptation increases after each spike. "
                 "Higher = stronger spike frequency adaptation.",
            'V_r': "Reset voltage (mV): Where voltage resets after spike. "
                   "Higher (closer to threshold) = easier to fire again = bursting.",
            'tau_w': "Adaptation time constant (ms): How fast adaptation decays. "
                     "Longer = sustained adaptation effect.",
        }
        return descriptions.get(param, "")
