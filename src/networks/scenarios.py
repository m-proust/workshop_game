from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor, PopulationRateMonitor,
    Network, defaultclock, start_scope, prefs,
    ms, mV, Hz, second, pA, nS, pF, Mohm
)
import numpy as np
from abc import ABC, abstractmethod

prefs.codegen.target = 'numpy'

LIF_EQS = '''
dv/dt = (-(v - v_rest) + R_m * (I_ext + I_syn)) / tau_m : volt
I_syn = I_exc - I_inh : amp
dI_exc/dt = -I_exc / tau_exc : amp
dI_inh/dt = -I_inh / tau_inh : amp
I_ext : amp
R_m : ohm
tau_m : second
v_rest : volt
tau_exc : second
tau_inh : second
'''

ADEX_EQS = '''
dv/dt = (-(v - E_L) + Delta_T * exp(clip((v - V_T) / Delta_T, -20, 20)) + R_m * (I_ext + I_syn - w)) / tau_m : volt
dw/dt = (a * (v - E_L) - w) / tau_w : amp
I_syn = I_exc - I_inh : amp
dI_exc/dt = -I_exc / tau_exc : amp
dI_inh/dt = -I_inh / tau_inh : amp
I_ext : amp
R_m : ohm
tau_m : second
E_L : volt
V_T : volt
Delta_T : volt
a : siemens
tau_w : second
tau_exc : second
tau_inh : second
'''

def create_E_cells(n_exc, input_drive_E):
    E = NeuronGroup(
        n_exc, ADEX_EQS,
        threshold='v > -40*mV',
        reset='v = E_L; w += 30*pA',  
        refractory=1.5*ms,            
        method='exponential_euler'
    )
    E.tau_m = 15*ms       
    E.R_m = 100*Mohm
    E.E_L = -70*mV
    E.V_T = -50*mV
    E.Delta_T = 2*mV
    E.a = 1*nS              
    E.tau_w = 80*ms        
    E.tau_exc = 3*ms      
    E.tau_inh = 6*ms      

    E.v = E.E_L + np.random.randn(n_exc) * 3*mV
    E.w = 0*pA
    E.I_ext = (input_drive_E + np.random.randn(n_exc) * 20) * pA
    E.I_exc = 0*pA
    E.I_inh = 0*pA
    return E


def create_PV_cells(n_pv, input_drive_PV):
    PV = NeuronGroup(
        n_pv, ADEX_EQS,
        threshold='v > -45*mV',     
        reset='v = E_L; w += 0*pA',  
        refractory=0.5*ms,        
        method='exponential_euler'
    )
    PV.tau_m = 5*ms       
    PV.R_m = 100*Mohm
    PV.E_L = -60*mV       
    PV.V_T = -52*mV
    PV.Delta_T = 0.5*mV    
    PV.a = 0*nS         
    PV.tau_w = 50*ms
    PV.tau_exc = 1*ms   
    PV.tau_inh = 2*ms   

    PV.v = PV.E_L + np.random.randn(n_pv) * 3*mV
    PV.w = 0*pA
    PV.I_ext = (input_drive_PV + np.random.randn(n_pv) * 15) * pA
    PV.I_exc = 0*pA
    PV.I_inh = 0*pA
    return PV


def create_SOM_cells(n_som, input_drive_SOM):
    SOM = NeuronGroup(
        n_som, ADEX_EQS,
        threshold='v > -40*mV',
        reset='v = E_L; w += 150*pA',  
        refractory=3*ms,
        method='exponential_euler'
    )
    SOM.tau_m = 20*ms
    SOM.R_m = 100*Mohm
    SOM.E_L = -70*mV
    SOM.V_T = -52*mV     
    SOM.Delta_T = 2*mV
    SOM.a = 2*nS            
    SOM.tau_w = 200*ms    
    SOM.tau_exc = 8*ms
    SOM.tau_inh = 30*ms    

    SOM.v = SOM.E_L + np.random.randn(n_som) * 3*mV
    SOM.w = 0*pA
    SOM.I_ext = (input_drive_SOM + np.random.randn(n_som) * 10) * pA
    SOM.I_exc = 0*pA
    SOM.I_inh = 0*pA
    return SOM

class BaseNetwork(ABC):
    def __init__(self):
        self.neurons = {}
        self.synapses = {}
        self.spike_monitors = {}
        self.rate_monitors = {}
        self.network = None
        self.is_setup = False

    @abstractmethod
    def setup(self, **kwargs):
        pass

    def _build_network(self):
        objects = []
        objects.extend(self.neurons.values())
        objects.extend(self.synapses.values())
        objects.extend(self.spike_monitors.values())
        objects.extend(self.rate_monitors.values())
        self.network = Network(objects)

    def run_step(self, duration_ms=100):
        if self.network is None:
            return {'spikes': {}, 'rates': {}}

        self.network.run(duration_ms * ms)
        return self._collect_data()

    def _collect_data(self):
        data = {
            'spikes': {},
            'rates': {},
        }

        for name, monitor in self.spike_monitors.items():
            data['spikes'][name] = {
                'times': np.array(monitor.t / ms),
                'indices': np.array(monitor.i)
            }

        for name, monitor in self.rate_monitors.items():
            data['rates'][name] = {
                'times': np.array(monitor.t / ms),
                'rates': np.array(monitor.rate / Hz)
            }

        return data



class E_PV_Network(BaseNetwork):
    def __init__(self):
        super().__init__()
        self.n_exc = 400   
        self.n_pv = 80  

        self.input_drive_E = 220.0  
        self.input_drive_PV = 0.0  

        self.J_EE = 8.0   
        self.J_EPV = 30.0 
        self.J_PVE = 35.0 
        self.J_PVPV = 20.0
 

        self.p_EE = 0.1   
        self.p_EPV = 0.3   
        self.p_PVE = 0.4   
        self.p_PVPV = 0.35  


    def setup(self, **kwargs):

        for param, value in kwargs.items():
            if hasattr(self, param) and value is not None:
                setattr(self, param, value)

        start_scope()
        defaultclock.dt = 0.05 * ms 

        self.neurons['E'] = create_E_cells(self.n_exc, self.input_drive_E)
        self.neurons['PV'] = create_PV_cells(self.n_pv, self.input_drive_PV)
        E = self.neurons['E']
        PV = self.neurons['PV']

        self.synapses['E_E'] = Synapses(E, E, 'w_syn : amp', on_pre='I_exc_post += w_syn')
        self.synapses['E_E'].connect(p=self.p_EE, condition='i != j')
        self.synapses['E_E'].w_syn = self.J_EE * pA

        self.synapses['E_PV'] = Synapses(E, PV, 'w_syn : amp', on_pre='I_exc_post += w_syn')
        self.synapses['E_PV'].connect(p=self.p_EPV)
        self.synapses['E_PV'].w_syn = self.J_EPV * pA



        self.synapses['PV_E'] = Synapses(PV, E, 'w_syn : amp', on_pre='I_inh_post += w_syn')
        self.synapses['PV_E'].connect(p=self.p_PVE)
        self.synapses['PV_E'].w_syn = self.J_PVE * pA


     

        self.synapses['PV_PV'] = Synapses(PV, PV, 'w_syn : amp', on_pre='I_inh_post += w_syn')
        self.synapses['PV_PV'].connect(p=self.p_PVPV, condition='i != j')
        self.synapses['PV_PV'].w_syn = self.J_PVPV * pA

        self.spike_monitors['E'] = SpikeMonitor(E)
        self.spike_monitors['PV'] = SpikeMonitor(PV)
        self.rate_monitors['E'] = PopulationRateMonitor(E)
        self.rate_monitors['PV'] = PopulationRateMonitor(PV)

        self._build_network()
        self.is_setup = True




class E_SOM_Network(BaseNetwork):

    def __init__(self):
        super().__init__()
        self.n_exc = 400  
        self.n_som = 80  

        self.input_drive_E = 250.0  
        self.input_drive_SOM = 0.0  

        self.J_EE = 8.0   
        self.J_ESOM = 30.0 
        self.J_SOME = 45.0 

        self.p_EE = 0.1 
        self.p_ESOM = 0.35 
        self.p_SOME = 0.35  

    def setup(self, input_drive_E=None, input_drive_SOM=None,
              J_EE=None, J_ESOM=None, J_SOME=None,
              p_EE=None, p_ESOM=None, p_SOME=None, **kwargs):

        if input_drive_E is not None: self.input_drive_E = input_drive_E
        if input_drive_SOM is not None: self.input_drive_SOM = input_drive_SOM
        if J_EE is not None: self.J_EE = J_EE
        if J_ESOM is not None: self.J_ESOM = J_ESOM
        if J_SOME is not None: self.J_SOME = J_SOME
        if p_EE is not None: self.p_EE = p_EE
        if p_ESOM is not None: self.p_ESOM = p_ESOM
        if p_SOME is not None: self.p_SOME = p_SOME

        start_scope()
        defaultclock.dt = 0.05 * ms  

        self.neurons['E'] = create_E_cells(self.n_exc, self.input_drive_E)
        self.neurons['SOM'] = create_SOM_cells(self.n_som, self.input_drive_SOM)
        E = self.neurons['E']
        SOM = self.neurons['SOM']


        self.synapses['E_E'] = Synapses(E, E, 'w_syn : amp', on_pre='I_exc_post += w_syn')
        self.synapses['E_E'].connect(p=self.p_EE, condition='i != j')
        self.synapses['E_E'].w_syn = self.J_EE * pA

        self.synapses['E_SOM'] = Synapses(E, SOM, 'w_syn : amp', on_pre='I_exc_post += w_syn')
        self.synapses['E_SOM'].connect(p=self.p_ESOM)
        self.synapses['E_SOM'].w_syn = self.J_ESOM * pA

        self.synapses['SOM_E'] = Synapses(SOM, E, 'w_syn : amp', on_pre='I_inh_post += w_syn')
        self.synapses['SOM_E'].connect(p=self.p_SOME)
        self.synapses['SOM_E'].w_syn = self.J_SOME * pA

        self.spike_monitors['E'] = SpikeMonitor(E)
        self.spike_monitors['SOM'] = SpikeMonitor(SOM)
        self.rate_monitors['E'] = PopulationRateMonitor(E)
        self.rate_monitors['SOM'] = PopulationRateMonitor(SOM)

        self._build_network()
        self.is_setup = True




class E_PV_SOM_Network(BaseNetwork):

    def __init__(self):
        super().__init__()
        self.n_exc = 400   
        self.n_pv = 80    
        self.n_som = 80  

        self.input_drive_E = 220.0  
        self.input_drive_PV = 0.0  
        self.input_drive_SOM = 0.0 
        self.J_EE = 8.0      
        self.J_EPV = 30.0  
        self.J_ESOM = 30.0  
        self.J_PVE = 35.0  
        self.J_PVPV = 20.0  
        self.J_SOME = 40.0  
        self.J_SOMPV = 20.0 

        self.p_EE = 0.1   
        self.p_EPV = 0.3  
        self.p_ESOM = 0.35 
        self.p_PVE = 0.4  
        self.p_PVPV = 0.35
        self.p_SOME = 0.3 
        self.p_SOMPV = 0.35 

    def setup(self, **kwargs):
        for param, value in kwargs.items():
            if hasattr(self, param) and value is not None:
                setattr(self, param, value)

        start_scope()
        defaultclock.dt = 0.05 * ms  
     
        self.neurons['E'] = create_E_cells(self.n_exc, self.input_drive_E)
        self.neurons['PV'] = create_PV_cells(self.n_pv, self.input_drive_PV)
        self.neurons['SOM'] = create_SOM_cells(self.n_som, self.input_drive_SOM)
        E = self.neurons['E']
        PV = self.neurons['PV']
        SOM = self.neurons['SOM']

        self.synapses['E_E'] = Synapses(E, E, 'w_syn : amp', on_pre='I_exc_post += w_syn')
        self.synapses['E_E'].connect(p=self.p_EE, condition='i != j')
        self.synapses['E_E'].w_syn = self.J_EE * pA

        self.synapses['E_PV'] = Synapses(E, PV, 'w_syn : amp', on_pre='I_exc_post += w_syn')
        self.synapses['E_PV'].connect(p=self.p_EPV)
        self.synapses['E_PV'].w_syn = self.J_EPV * pA

        self.synapses['E_SOM'] = Synapses(E, SOM, 'w_syn : amp', on_pre='I_exc_post += w_syn')
        self.synapses['E_SOM'].connect(p=self.p_ESOM)
        self.synapses['E_SOM'].w_syn = self.J_ESOM * pA

        self.synapses['PV_E'] = Synapses(PV, E, 'w_syn : amp', on_pre='I_inh_post += w_syn')
        self.synapses['PV_E'].connect(p=self.p_PVE)
        self.synapses['PV_E'].w_syn = self.J_PVE * pA

        self.synapses['SOM_E'] = Synapses(SOM, E, 'w_syn : amp', on_pre='I_inh_post += w_syn')
        self.synapses['SOM_E'].connect(p=self.p_SOME)
        self.synapses['SOM_E'].w_syn = self.J_SOME * pA

        self.synapses['SOM_PV'] = Synapses(SOM, PV, 'w_syn : amp', on_pre='I_inh_post += w_syn')
        self.synapses['SOM_PV'].connect(p=self.p_SOMPV)
        self.synapses['SOM_PV'].w_syn = self.J_SOMPV * pA

        self.synapses['PV_PV'] = Synapses(PV, PV, 'w_syn : amp', on_pre='I_inh_post += w_syn')
        self.synapses['PV_PV'].connect(p=self.p_PVPV, condition='i != j')
        self.synapses['PV_PV'].w_syn = self.J_PVPV * pA


        self.spike_monitors['E'] = SpikeMonitor(E)
        self.spike_monitors['PV'] = SpikeMonitor(PV)
        self.spike_monitors['SOM'] = SpikeMonitor(SOM)
        self.rate_monitors['E'] = PopulationRateMonitor(E)
        self.rate_monitors['PV'] = PopulationRateMonitor(PV)
        self.rate_monitors['SOM'] = PopulationRateMonitor(SOM)

        self._build_network()
        self.is_setup = True



