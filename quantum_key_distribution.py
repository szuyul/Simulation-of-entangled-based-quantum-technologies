import numpy as np
import math
import random
import statistics
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr   

# Quantum key distribution based on entangled photons proposed by Artur Ekert (E91 protocol)

def w2f_converter(wavelength):
    c = 3e8
    omega = 2*math.pi*(c/wavelength)
    return omega

class single_photon():
    def __init__(self, v, angle=math.pi/4):
        self.v = v

        # linear polarization state in |x>, |y> coordinates
        self.pol_state = np.array([[math.cos(angle)], 
                                   [math.sin(angle)]])

    def measured(self, P):
        # The @ operator can be used as a shorthand for np.matmul on ndarrays.
        prob = float(np.round(self.pol_state.T @ (P @ self.pol_state), 5)) # the probability of the state passing through the polarizer
        self.a = np.random.binomial(1, prob, 1)
        return self.a

class entangled_qubit():
    def __init__(self, v, angle=math.pi/4):
        self.v = v

        # linear polarization state in |x1, x2>, |y1, y2> coordinates
        self.pol_state = np.array([[math.cos(angle)], 
                                   [math.sin(angle)]])
        self.status = 'untouched'

    def measured(self, polarizer_angle):
        if self.status == 'untouched':
            self.status = 'collapsed'
            prob = 0.5 # as if the state appears unpolarized with 50% chances passing through the polarizer
            self.alpha = polarizer_angle # the qubit now knows the first polarizer's angle
            self.a = np.random.binomial(1, prob, 1) # the qubit reveals the first half result
            return self.a
        else: # collapsed entangled qubit
            h = bool(self.a)
            prob = h*math.cos(polarizer_angle - self.alpha)**2 + (not h)*math.sin(polarizer_angle - self.alpha)**2 # conditional probability given the first half of detection results
            self.b = np.random.binomial(1, prob, 1) # the qubit reveals the second half result
            return self.b

class polarizer():
    def __init__(self, angle=0):
        self.angle = angle

    @property
    def photon_operation(self):
        cos = math.cos(self.angle)
        sin = math.sin(self.angle)
        operator = np.array([[cos**2, sin*cos], 
                             [cos*sin, sin**2]])
        return operator

    def set_angle(self, angle):
        self.angle = angle

    def measure(self, input):
        if type(input).__name__ == 'single_photon':
            output = input.measured(self.photon_operation)
        else: # entangled_qubit case
            output = input.measured(self.angle)
        return int(output)

class observer():
    def __init__(self, name):
        self.name = name
        self.LP = polarizer(0)

        self.angle_choices = [0]
        self.angle_recording = []
        self.result_recording = []
    
    def set_angle_choices(self, angle_choices):
        self.angle_choices = angle_choices

    def observe(self, input):
        angle = random.choices(self.angle_choices, k=1)[0]
        self.LP.set_angle(angle)
        result = self.LP.measure(input)
        self.record(result)
        return result
    
    def record(self, result):
        self.angle_recording.append(self.LP.angle)
        self.result_recording.append(result)

    def copy_photon(self, input, result):
        return single_photon(input.v, self.LP.angle) if result == 1 else single_photon(input.v, self.LP.angle + math.pi/2)

    def present_recording(self):
        recording = np.array([self.angle_recording, self.result_recording])
        return recording
    
    def reset_recording(self):
        self.angle_recording.clear()
        self.result_recording.clear()

    def correlation_test(self, seq_1, seq_2, angle_diff):
        # the person only broadcasts to self his chosen basis (polarizer angles)
        idx = [i for i in range(N) if abs(seq_1[0,i] - seq_2[0,i]) == angle_diff]
        # self only broadcasts to the person the bits (idx of the sequence) to keep
        # with the same basis, self and the person both have keys now, and should be the same if no others are listening
        sub_seq_1 = [seq_1[1,i] for i in idx]
        sub_seq_2 = [seq_2[1,i] for i in idx]
        exp_correlation, _ = pearsonr(sub_seq_1, sub_seq_2)
        # the results from other basis can be broadcasted for checking eavesdropping
        exp_thy_diff = abs(exp_correlation - math.cos(2*angle_diff))
        return exp_correlation, exp_thy_diff, sub_seq_1, sub_seq_2

    def compare_results(self, person):
        correlation, _, key, _ = self.correlation_test(self.present_recording(), person.present_recording(), 0)
        print( 'Corr(' + self.name + ', ' + person.name + ') = ' + str(round(correlation, 3)) )
        return key, self.present_recording(), person.present_recording()

    def eavesdropping_test(self, person, test_angle):
        exp_thy_diff_list = []
        for i in range(len(test_angle)):
            _, exp_thy_diff, _, _ = self.correlation_test(self.present_recording(), person.present_recording(), test_angle[i])
            exp_thy_diff_list.append(exp_thy_diff)

        if statistics.mean(exp_thy_diff_list) > 0.1: # ideally this should be Bell's inequality test, |S| < 2
            print('someone is eavesdropping...!') 

#%% light source settings
wavelength = 5.513e-7
v = w2f_converter(wavelength)

#%% single photon protocol scenario
Alice = observer('Alice') # default polarizer along x axis, angle is 0
Bob = observer('Bob')

N = 1000
for i in range(N):
    angle = random.choices([0, math.pi/2], k=1)[0] # photon's polarization state
    photon_a = single_photon(v, angle)
    photon_b = single_photon(v, angle)

    Alice.observe(photon_a)
    Bob.observe(photon_b)

print('naive single photon scenario (observers using the same or perpendicular LP wrt photons to establish reliable key)')
key, _, _ = Alice.compare_results(Bob)
Alice.eavesdropping_test(Bob, [0])
Alice.reset_recording()
Bob.reset_recording()


#%% eavesdropping in single photon scenario
Eve = observer('Eve')

N = 1000
for i in range(N):
    angle = random.choices([0, math.pi/2], k=1)[0] # photon's polarization state
    photon_a = single_photon(v, angle)
    photon_b = single_photon(v, angle)

    Alice.observe(photon_a)
    result = Eve.observe(photon_b)
    photon_copied = Eve.copy_photon(photon_b, result)

    Bob.observe(photon_copied)

print('eavesdropping in naive single photon scenario (unaware of Eve lisening)')
key, _, _ = Alice.compare_results(Bob)
Alice.eavesdropping_test(Bob, [0])
Alice.reset_recording()
Bob.reset_recording()
Eve.reset_recording()


#%% E91 QKD protocol scenario
polarizer_angle_choices = [i*math.pi/8 for i in range(0, 4)]
Alice.set_angle_choices(polarizer_angle_choices)
Bob.set_angle_choices(polarizer_angle_choices)
Eve.set_angle_choices(polarizer_angle_choices)

N = 1000
for i in range(N):
    qubit = entangled_qubit(v, 0)

    Alice.observe(qubit)
    Bob.observe(qubit)

print('E91 QKD protocol scenario (arbitrary LP angle choices to photons allow reliable key)')
key, _, _ = Alice.compare_results(Bob)
Alice.eavesdropping_test(Bob, [math.pi/8, 3*math.pi/8])
Alice.reset_recording()
Bob.reset_recording()


#%% eavesdropping in QKD scenario
N = 1000
for i in range(N):
    qubit = entangled_qubit(v, 0)

    Alice.observe(qubit)
    result = Eve.observe(qubit)
    photon_copied = Eve.copy_photon(qubit, result)
    
    Bob.observe(photon_copied)

print('eavesdropping in QKD scenario (detect Eve by correlation tests)')
key, _, _ = Alice.compare_results(Bob)
Alice.eavesdropping_test(Bob, [math.pi/8, 3*math.pi/8])
# Eve.compare_results(Alice)
Alice.reset_recording()
Bob.reset_recording()
Eve.reset_recording()

i = 1