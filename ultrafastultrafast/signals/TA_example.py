#Standard python libraries
import copy
import os
import time

#Dependencies
import numpy as np
import warnings
import matplotlib.pyplot as plt
from pyfftw.interfaces.numpy_fft import fft, fftshift, ifft, ifftshift, fftfreq

#UF2
from ultrafastultrafast import Wavepackets

class TransientAbsorption(Wavepackets):
    """This class uses WavepacketBuilder to calculate the perturbative 
wavepackets needed to calculate the frequency-resolved pump-probe spectrum """
    
    def __init__(self,parameter_file_path,*, num_conv_points=138,
                 initial_state=0,dt=0.1,total_num_time_points = 2000):
        
        super().__init__(parameter_file_path, num_conv_points=num_conv_points,
                         initial_state=initial_state, dt=dt,
                         total_num_time_points = total_num_time_points)
        self.trim_mu_impulsive()

    def trim_mu_impulsive(self):
        """Trims down dipole operator to the minimum number of states needed, given that 
            the dipole operator has already been pruned (see ultrafastultrafast.DipolePruning).
            This trimming is done based upon the initial state the wavefunction begins in.  
            The initial state is usually the lowest energy ground state, unless thermal
            averaging is to be performed"""

        psi0_mask = self.psi0['bool_mask']
        # doesn't matter if I do this with x, y or z
        m10 = self.original_mu_GSM_to_SEM[:,:,0]
        m21 = self.original_mu_SEM_to_DEM[:,:,0]
        bool10 = self.original_mu_GSM_to_SEM_boolean
        bool21 = self.original_mu_SEM_to_DEM_boolean
        
        dipole_matrix10, SEM_mask = self.mask_dipole_matrix(bool10,m10,psi0_mask)
        dipole_matrix01, GSM_mask = self.mask_dipole_matrix(bool10.T,m10.T,SEM_mask)
        dipole_matrix21, DEM_mask = self.mask_dipole_matrix(bool21,m21,SEM_mask)

        masks = [GSM_mask,SEM_mask,DEM_mask]

        self.psi0['bool_mask'] = self.psi0['bool_mask'][GSM_mask]

        self.trim_mu(masks)

        # This must be redone!
        self.electric_field_mask()
        self.set_U0()

    def set_pulse_times(self,delay_time):
        """Sets a list of pulse times for the pump-probe calculation assuming
that the lowest-order, 4-wave mixing signals will be calculated, and so 4
interactions will be considered
"""
        self.pulse_times = [0,0,delay_time,delay_time]
        
    def set_pulse_shapes(self,pump_field,probe_field,*,plot_fields = True):
        """Sets a list of 4 pulse amplitudes, given an input pump shape and probe
shape. Assumes 4-wave mixing signals, and so 4 interactions
"""
        self.efields = [pump_field,pump_field,probe_field,probe_field]

        if self.efield_t.size == 1:
            pass
        else:
            pump_tail = np.max(np.abs([pump_field[0],pump_field[-1]]))
            probe_tail = np.max(np.abs([probe_field[0],probe_field[-1]]))

            if pump_field.size != self.efield_t.size:
                warnings.warn('Pump must be evaluated on efield_t, the grid defined by dt and num_conv_points')
            if probe_field.size != self.efield_t.size:
                warnings.warn('Probe must be evaluated on efield_t, the grid defined by dt and num_conv_points')

            if pump_tail > np.max(np.abs(pump_field))/100:
                warnings.warn('Consider using larger num_conv_points, pump does not decay to less than 1% of maximum value in time domain')
            if probe_tail > np.max(np.abs(probe_field))/100:
                warnings.warn('Consider using larger num_conv_points, probe does not decay to less than 1% of maximum value in time domain')

            pump_fft = fftshift(fft(ifftshift(pump_field)))*self.dt
            probe_fft = fftshift(fft(ifftshift(probe_field)))*self.dt
            pump_fft_tail = np.max(np.abs([pump_fft[0],pump_fft[-1]]))
            probe_fft_tail = np.max(np.abs([probe_fft[0],probe_fft[-1]]))

            if pump_fft_tail > np.max(np.abs(pump_fft))/100:
                warnings.warn('''Consider using smaller value of dt, pump does not decay to less than 1% of maximum value in frequency domain''')

            if probe_fft_tail > np.max(np.abs(probe_fft))/100:
                warnings.warn('''Consider using smaller value of dt, probe does not decay to less than 1% of maximum value in frequency domain''')

            if plot_fields:
                fig, axes = plt.subplots(2,2)
                l1,l2, = axes[0,0].plot(self.efield_t,np.real(pump_field),self.efield_t,np.imag(pump_field))
                plt.legend([l1,l2],['Real','Imag'])
                axes[0,1].plot(self.efield_w,np.real(pump_fft),self.efield_w,np.imag(pump_fft))
                axes[1,0].plot(self.efield_t,np.real(probe_field),self.efield_t,np.imag(probe_field))
                axes[1,1].plot(self.efield_w,np.real(probe_fft),self.efield_w,np.imag(probe_fft))

                axes[0,0].set_ylabel('Pump Amp')
                axes[1,0].set_ylabel('Probe Amp')
                axes[1,0].set_xlabel('Time')
                axes[1,1].set_xlabel('Frequency')

                fig.suptitle('Check that pump and probe are well-resolved in time and frequency')
            
        

    def calculate_pump_wavepackets(self):
        """Calculates the wavepackets that involve only the pump, and therefore
do not need to be recalculated for different delay times
"""
        # First order
        self.psi1_a = self.up(self.psi0, pulse_number = 0)
        self.psi1_b = self.up(self.psi0, pulse_number = 1)

        # Second order
        self.psi2_ab = self.down(self.psi1_a, pulse_number = 1)
        
    def calculate_probe_wavepackets(self):
        # First order
        self.psi1_c = self.up(self.psi0, pulse_number = 2,gamma=self.gamma)

        # Second order
        self.psi2_ac = self.down(self.psi1_a, pulse_number = 2,gamma=self.gamma)

        if 'DEM' in self.manifolds:
            self.psi2_bc = self.up(self.psi1_b, pulse_number = 2,gamma=self.gamma)

        # Third order
        self.psi3_abc = self.up(self.psi2_ab, pulse_number = 2,gamma=self.gamma,
                                new_manifold_mask=self.psi1_c['bool_mask'].copy())

    def calculate_overlap_wavepackets(self):
        """These diagrams only contribute when the two pulses either overlap,
or when the probe comes before the pump
"""
        self.psi2_ca = self.down(self.psi1_c, pulse_number = 0)
        self.psi3_cab = self.up(self.psi2_ca, pulse_number = 1,
                                new_manifold_mask=self.psi1_b['bool_mask'].copy())

        if 'DEM' in self.manifolds:
            self.psi2_cb = self.up(self.psi1_c, pulse_number = 1)
            self.psi3_cba = self.down(self.psi2_cb, pulse_number = 0,
                                new_manifold_mask=self.psi1_a['bool_mask'].copy())
            self.psi3_bca = self.down(self.psi2_bc, pulse_number = 0,
                                new_manifold_mask=self.psi1_a['bool_mask'].copy())

    ### Normal Diagrams
    def GSB1(self):
        return self.dipole_expectation(self.psi2_ab, self.psi1_c)

    def GSB2(self):
        return self.dipole_expectation(self.psi0,self.psi3_abc)

    def SE(self):
        return self.dipole_expectation(self.psi2_ac,self.psi1_b)

    def ESA(self):
        return self.dipole_expectation(self.psi1_a,self.psi2_bc)

    ### Overlap Diagrams
    def GSB3(self):
        return self.dipole_expectation(self.psi0,self.psi3_cab)

    def extra4(self):
        return self.dipole_expectation(self.psi0,self.psi3_cba)

    def extra5(self):
        return self.dipole_expectation(self.psi0,self.psi3_bca)

    def extra6(self):
        return self.dipole_expectation(self.psi1_a,self.psi2_cb)

    ### Calculating Spectra

    def calculate_normal_signals(self):
        tot_sig = self.SE() + self.GSB1() + self.GSB2()
        if 'DEM' in self.manifolds:
            tot_sig += self.ESA()
        return tot_sig

    def calculate_overlap_signals(self):
        overlap_sig = self.GSB3()
        if 'DEM' in self.manifolds:
            additional_sig = self.extra4() + self.extra5() + self.extra6()
            overlap_sig += additional_sig
        return overlap_sig

    def calculate_pump_probe_spectrum(self,delay_time,*,
                                      recalculate_pump_wavepackets=True,
                                      local_oscillator_number = -1):
        """Calculates the pump-probe spectrum for the delay_time specified. 
Boolean arguments:

recalculate_pump_wavepackets - must be set to True if any aspect of the electric 
field has changed since the previous calculation. Otherwise they can be re-used.

"""
        delay_index, delay_time = self.get_closest_index_and_value(delay_time,
                                                                   self.t)
        self.set_pulse_times(delay_time)
        
        if recalculate_pump_wavepackets:
            self.calculate_pump_wavepackets()
        self.calculate_probe_wavepackets()
        signal_field = self.calculate_normal_signals()
        if delay_index < self.efield_t.size*3/2:
            # The pump and probe are still considered to be over-lapping
            self.calculate_overlap_wavepackets()
            signal_field += self.calculate_overlap_signals()

        signal = self.polarization_to_signal(signal_field, local_oscillator_number = local_oscillator_number)
        return signal

    def calculate_pump_probe_spectra_vs_delay_time(self,delay_times):
        """
"""
        self.delay_times = delay_times

        min_sig_decay_time = self.t[-1] - (delay_times[-1])
        if self.gamma == 0:
            pass
        elif min_sig_decay_time < self.gamma_res/self.gamma:
            if min_sig_decay_time < 0:
                warnings.warn("""Time mesh is not long enough to support requested
                number of delay time points""")
            else:
                warnings.warn("""Spectra may not be well-resolved for all delay times. 
                For final delay time signal decays to {:.7f} of orignal value.  
                Consider selecting larger gamma value or a longer time 
                mesh""".format(np.exp(-min_sig_decay_time*self.gamma)))

        t0 = time.time()

        self.set_pulse_times(0)
        self.calculate_pump_wavepackets()
        t0_b = time.time()
        signal = np.zeros((self.w.size,delay_times.size))

        t1 = time.time()

        for n in range(delay_times.size):
            signal[:,n] = self.calculate_pump_probe_spectrum(delay_times[n], recalculate_pump_wavepackets=False)

        t2 = time.time()
        self.time_to_calculate_no_pump = t2-t1
        self.time_to_calculate = t2-t0
        self.time_to_calculate_no_pump = t1 - t0_b

        self.signal_vs_delay_times = signal

        N_k_GSM = self.psi2_ab['bool_mask'].sum()
        N_k_SEM = self.psi1_a['bool_mask'].sum()
        self.total_used_eigenvalues = {'N_k_GSM':N_k_GSM,'N_k_SEM':N_k_SEM}
        if 'DEM' in self.manifolds:
            self.total_used_eigenvalues['N_k_DEM'] = self.psi2_bc['bool_mask'].sum()

        return signal

    def save_pump_probe_spectra_vs_delay_time(self):
        save_name = os.path.join(self.base_path,'TA_spectra.npz')
        np.savez(save_name,signal = self.signal_vs_delay_times, delay_times = self.delay_times,
                 frequencies = self.w,time_to_calculate = self.time_to_calculate,
                 **self.total_used_eigenvalues)

    def load_pump_probe_spectra_vs_delay_time(self):
        load_name = os.path.join(self.base_path,'TA_spectra.npz')
        arch = np.load(load_name)
        self.signal_vs_delay_times = arch['signal']
        self.delay_times = arch['delay_times']
        self.w = arch['frequencies']
        try:
            self.time_to_calculate = arch['time_to_calculate']
        except KeyError:
            pass

        try:
            N_k_GSM = arch['N_k_GSM']
            N_k_SEM = arch['N_k_SEM']
            self.total_used_eigenvalues = {'N_k_GSM':N_k_GSM,'N_k_SEM':N_k_SEM}
            if 'DEM' in self.manifolds:
                self.total_used_eigenvalues['N_k_DEM'] = N_k_DEM = arch['N_k_DEM']

        except KeyError:
            pass

    def subtract_DC(self,sig):
        sig_fft = fft(sig,axis=1)
        sig_fft[:,0] = 0
        sig = np.real(ifft(sig_fft))
        return sig
        
    def plot_pump_probe_spectra(self,*,frequency_range=[-1000,1000], subtract_DC = True, create_figure=True,
                                color_range = 'auto',draw_colorbar = True,save_fig=True,return_signal=False):
        """Plots the transient absorption spectra with detection frequency on the
        y-axis and delay time on the x-axis.

        Args:
            frequency_range (list): sets the min (list[0]) and max (list[1]) detection frequency for y-axis
            subtract_DC (bool): if True subtract the DC component of the TA
            color_range (list): sets the min (list[0]) and max (list[1]) value for the colorbar
            draw_colorbar (bool): if True add a colorbar to the plot
            save_fig (bool): if True save the figure that is produced
        """
        # Cut out unwanted detection frequency points
        w_ind = np.where((self.w > frequency_range[0]) & (self.w < frequency_range[1]))[0]
        w = self.w[w_ind]
        sig = self.signal_vs_delay_times[w_ind,:]

        if subtract_DC:
            sig_fft = fft(sig,axis=1)
            sig_fft[:,0] = 0
            sig = np.real(ifft(sig_fft))
        ww, tt = np.meshgrid(self.delay_times,w)
        if create_figure:
            plt.figure()
        if color_range == 'auto':
            plt.pcolormesh(ww,tt,sig)
        else:
            plt.pcolormesh(ww,tt,sig,vmin=color_range[0],vmax=color_range[1])
        if draw_colorbar:
            plt.colorbar()
        plt.xlabel('Delay time ($\omega_0^{-1}$)',fontsize=16)
        plt.ylabel('Detection Frequency ($\omega_0$)',fontsize=16)
        if save_fig:
            plt.savefig(os.path.join(self.base_path,'TA_spectra'))
        if return_signal:
            return ww,tt,sig

    def plot2d_fft(self,*,delay_time_start = 1,create_figure=True,color_range = 'auto',subtract_DC=True,
                   draw_colorbar = True,frequency_range=[-1000,1000],normalize=False,phase=False,
                   save_fig=True,wT_frequency_range = 'auto'):
        w_ind = np.where((self.w > frequency_range[0]) & (self.w < frequency_range[1]))[0]
        w = self.w[w_ind]
        sig = self.signal_vs_delay_times[w_ind,:]

        delay_time_indices = np.where(self.delay_times > delay_time_start)[0]
        delay_times = self.delay_times[delay_time_indices]
        sig = sig[:,delay_time_indices]
        if normalize:
            sig /= np.dot(self.dipoles,self.dipoles)**2
        wT = fftshift(fftfreq(delay_times.size,d=(delay_times[1] - delay_times[0])))*2*np.pi
        sig_fft = fft(sig,axis=1)
        if subtract_DC:
            sig_fft[:,0] = 0
        sig_fft = fftshift(sig_fft,axes=(1))
        
        ww, wTwT = np.meshgrid(wT,w)

        if create_figure:
            plt.figure()

        if phase:
            plt.title('Phase')
            plot_sig = np.arctan2(np.imag(sig_fft),np.real(sig_fft))
        else:
            plt.title('Magnitude')
            plot_sig = np.abs(sig_fft)
        if color_range == 'auto':
            plt.pcolormesh(ww,wTwT,plot_sig)
        else:
            plt.pcolormesh(ww,wTwT,plot_sig,vmin=color_range[0],vmax=color_range[1])
        if draw_colorbar:
            plt.colorbar()
        plt.xlabel('$\omega_T$ ($\omega_0$)',fontsize=16)
        plt.ylabel('Detection Frequency ($\omega_0$)',fontsize=16)
        if wT_frequency_range == 'auto':
            plt.xlim([0,np.max(wT)])
        else:
            plt.xlim(wT_frequency_range)
        if save_fig:
            plt.savefig(self.base_path + 'TA_spectra_fft')
