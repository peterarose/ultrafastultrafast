#Standard python libraries
import copy
import warnings

#Dependencies
import numpy as np
import matplotlib.pyplot as plt
from pyfftw.interfaces.numpy_fft import fft, fftshift, ifft, ifftshift, fftfreq

#UF2
from ultrafastultrafast import Wavepackets

class TransientAbsorption5thOrder(Wavepackets):
    """This class uses UF2 to calculate the perturbative wavepackets needed 
to calculate one of the 5th order contributions to the transient absorption
signal. This is the signal proportional to pump^4*probe^2.  A separate
class would be needed to calculate the contribution due to pump^2*probe^4.
Note that this implementation is only valid for systems that have at most
2 electronic excitations.  If 3 electronic excitations are possible, more
diagrams must be considered than are currently included."""
    
    def __init__(self,parameter_file_path,*, num_conv_points=138, initial_state=0,
                 dt=0.1,total_num_time_points = 3686):
        super().__init__(parameter_file_path, num_conv_points=num_conv_points,
                         initial_state=initial_state, dt=dt,
                         total_num_time_points = total_num_time_points)

    def set_pulse_times(self,delay_time):
        """Sets a list of pulse times for the pump-probe calculation assuming
that the lowest-order, 4-wave mixing signals will be calculated, and so 4
interactions will be considered
"""
        self.pulse_times = [0,0,0,0,delay_time,delay_time]
        
    def set_pulse_shapes(self,pump_field,probe_field, *, plot_fields = True):
        """Sets a list of 4 pulse amplitudes, given an input pump shape and probe
shape. Assumes 4-wave mixing signals, and so 4 interactions.

pump_field must be evaluated on efield_t time grid
probe_field must be evaluated on efield_t time grid
"""
        self.efields = [pump_field,pump_field,pump_field,pump_field,probe_field,probe_field]
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
        self.psi1_d = self.up(self.psi0, pulse_number = 3)

        # Second order
        self.psi2_ab_1 = self.down(self.psi1_a, pulse_number = 1)
        self.psi2_dc_1 = self.down(self.psi1_d, pulse_number = 2)
        
        if 'DEM' in self.manifolds:
            self.psi2_ab_2 = self.up(self.psi1_a, pulse_number = 1) # Don't need this mask
            self.psi2_dc_2 = self.up(self.psi1_d, pulse_number = 2)

        self.masks = [self.psi2_ab_1['bool_mask'].copy(), # GSM mask
                      self.psi1_a['bool_mask'].copy()] # SEM mask
        if 'DEM' in self.manifolds:
            self.masks.append(self.psi2_ab_2['bool_mask'].copy()) # DEM mask

        # Third order
        self.psi3_abc_1 = self.up(self.psi2_ab_1, pulse_number = 2)
        if 'DEM' in self.manifolds:
            self.psi3_abc_2 = self.down(self.psi2_ab_2, pulse_number = 2)

        # Fourth order
        self.psi4_abcd_1 = self.down(self.psi3_abc_1, pulse_number = 3,
                                     new_manifold_mask = self.masks[0])
        if 'DEM' in self.manifolds:
            self.psi4_abcd_2 = self.down(self.psi3_abc_2, pulse_number = 3,
                                         new_manifold_mask = self.masks[0])
        
    def calculate_probe_wavepackets(self):
        # First order
        self.psi1_e = self.up(self.psi0, pulse_number = 4,gamma=self.gamma)

        # Second order
        self.psi2_de_1 = self.down(self.psi1_d, pulse_number = 4,gamma=self.gamma)

        if 'DEM' in self.manifolds:
            self.psi2_de_2 = self.up(self.psi1_d, pulse_number = 4,gamma=self.gamma)

        # Third order
        self.psi3_abe_1 = self.up(self.psi2_ab_1, pulse_number = 4,gamma=self.gamma)
        if 'DEM' in self.manifolds:
            self.psi3_abe_2 = self.down(self.psi2_ab_2, pulse_number = 4, gamma=self.gamma)

        # Fourth order
        self.psi4_abce_1_GSM = self.down(self.psi3_abc_1, pulse_number = 4, gamma=self.gamma,
                                   new_manifold_mask = self.masks[0])
        if 'DEM' in self.manifolds:
            self.psi4_abce_2_GSM = self.down(self.psi3_abc_2, pulse_number = 4, gamma=self.gamma,
                                             new_manifold_mask = self.masks[0])
            self.psi4_abce_1_DEM = self.up(self.psi3_abc_1, pulse_number = 4, gamma=self.gamma,
                                           new_manifold_mask = self.masks[2])
            self.psi4_abce_2_DEM = self.up(self.psi3_abc_2, pulse_number = 4, gamma=self.gamma,
                                           new_manifold_mask = self.masks[2])

        # Fifth order
        self.psi5_abcde_1 = self.up(self.psi4_abcd_1, pulse_number = 4, gamma=self.gamma,
                                    new_manifold_mask = self.masks[1])
        if 'DEM' in self.manifolds:
            self.psi5_abcde_2 = self.up(self.psi4_abcd_2, pulse_number = 4, gamma = self.gamma,
                                        new_manifold_mask = self.masks[1])

    def calculate_overlap_wavepackets(self):
        """These diagrams only contribute when the two pulses either overlap,
or when the probe comes before the pump
"""
        # Second Order
        self.psi2_ea_1 = self.down(self.psi1_e, pulse_number = 0)
        self.psi2_ae_1 = self.down(self.psi1_a, pulse_number = 4)
        
        if 'DEM' in self.manifolds:
            self.psi2_ea_2 = self.up(self.psi1_e, pulse_number = 0)
            self.psi2_ed_2 = self.up(self.psi1_e, pulse_number = 3)
            self.psi2_ae_2 = self.up(self.psi1_a, pulse_number = 4)
            

        # Third Order
        self.psi3_eab_1 = self.up(self.psi2_ea_1, pulse_number = 1)
        self.psi3_aeb_1 = self.up(self.psi2_ae_1, pulse_number = 1)
        self.psi3_abe_1 = self.up(self.psi2_ab_1, pulse_number = 4)
        
        if 'DEM' in self.manifolds:
            self.psi3_eab_2 = self.down(self.psi2_ea_2, pulse_number = 1)
            self.psi3_aeb_2 = self.down(self.psi2_ae_2, pulse_number = 1)
            self.psi3_abe_2 = self.down(self.psi2_ab_2, pulse_number = 4)

        # Fourth Order
        self.psi4_eabc_1_GSM = self.down(self.psi3_eab_1, pulse_number = 2)

        self.psi4_aebc_1_GSM = self.down(self.psi3_aeb_1, pulse_number = 2)
        
        self.psi4_abec_1_GSM = self.down(self.psi3_abe_1, pulse_number = 2)
        
        if 'DEM' in self.manifolds:
            self.psi4_eabc_1_DEM = self.up(self.psi3_eab_1, pulse_number = 2)
            self.psi4_eabc_2_GSM = self.down(self.psi3_eab_2, pulse_number = 2)
            self.psi4_eabc_2_DEM = self.up(self.psi3_eab_2, pulse_number = 2)

            self.psi4_aebc_2_GSM = self.down(self.psi3_aeb_2, pulse_number = 2)
            self.psi4_aebc_2_DEM = self.up(self.psi3_aeb_2, pulse_number = 2)
            
            self.psi4_abec_1_DEM = self.up(self.psi3_abe_1, pulse_number = 2)
            self.psi4_abec_2_GSM = self.down(self.psi3_abe_2, pulse_number = 2)
            

        # Fifth Order
        self.psi5_eabcd_1a = self.up(self.psi4_eabc_1_GSM, pulse_number = 3)
        
        self.psi5_abecd_1a = self.up(self.psi4_abec_1_GSM, pulse_number = 3)


        if 'DEM' in self.manifolds:
            self.psi5_eabcd_2a = self.up(self.psi4_eabc_2_GSM, pulse_number = 3)
            self.psi5_eabcd_1b = self.down(self.psi4_eabc_1_DEM, pulse_number = 3)
            self.psi5_eabcd_2b = self.down(self.psi4_eabc_2_DEM, pulse_number = 3)
            
            self.psi5_aebcd_2a = self.up(self.psi4_aebc_2_GSM, pulse_number = 3)
            self.psi5_aebcd_2b = self.down(self.psi4_aebc_2_DEM, pulse_number = 3)

            self.psi5_abecd_1b = self.down(self.psi4_abec_1_DEM, pulse_number = 3)
            
            self.psi5_abced_1b = self.down(self.psi4_abce_1_DEM, pulse_number = 3)
            self.psi5_abced_2b = self.down(self.psi4_abce_2_DEM, pulse_number = 3)

        
    ### Fifth Order Overlap Diagrams

    def o1(self):
        return self.dipole_expectation(self.psi3_abc_2, self.psi2_ed_2)

    def o15(self):
        return self.dipole_expectation(self.psi3_abc_1, self.psi2_ed_2)

    def o2(self):
        return self.dipole_expectation(self.psi2_dc_1, self.psi3_eab_1)

    def o13(self):
        return self.dipole_expectation(self.psi2_dc_1, self.psi3_eab_2)

    def o5(self):
        return self.dipole_expectation(self.psi1_d, self.psi4_eabc_1_DEM)

    def o11(self):
        return self.dipole_expectation(self.psi1_d, self.psi4_eabc_2_DEM)

    def o4(self):
        return self.dipole_expectation(self.psi0, self.psi5_eabcd_1a)

    def o22(self):
        return self.dipole_expectation(self.psi0, self.psi5_eabcd_1b)

    def o9(self):
        return self.dipole_expectation(self.psi0, self.psi5_eabcd_2b)

    def o16(self):
        return self.dipole_expectation(self.psi0, self.psi5_eabcd_2a)

    def o12(self):
        return self.dipole_expectation(self.psi2_dc_1, self.psi3_aeb_2)

    def o14(self):
        return self.dipole_expectation(self.psi3_aeb_1, self.psi2_dc_2)

    def o10(self):
        return self.dipole_expectation(self.psi1_d, self.psi4_aebc_2_DEM)

    def o8(self):
        return self.dipole_expectation(self.psi0, self.psi5_aebcd_2b)

    def o17(self):
        return self.dipole_expectation(self.psi0, self.psi5_aebcd_2a)

    def o18(self):
        return self.dipole_expectation(self.psi1_d, self.psi4_abec_1_DEM)

    def o7(self):
        return self.dipole_expectation(self.psi4_abec_2_GSM,self.psi1_d)
    
    def o3(self):
        return self.dipole_expectation(self.psi0, self.psi5_abecd_1a)

    def o19(self):
        return self.dipole_expectation(self.psi0, self.psi5_abecd_1b)

    def o20(self):
        return self.dipole_expectation(self.psi0, self.psi5_abced_1b)

    def o21(self):
        return self.dipole_expectation(self.psi0, self.psi5_abced_2b)

    def o6(self):
        return self.dipole_expectation(self.psi4_aebc_1_GSM,self.psi1_d)

    ### Fifth Order Diagrams

    def d1(self):
        return self.dipole_expectation(self.psi3_abe_2,self.psi2_dc_2)

    def d2(self):
        return self.dipole_expectation(self.psi2_dc_1,self.psi3_abe_1)

    def d3(self):
        return self.dipole_expectation(self.psi0,self.psi5_abcde_1)

    def d4(self):
        return self.dipole_expectation(self.psi1_d,self.psi4_abce_1_DEM)

    def d5(self):
        return self.dipole_expectation(self.psi2_de_1,self.psi3_abc_1)

    def d6(self):
        return self.dipole_expectation(self.psi3_abc_2,self.psi2_de_2)

    def d7(self):
        return self.dipole_expectation(self.psi4_abce_2_GSM,self.psi1_d)

    def d8(self):
        return self.dipole_expectation(self.psi4_abcd_1,self.psi1_e)

    def d9(self):
        return self.dipole_expectation(self.psi4_abcd_2,self.psi1_e)

    def d10(self):
        return self.dipole_expectation(self.psi0,self.psi5_abcde_2)

    def d11(self):
        return self.dipole_expectation(self.psi1_d,self.psi4_abce_2_DEM)

    def d12(self):
        return self.dipole_expectation(self.psi2_de_1,self.psi3_abc_2)

    def d13(self):
        return self.dipole_expectation(self.psi4_abce_1_GSM,self.psi1_d)

    def d14(self):
        return self.dipole_expectation(self.psi3_abc_1,self.psi2_de_2)                     

    ### Calculating Spectra

    def calculate_normal_signals(self):
        tot_sig = self.d2() + self.d3() + self.d5() + self.d8() + self.d13()
        if 'DEM' in self.manifolds:
            dem_sig = (self.d1() + self.d4() + self.d6() + self.d7() + self.d9() 
                       + self.d10() + self.d11() + self.d12() + self.d14() )
            
            tot_sig += dem_sig
        return tot_sig

    def calculate_overlap_signals(self):
        ov_sig = self.o2() + self.o3() + self.o4() + self.o6()
        if 'DEM' in self.manifolds:
            dem_sig = (self.o1() + self.o5() + self.o7() + self.o8() + self.o9()
                       + self.o10() + self.o11() + self.o12() + self.o13()
                       + self.o14() + self.o15() + self.o16() + self.o17()
                       + self.o18() + self.o19() + self.o20() + self.o21()
                       + self.o22() )
            ov_sig += dem_sig

        return ov_sig

    def calculate_pump_probe_spectrum(self,delay_time,*,
                                      recalculate_pump_wavepackets=True,
                                      local_oscillator_number = -1):
        """Calculates the pump-probe spectrum for the delay_time specified. 
Boolean arguments:

recalculate_pump_wavepackets - must be set to True if any aspect of the electric field has changed
since the previous calculation. Otherwise they can be re-used.

"""
        delay_index, delay_time = self.get_closest_index_and_value(delay_time,
                                                                   self.t)
        self.set_pulse_times(delay_time)
        
        if recalculate_pump_wavepackets:
            self.calculate_pump_wavepackets()
        self.calculate_probe_wavepackets()
        signal_field = self.calculate_normal_signals()
        if delay_index < self.efield_t.size*3/2:
            # The pump and probe are still considered to be overlapping
            self.calculate_overlap_wavepackets()
            signal_field += self.calculate_overlap_signals()

        signal = self.polarization_to_signal(signal_field, local_oscillator_number = local_oscillator_number)
        return signal

    def calculate_pump_probe_spectra_vs_delay_time(self,delay_times):
        """
"""
        self.delay_times = delay_times

        min_sig_decay_time = self.t[-1] - (delay_times[-1])
        if min_sig_decay_time < 5/self.gamma:
            if min_sig_decay_time < 0:
                warnings.warn("""Time mesh is not long enough to support requested
                number of delay time points""")
            else:
                warnings.warn("""Spectra may not be well-resolved for all delay times. 
                For final delay time signal decays to {:.7f} of orignal value.  
                Consider selecting larger gamma value or a longer time 
                mesh""".format(np.exp(-min_sig_decay_time*self.gamma)))

        self.set_pulse_times(0)
        self.calculate_pump_wavepackets()
        signal = np.zeros((self.w.size,delay_times.size))

        for n in range(delay_times.size):
            signal[:,n] = self.calculate_pump_probe_spectrum(delay_times[n], recalculate_pump_wavepackets=False)

        self.signal_vs_delay_times = signal

        return signal

    def save_pump_probe_spectra_vs_delay_time(self):
        save_name = self.base_path + 'TA_spectra_5th_order.npz'
        np.savez(save_name,signal = self.signal_vs_delay_times, delay_times = self.delay_times, frequencies = self.w)

    def load_pump_probe_spectra_vs_delay_time(self):
        load_name = self.base_path + 'TA_spectra_5th_order.npz'
        arch = np.load(load_name)
        self.signal_vs_delay_times = arch['signal']
        self.delay_times = arch['delay_times']
        self.w = arch['frequencies']

    def plot_pump_probe_spectra(self,*,frequency_range=[-1000,1000], subtract_DC = True, create_figure=True,
               color_range = 'auto',draw_colorbar = True,save_fig=True):
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
            plt.savefig(self.base_path + 'TA_spectra_5th_order')

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
            plt.savefig(self.base_path + 'TA_spectra_5th_order_fft')


class TransientAbsorption5thOrderGSB(TransientAbsorption5thOrder):

    def calculate_normal_signals(self):
        tot_sig = self.d2() + self.d3() + self.d8()
        return tot_sig
