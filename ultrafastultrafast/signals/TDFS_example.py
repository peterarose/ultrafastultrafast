"""Based upon Conformation and Electronic Population Transfer in Membrane- Supported Self-Assembled Porphyrin Dimers by 2D Fluorescence Spectroscopy 
by
Alejandro Perdomo-Ortiz, Julia R. Widom, Geoffrey A. Lott, Alań Aspuru-Guzik, and Andrew H. Marcus
doi: 10.1021/jp305916x
"""

#Standard python libraries
import copy
import os
import time
import warnings

#Dependencies
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pyfftw.interfaces.numpy_fft import fft, fftshift, ifft, ifftshift, fftfreq, fftn, ifftn

#UF2
from ultrafastultrafast import Wavepackets

class TDFS(Wavepackets):
    """This class uses WavepacketBuilder to calculate the perturbative 
wavepackets needed to calculate the frequency-resolved pump-probe spectrum """
    
    def __init__(self,parameter_file_path,t21_array,t32_array,t43_array,*,
                 initial_state=0,num_conv_points=138,dt=0.1):
        self.t21_array = t21_array
        self.t32_array = t32_array
        self.t43_array = t43_array

        self.T21,self.T32,self.T43 = np.meshgrid(self.t21_array,self.t32_array,
                                                 self.t43_array,indexing='ij')

        final_time = np.max(t21_array) + np.max(t32_array) + np.max(t43_array)
        total_t_points = int(final_time/dt) + num_conv_points + 1
        
        super().__init__(parameter_file_path, num_conv_points=num_conv_points,
                         initial_state=initial_state, dt=dt,
                         total_num_time_points = total_t_points)
        self.trim_mu_impulsive()

        self.f_yield = 2 #quantum yield of doubly excited manifold relative to singly excited manifold

    def trim_mu_impulsive(self):
        """Trims down dipole operator to the minimum number of states needed, given that 
            the dipole operator has already been pruned (see ultrafastultrafast.DipolePruning).
            This trimming is done based upon the initial state the wavefunction begins in.  
            The initial state is usually the lowest energy ground state, unless thermal
            averaging is to be performed"""

        psi0_mask = self.psi0['bool_mask']
        # doesn't matter if I do this with x, y or z
        m10 = self.original_mu_GSM_to_SEM[:,:,0]
        bool10 = self.original_mu_GSM_to_SEM_boolean
        if 'DEM' in self.manifolds:
            m21 = self.original_mu_SEM_to_DEM[:,:,0]
            bool21 = self.original_mu_SEM_to_DEM_boolean
        
        dipole_matrix10, SEM_mask = self.mask_dipole_matrix(bool10,m10,psi0_mask)
        dipole_matrix01, GSM_mask = self.mask_dipole_matrix(bool10.T,m10.T,SEM_mask)
        masks = [GSM_mask,SEM_mask]
        if 'DEM' in self.manifolds:
            dipole_matrix21, DEM_mask = self.mask_dipole_matrix(bool21,m21,SEM_mask)
            masks.append(DEM_mask)

        self.psi0['bool_mask'] = self.psi0['bool_mask'][GSM_mask]

        self.trim_mu(masks)

        # This must be redone!
        self.electric_field_mask()
        self.set_U0()

    def set_pulse_times(self,t21,t32,t43):
        """Sets a list of pulse times for the 2DFS calculation. The first pulse
            is set as arriving at t = 0."""
        self.pulse_times = [0,t21,t21+t32,t21+t32+t43]
        
    def set_pulse_shapes(self,efield_list,*,plot_fields = False):
        """Sets the input list of electric field shapes, and checks that they
            are well-enough resolved. All fields should be defined on the time 
            grid self.efield_t.  We recommend working in the rotating wave
            approximation and rotating away the carrier frequency of the pulse.
            At present, if the different pulses have different center frequencies,
            we recommend you rotate away the average carrier frequency."""
        self.efields = efield_list
        if self.efield_t.size == 1:
            # M = 1 is the impulsive limit
            pass
        else:
            for field in self.efields:
                self.check_efield_resolution(field,plot_fields = plot_fields)

    def check_efield_resolution(self,efield,*,plot_fields = False):
        efield_tail = np.max(np.abs([efield[0],efield[-1]]))

        if efield.size != self.efield_t.size:
            warnings.warn('Pump must be evaluated on efield_t, the grid defined by dt and num_conv_points')

        if efield_tail > np.max(np.abs(efield))/100:
            warnings.warn('Consider using larger num_conv_points, pump does not decay to less than 1% of maximum value in time domain')
            
        efield_fft = fftshift(fft(ifftshift(efield)))*self.dt
        efield_fft_tail = np.max(np.abs([efield_fft[0],efield_fft[-1]]))
        
        if efield_fft_tail > np.max(np.abs(efield_fft))/100:
            warnings.warn('''Consider using smaller value of dt, pump does not decay to less than 1% of maximum value in frequency domain''')

        if plot_fields:
            fig, axes = plt.subplots(1,2)
            l1,l2, = axes[0].plot(self.efield_t,np.real(efield),self.efield_t,np.imag(efield))
            plt.legend([l1,l2],['Real','Imag'])
            axes[1].plot(self.efield_w,np.real(efield_fft),self.efield_w,np.imag(efield_fft))

            axes[0].set_ylabel('Electric field Amp')
            axes[0].set_xlabel('Time ($\omega_0^{-1})$')
            axes[1].set_xlabel('Frequency ($\omega_0$)')

            fig.suptitle('Check that efield is well-resolved in time and frequency')
            plt.show()

    def inner_product(self,bra_dict,ket_dict,*,time_index = -1):
        """Calculate inner product given an input bra and ket dictionary
            at a time given by the time index argument.  Default is -1, since
            2DFS is concerned with the amplitude of arriving in the given manifold
            after the pulse has finished interacting with the system."""
        bra = np.conjugate(bra_dict['psi'][:,-1])
        ket = ket_dict['psi'][:,-1]
        return np.dot(bra,ket)

    def calculate_a_wavepackets(self):
        self.psi_a = self.up(self.psi0, pulse_number = 0)

    def calculate_b_wavepackets(self):
        # First order
        self.psi_b = self.up(self.psi0, pulse_number = 1)
        
        # Second order
        self.psi_ab = self.down(self.psi_a, pulse_number = 1)

    def calculate_c_wavepackets(self):
        # First order
        self.psi_c = self.up(self.psi0, pulse_number = 2)

        # Second order
        self.psi_bc_GSM = self.down(self.psi_b, pulse_number = 2)
        self.psi_ac_GSM = self.down(self.psi_a, pulse_number = 2)
        if 'DEM' in self.manifolds:
            self.psi_bc_DEM = self.up(self.psi_b, pulse_number = 2)
            self.psi_ac_DEM = self.up(self.psi_a, pulse_number = 2)

        # Third order
        self.psi_abc = self.up(self.psi_ab, pulse_number = 2,
                               new_manifold_mask=self.psi_a['bool_mask'].copy())

    def calculate_d_wavepackets(self):
        # First order
        self.psi_d = self.up(self.psi0, pulse_number = 3)

        # Second order
        if 'DEM' in self.manifolds:
            self.psi_ad = self.up(self.psi_a, pulse_number = 3)
            self.psi_bd = self.up(self.psi_b, pulse_number = 3)

        # Third order
        self.psi_abd = self.up(self.psi_ab, pulse_number = 3,
                               new_manifold_mask=self.psi_a['bool_mask'].copy())

        self.psi_ac_GSM_d = self.up(self.psi_ac_GSM, pulse_number = 3,
                                    new_manifold_mask=self.psi_a['bool_mask'].copy())
        self.psi_bc_GSM_d = self.up(self.psi_bc_GSM, pulse_number = 3,
                                    new_manifold_mask=self.psi_a['bool_mask'].copy())

        if 'DEM' in self.manifolds:
            self.psi_ac_DEM_d = self.down(self.psi_ac_DEM, pulse_number = 3,
                                          new_manifold_mask=self.psi_a['bool_mask'].copy())
            self.psi_bc_DEM_d = self.down(self.psi_bc_DEM, pulse_number = 3,
                                          new_manifold_mask=self.psi_a['bool_mask'].copy())
            

    def nonrephasing(self):
        # Q*5a
        signal = self.inner_product(self.psi_d,self.psi_abc)
        # Q2a
        signal += self.inner_product(self.psi_bc_GSM_d,self.psi_a)
        if 'DEM' in self.manifolds:
            # Q*3b
            signal += self.inner_product(self.psi_b,self.psi_ac_DEM_d)
            # Q7b
            signal += self.f_yield * self.inner_product(self.psi_bd,self.psi_ac_DEM)

        return signal
        
    def rephasing(self):
        # Q4a
        signal = self.inner_product(self.psi_abd,self.psi_c)
        # Q3a
        signal += self.inner_product(self.psi_ac_GSM_d,self.psi_b)
        
        if 'DEM' in self.manifolds:
            # Q*2b
            signal += self.inner_product(self.psi_a,self.psi_bc_DEM_d)
            # Q*8b
            signal += self.f_yield * self.inner_product(self.psi_ad,self.psi_bc_DEM)

        return signal

    ### Calculating Spectra

    def calculate_signals(self,*,save_flag = True):
        """Calculates the rephasing and non-rephasing signals."""
        self.rephasing_signal = np.zeros((self.t21_array.size,self.t32_array.size,self.t43_array.size),dtype='complex')
        self.nonrephasing_signal = np.zeros((self.t21_array.size,self.t32_array.size,self.t43_array.size),dtype='complex')

        t21 = self.t21_array[0]
        t32 = self.t32_array[0]
        t43 = self.t43_array[0]
        self.set_pulse_times(t21,t32,t43)
        self.calculate_a_wavepackets()
        for i in range(self.t21_array.size):
            t21 = self.t21_array[i]
            self.set_pulse_times(t21,t32,t43)
            self.calculate_b_wavepackets()
            for j in range(self.t32_array.size):
                t32 = self.t32_array[j]
                self.set_pulse_times(t21,t32,t43)
                self.calculate_c_wavepackets()
                for k in range(self.t43_array.size):
                    t43 = self.t43_array[k]
                    self.set_pulse_times(t21,t32,t43)
                    self.calculate_d_wavepackets()
                    self.rephasing_signal[i,j,k] = self.rephasing()
                    self.nonrephasing_signal[i,j,k] = self.nonrephasing()

        self.rephasing_signal *= np.exp(-self.Gamma_H*(self.T21 + self.T43) - self.sigma_I**2/2*(self.T21-self.T43)**2)
        self.nonrephasing_signal *= np.exp(-self.Gamma_H*(self.T21 + self.T43) - self.sigma_I**2/2*(self.T21+self.T43)**2)

        if save_flag:
            self.save()
        
        return self.rephasing_signal, self.nonrephasing_signal

    def save(self):
        save_name = os.path.join(self.base_path,'2DFS_spectra.npz')
        np.savez(save_name,rephasing_signal = self.rephasing_signal, nonrephasing_signal = self.nonrephasing_signal,
                 t21_array = self.t21_array, t32_array = self.t32_array, t43_array = self.t43_array)

    def load(self):
        save_name = os.path.join(self.base_path,'2DFS_spectra.npz')
        np.savez(save_name,rephasing_signal = self.rephasing_signal, nonrephasing_signal = self.nonrephasing_signal,
                 t21_array = self.t21_array, t32_array = self.t32_array, t43_array = self.t43_array)

    def load_eigen_params(self):
        with open(os.path.join(self.base_path,'eigen_params.yaml'),'r') as yamlstream:
            eigen_params = yaml.load(yamlstream,Loader=yaml.SafeLoader)
            self.truncation_size = eigen_params['final truncation size']
            self.ground_ZPE = eigen_params['ground zero point energy']
            self.ground_to_excited_transition = eigen_params['ground to excited transition']

        
    def plot_fixed_t32(self,t32_time,*,part = 'real',signal='rephasing',ft = True,savefig = True, omega_0 = 1):
        """"""
        self.load_eigen_params()
        t32_index, t32_time = self.get_closest_index_and_value(t32_time,self.t32_array)
        dt21 = self.t21_array[1] - self.t21_array[0]
        dt43 = self.t43_array[1] - self.t43_array[0]
        if signal == 'rephasing':
            sig = self.rephasing_signal[:,t32_index,:]
        elif signal == 'nonrephasing':
            sig = self.nonrephasing_signal[:,t32_index,:]

        if ft:
            w21 = fftshift(fftfreq(self.t21_array.size,d=dt21))*2*np.pi
            w21 += self.ground_to_excited_transition + self.center
            w21 *= omega_0
            w43 = fftshift(fftfreq(self.t43_array.size,d=dt43))*2*np.pi
            w43 += self.ground_to_excited_transition + self.center
            w43 *= omega_0
            X,Y = np.meshgrid(w21,w43,indexing='ij')
            if signal == 'nonrephasing':
                ifft_t21_norm = self.t21_array.size * dt21
                ifft_t43_norm = self.t43_array.size * dt43
                sig = fftshift(ifftn(sig,axes=(0,1)),axes=(0,1))*ifft_t21_norm*ifft_t43_norm
            elif signal == 'rephasing':
                fft_t21_norm = dt21
                ifft_t43_norm = self.t43_array.size * dt43
                sig = fftshift(ifft(sig,axis=1),axes=(1))*ifft_t43_norm
                sig = fftshift(fft(sig,axis=0),axes=(0))*fft_t21_norm
            if omega_0 == 1:
                xlab = '$\omega_{21}$ ($\omega_0$)'
                ylab = '$\omega_{43}$ ($\omega_0$)'
            else:
                xlab = '$\omega_{21}$ (cm$^{-1}$)'
                ylab = '$\omega_{43}$ (cm$^{-1}$)'
        else:
            X = self.T21[:,0,:]
            Y = self.T43[:,0,:]
            xlab = r'$t_{21}$ ($\omega_0^{-1}$)'
            ylab = r'$t_{43}$ ($\omega_0^{-1}$)'
        
        if part == 'real':
            sig = sig.real
        if part == 'imag':
            sig = sig.imag
        
        plt.figure()
        plt.contour(X,Y,sig,12,colors='k')
        plt.contourf(X,Y,sig,12)
        plt.title(part+' '+signal + r' at $t_{32}$'+' = {}'.format(t32_time))
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.xlim([17000,21000])
        plt.ylim([17000,21000])
        plt.colorbar()
        if savefig:
            fig_name = os.path.join(self.base_path,part+'_'+signal + '_t_32_{}.png'.format(t32_time))
            plt.savefig(fig_name)
        plt.show()
