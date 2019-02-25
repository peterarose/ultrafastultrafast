#Dependencies
import numpy as np
import matplotlib.pyplot as plt
from pyfftw.interfaces.numpy_fft import fft, fftshift, ifft, ifftshift, fftfreq

#5th order contribution to TA signal, implemented using UF2
from ultrafastultrafast.signals import TransientAbsorption5thOrder

"""The following definitions of I6_mat and kdelvec are based
upon the formulas given in Appendix B of Molecular Quantum 
Electrodynamics, by Akbar Salam
"""

I6_mat = np.eye(15)*8
I6_mat += np.diag([-5,-5,2,-5,-5,2,-5,-5,2,-5,-5,2,-5,-5],1)
I6_mat += np.diag([-5,2,2,-5,2,2,-5,2,2,-5,2,2,-5],2)
I6_mat += np.diag(np.ones(12)*-5,3)
I6_mat += np.diag(np.ones(11)*2,4)
I6_mat += np.diag([2,2,-5,2,-5,2,2,2,-5,2],5)
I6_mat += np.diag([-5,2,2,2,2,-5,-5,2,2],6)
I6_mat += np.diag([2,-5,2,-5,2,-5,2,-5],7)
I6_mat += np.diag([2,-5,-5,2,2,2,2],8)
I6_mat += np.diag(np.ones(6)*2,9)
I6_mat += np.diag([2,2,-5,-5,-5],10)
I6_mat += np.diag([-5,2,2,2],11)
I6_mat += np.diag([2,-5,2],12)
I6_mat += np.diag([2,2],13)
I6_mat += np.diag([-5],14)

I6_mat += I6_mat.T
I6_mat /= 210

def kdel(x,y):
    if x == y:
        return 1
    else:
        return 0

def kdel3(a,b,c,d,e,f):
    return kdel(a,b)*kdel(c,d)*kdel(e,f)

def kdelvec(i,j,k,l,m,n):
    vec = [kdel3(i,j,k,l,m,n),
           kdel3(i,j,k,m,n,l),
           kdel3(i,j,k,n,l,m),
           kdel3(i,k,j,l,m,n),
           kdel3(i,k,j,m,n,l),
           kdel3(i,k,j,n,l,m),
           kdel3(i,l,j,k,m,n),
           kdel3(i,l,j,m,k,n),
           kdel3(i,l,j,n,k,m),
           kdel3(i,m,j,k,n,l),
           kdel3(i,m,j,l,k,n),
           kdel3(i,m,j,n,k,l),
           kdel3(i,n,j,k,l,m),
           kdel3(i,n,j,l,k,m),
           kdel3(i,n,j,m,k,l)]
    return np.array(vec)

class TransientAbsorption5thOrderIsotropicAverage(object):
    """This class performs the isotropic average of the 6th order tensor
which is the material response produced by 6-wave mixing process"""
    def __init__(self,parameter_file_path,efield_polarization,*, num_conv_points=138,
                 initial_state=0,dt=0.1,total_num_time_points = 3686):
        self.TA = TransientAbsorption5thOrder(parameter_file_path,
                                      num_conv_points=num_conv_points,
                                      initial_state=initial_state, dt=dt,
                                      total_num_time_points=total_num_time_points)

        self.efield_polarization = efield_polarization
        self.base_path = self.TA.base_path

    def set_pulse_shapes(self,*args):
        self.TA.set_pulse_shapes(*args)

    def calculate_spectra(self,delay_times):
        left_vec = kdelvec(*self.efield_polarization)

        xyz = ['x','y','z']

        pol_options = []
        for i in range(3):
            if not np.allclose(self.TA.mu_GSM_to_SEM[:,:,i],0):
                pol_options.append(xyz[i])

        signal = np.zeros((self.TA.w.size,delay_times.size))

        for i in pol_options:
            for j in pol_options:
                for k in pol_options:
                    for l in pol_options:
                        for m in pol_options:
                            for n in pol_options:
                                right_vec = kdelvec(i,j,k,l,m,n)
                                if np.allclose(right_vec,0):
                                    pass
                                else:
                                    self.TA.set_polarization_sequence([i,j,k,l,m,n])
                                    weight = I6_mat.dot(right_vec)
                                    weight = np.dot(left_vec,weight)
                                    signal += weight * self.TA.calculate_pump_probe_spectra_vs_delay_time(delay_times)

        self.signal_vs_delay_times = signal
        self.delay_times = delay_times
        self.w = self.TA.w

        return signal

    def save_pump_probe_spectra_vs_delay_time(self):
        save_name = self.base_path + 'TA_spectra_5th_order_iso_ave.npz'
        np.savez(save_name,signal = self.signal_vs_delay_times, delay_times = self.delay_times, frequencies = self.w)

    def load_pump_probe_spectra_vs_delay_time(self):
        load_name = self.base_path + 'TA_spectra_5th_order_iso_ave.npz'
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
            plt.savefig(self.base_path + 'TA_spectra_5th_order_iso_ave')
                                    


if __name__=='__main__':
    print(kdelvec('x','x','x','x','x','x'))
    print(kdelvec('x','x','x','x','y','y'))
    print(kdelvec('x','x','x','y','y','y'))
    right = kdelvec('x','x','x','x','y','y')
    left = kdelvec('x','x','x','x','x','x')
    print(left)
    rightprod = I6_mat.dot(right)
    print(rightprod)
    prod = np.dot(left,rightprod)
    print(prod)
