#Standard python libraries
import numpy as np
import os
import yaml
import warnings
import itertools
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets
from scipy.sparse import kron
from numpy.polynomial.hermite import hermval
from scipy.special import factorial
import copy
import pyfftw
from pyfftw.interfaces.numpy_fft import fft, fftshift, ifft, ifftshift, fftfreq

class HeavisideConvolve:
    """This class calculates the discrete convolution of an array with the heaviside step function

    Attributes:
        size (int) : number of linear convolution points
        theta_fft (numpy.ndarray) : discrete fourier transform of the step function
        a : aligned array of zeros for use with the fftw algorithm
        b : empty aligned array for use with the fftw algorithm
        c : empty aligned array for use with the fftw algorithm
        fft : method for calculating the FFT of a (stores the result in b)
        ifft : method for calculating the IFFT of b (stores the result in c)
        
"""
    def __init__(self,arr_size):
        """
        Args:
            arr_size (int): number of points desired for the linear convolution"""
        self.size = arr_size
        self.theta_fft = self.heaviside_fft()
        # The discrete convolution is inherently circular. Therefore we perform the
        # convolution using 2N-1 points
        self.a = pyfftw.empty_aligned(2*self.size - 1, dtype='complex128', n=16)
        self.b = pyfftw.empty_aligned(2*self.size - 1, dtype='complex128', n=16)
        self.c = pyfftw.empty_aligned(2*self.size - 1, dtype='complex128', n=16)
        pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(10)
        self.fft = pyfftw.FFTW(self.a, self.b)
        self.ifft = pyfftw.FFTW(self.b,self.c,direction='FFTW_BACKWARD')
        self.a[:] = 0

    def heaviside_fft(self,*,value_at_zero=0.5):
        """This method calculates the FFT of the heaviside step function
        
        Args:
            value_at_zero (float): value of the heaviside step function at x = 0

        Returns:
            numpy.ndarray: the FFT of the heaviside step function
"""
        # The discrete convolution is inherently circular. Therefore we perform the
        # convolution using 2N-1 points. Spacing dx is irrelevant for evaluating
        # the heaviside step function. However it is crucial that the x = 0 is included
        t = np.arange(-self.size+1,self.size)
        y = np.heaviside(t,value_at_zero)
        return fft(y)

    def fft_convolve(self,arr,*,d=1):
        """This method calculates the linear convolution of an input with the heaviside step function
        
        Args:
            arr (numpy.ndarray): 1d array of input function values f(x)
            d (float): spacing size of grid f(x) is evaluated on, dx

        Returns:
            numpy.ndarray: linear convolution of arr with heaviside step function
"""
        self.a[:arr.size] = arr

        self.b = self.fft()
        
        self.b *= self.theta_fft

        self.b = self.b

        self.c = self.ifft()
        # Return only the results of the linear convolution
        return self.c[-arr.size:] * d

    def fft_convolve2(self,arr,*,d=1):
        """This method loops over fft_convolve in order to perform the convolution
of input array with the heaviside step function along the second axis of arr

        Args:
            arr (numpy.ndarray): 2d array of input function values f_i(x), where i is the 1st index of the array
            d (float): spacing size of grid f_i(x) is evaluated on, dx

        Returns:
            numpy.ndarray: 2d array of linear convolution of arr with heaviside step function along the second axis of arr
"""
        size0,size1 = arr.shape
        for i in range(size0):
            arr[i,:] = self.fft_convolve(arr[i,:],d=d)
        return arr

class UF2(HeavisideConvolve):
    """This class is designed to calculate perturbative wavepackets in the
light-matter interaction given the eigenvalues of the unperturbed 
hamiltonian and the material dipole operator evaluated in the
eigenbasis of the unperturbed hamiltonian.

    Attributes:

"""
    def __init__(self,parameter_file_path,*, num_conv_points=138, dt=0.1,
                 initial_state=0, total_num_time_points = 2000):
        self.base_path = parameter_file_path

        self.set_homogeneous_linewidth(0.05)

        self.load_eigenvalues()

        ### store original eigenvalues for recentering purposes
        self.orignal_eigenvalues = copy.deepcopy(self.eigenvalues)

        self.load_mu()

        self.efield_t = np.arange(-(num_conv_points//2),num_conv_points//2+num_conv_points%2) * dt
        self.efield_w = 2*np.pi*fftshift(fftfreq(self.efield_t.size,d=dt))
        self.electric_field_mask()

        # Code will not actually function until the following three empty lists are set by the user
        self.efields = [] #initialize empty list of electric field shapes
        self.polarization_sequence = [] #initialize empty polarization sequence
        self.pulse_times = [] #initialize empty list of pulse arrival times
        
        HeavisideConvolve.__init__(self,num_conv_points)
        
        # Initialize time array to be used for all desired delay times
        self.t = np.arange(-(total_num_time_points//2),total_num_time_points//2+total_num_time_points%2)*dt
        # The first pulse is assumed to arrive at t = 0, therefore shift array so that
        # it includes only points where the signal will be nonzero (number of negative time points
        # is essentially based upon width of the electric field, via the proxy of the size parameter
        self.t += self.t[-(self.size//2+1)]
        self.dt = dt
        
        f = fftshift(fftfreq(self.t.size-self.t.size%2,d=self.dt))
        self.w = 2*np.pi*f
        
        # Initialize unperturbed wavefunction
        self.set_psi0(initial_state)

        # Define the unitary operator for each manifold
        self.set_U0()

    def set_psi0(self,initial_state):
        """Creates the unperturbed wavefunction. This code does not 
support initial states that are coherent super-positions of eigenstates.
To perform thermal averaging, recalculate spectra for each initial
state that contributes to the thermal ensemble.
        Args:
            initial_state (int): index for initial eigenstate in GSM
"""
        psi0 = np.ones((1,self.t.size),dtype=complex)
        bool_mask = np.zeros(self.eigenvalues[0].size,dtype='bool')
        bool_mask[initial_state] = True
        
        # This code expects wavefunctions represented as dictionaries in the following format
        psi0_dict = {'psi':psi0,'manifold_num':0,'bool_mask':bool_mask}
        self.psi0 = psi0_dict

    def set_U0(self):
        """Calculates and stores the time-evolution operator for the unperturbed hamiltonian.
Time evolution is handled separately in each manifold, so the time-evolution operator is
stored as a list, called self.unitary.
"""
        self.unitary = []
        for i in range(len(self.eigenvalues)):
            E = self.eigenvalues[i]
            self.unitary.append( np.exp(1j * E[:,np.newaxis] * self.t[np.newaxis,:]) )

    def set_homogeneous_linewidth(self,gamma):
        self.gamma = gamma

    def get_closest_index_and_value(self,value,array):
        """Given an array and a desired value, finds the closest actual value 
stored in that array, and returns that value, along with its corresponding 
array index
"""
        index = np.argmin(np.abs(array - value))
        value = array[index]
        return index, value

    def load_eigenvalues(self):
        """Load in known eigenvalues. Must be stored as a numpy archive file,
with keys: GSM, SEM, and optionally DEM.  The eigenvalues for each manifold
must be 1d arrays, and are assumed to be ordered by increasing energy. The
energy difference between the lowest energy ground state and the lowest 
energy singly-excited state should be set to 0
"""
        eigval_save_name = os.path.join(self.base_path,'eigenvalues.npz')
        eigval_archive = np.load(eigval_save_name)
        self.manifolds = eigval_archive.keys()
        self.eigenvalues = [eigval_archive[key] for key in self.manifolds]

    def load_mu(self):
        """Load the precalculated dipole overlaps.  The dipole operator must
be stored as a .npy file, and must contain a single array, with three indices:
(upper manifold eigenfunction, lower manifold eigenfunction, cartesian coordinate).
There must be one or two files, one describing the overlap between the ground and 
singly-excited manifold, and one describing the dipole overlap between the 
singly-excited and doubly-excited manifold (optional)"""
        file_name = os.path.join(self.base_path,'mu_GSM_to_SEM_cartesian.npy')
        file_name_bool = os.path.join(self.base_path,'GSM_to_SEM_boolean_overlaps.npy')
        self.mu_GSM_to_SEM = np.load(file_name)
        try:
            self.mu_GSM_to_SEM_boolean = np.load(file_name_bool)
        except FileNotFoundError:
            self.mu_GSM_to_SEM_boolean = np.ones(self.mu_GSM_to_SEM.shape[:2],dtype='bool')

        if 'DEM' in self.manifolds:
            file_name = os.path.join(self.base_path,'mu_SEM_to_DEM_cartesian.npy')
            file_name_bool = os.path.join(self.base_path,'SEM_to_DEM_boolean_overlaps.npy')
            self.mu_SEM_to_DEM = np.load(file_name)
            try:
                self.mu_SEM_to_DEM_boolean = np.load(file_name_bool)
            except FileNotFoundError:
                self.mu_SEM_to_DEM_boolean = np.ones(self.mu_SEM_to_DEM.shape[:2],dtype='bool')

    def recenter(self,new_center = 0):
        """Substracts new_center from the SEM eigenvalues, and 2*new_center from the DEM.
This is the same as changing the frequency-domain center of the pulse, but is more
efficient from the perspective of the code  """
        self.eigenvalues[1] = self.original_eigenvalues[1] - recenter
        if 'DEM' in self.manifolds:
            self.eigenvalues[2] = self.original_eigenvalues[2] - 2*recenter

    def extend_wavefunction(self,psi_dict,pulse_start_ind,pulse_end_ind,*,check_flag = False):
        """Perturbative wavefunctions are calculated only during the time where the given pulse
is non-zero.  This function extends the wavefunction beyond those bounds by taking all values
before the interaction to be zero, and all the values to be constant (in the interaction 
picture)"""
        if check_flag:
            self.asymptote(psi_dict)

        t_slice = slice(pulse_start_ind, pulse_end_ind,1)

        psi = psi_dict['psi']

        total_psi = np.zeros((psi.shape[0],self.t.size),dtype='complex')
        total_psi[:,t_slice] = psi
        asymptote = psi_dict['psi'][:,-1]
        total_psi[:,pulse_end_ind:] = asymptote[:,np.newaxis]

        psi_dict['psi'] = total_psi
        return psi_dict

    def asymptote(self,psi_dict):
        """Check that the given wavefunction does in fact asymptote as expected"""
        #Unpack psi_dict
        psi = psi_dict['psi']
        psi_trunc = psi[:,-4:]
        psi_diff = np.diff(psi_trunc,axis=-1)
        max_psi = np.max(np.abs(psi),axis=-1)
        psi_rel_diff = psi_diff / max_psi[:,np.newaxis]
        if np.max(np.abs(psi_rel_diff)) > 1E-6:
            warnings.warn('Failed to find asymptote, max rel diff is {:.2e}'.format(np.max(np.abs(psi_rel_diff))))
        
    ### Setting the electric field to be used

    def set_polarization_sequence(self,polarization_list):
        """Sets the sequences used for either parallel or crossed pump and probe
        
        Args:
            polarization_list (list): list of four strings, can be 'x' or 'y'
        Returns:
            None: sets the attribute polarization sequence
"""

        x = np.array([1,0,0])
        y = np.array([0,1,0])
        pol_options = {'x':x,'y':y}

        self.polarization_sequence = [pol_options[pol] for pol in polarization_list]

    ### Tools for recursively calculating perturbed wavepackets using TDPT

    def dipole_matrix(self,starting_manifold_num,next_manifold_num,pulse_number):
        """Calculates the dipole matrix that connects eigenstates from one 
manifold to the next, using the known dipole moments and the efield 
polarization, determined by the pulse number.
Returns a boolean matrix listing which entries are nonzero (precalculated), 
and the actual overlap values as the second matrix."""
        pol = self.polarization_sequence[pulse_number]
        upper_manifold_num = max(starting_manifold_num,next_manifold_num)
        
        if abs(starting_manifold_num - next_manifold_num) != 1:
            warnings.warn('Can only move from manifolds 0 to 1 or 1 to 2')
            return None
        
        if upper_manifold_num == 1:
            boolean_matrix = self.mu_GSM_to_SEM_boolean
            overlap_matrix = np.tensordot(self.mu_GSM_to_SEM,pol,axes=(-1,0))
        elif upper_manifold_num == 2:
            boolean_matrix = self.mu_SEM_to_DEM_boolean
            overlap_matrix = np.tensordot(self.mu_SEM_to_DEM,pol,axes=(-1,0))

        if starting_manifold_num > next_manifold_num:
            # Take transpose if transition is down rather than up
            boolean_matrix =  boolean_matrix.T
            overlap_matrix = np.conjugate(overlap_matrix.T)

        return boolean_matrix, overlap_matrix

    def electric_field_mask(self):
        """This method determines which molecular transitions will be 
supported by the electric field.  We assume that the electric field has
0 amplitude outside the minimum and maximum frequency immplied by the 
choice of dt and num_conv_points.  Otherwise we will inadvertently 
alias transitions onto nonzero electric field amplitudes.
"""
        eig0 = self.eigenvalues[0]
        eig1 = self.eigenvalues[1]
        diff10 = eig1[:,np.newaxis] - eig0[np.newaxis,:]

        # The only transitions allowed by the electric field shape are
        inds_allowed10 = np.where((diff10 > self.efield_w[0]) & (diff10 < self.efield_w[-1]))
        mask10 = np.zeros(diff10.shape,dtype='bool')
        mask10[inds_allowed10] = 1
        self.mu_GSM_to_SEM_boolean *= mask10
        self.mu_GSM_to_SEM *= mask10[:,:,np.newaxis]

        if 'DEM' in self.manifolds:
            eig2 = self.eigenvalues[2]
            diff21 = eig2[:,np.newaxis] - eig1[np.newaxis,:]
            inds_allowed21 = np.where((diff21 >= self.efield_w[0]) & (diff21 <= self.efield_w[-1]))
            mask21 = np.zeros(diff21.shape,dtype='bool')
            mask21[inds_allowed21] = 1
            self.mu_SEM_to_DEM_boolean *= mask21
            self.mu_SEM_to_DEM *= mask21[:,:,np.newaxis]

    def mask_dipole_matrix(self,boolean_matrix,overlap_matrix,
                           starting_manifold_mask,*,next_manifold_mask = None):
        """Takes as input the boolean_matrix and the overlap matrix that it 
corresponds to. Also requires the starting manifold mask, which specifies
which states have non-zero amplitude, given the signal tolerance requested.
Trims off unnecessary starting elements, and ending elements. If 
next_manifold_mask is None, then the masking is done automatically
based upon which overlap elements are nonzero. If next_manifold_mask is
a 1D numpy boolean array, it is used as the mask for next manifold."""
        boolean_matrix = boolean_matrix[:,starting_manifold_mask]
        overlap_matrix = overlap_matrix[:,starting_manifold_mask]

        #Determine the nonzero elements of the new psi, in the
        #eigenenergy basis, n_nonzero
        if type(next_manifold_mask) is np.ndarray:
            n_nonzero = next_manifold_mask
        else:
            n_nonzero = np.any(boolean_matrix,axis=1)

        overlap_matrix = overlap_matrix[n_nonzero,:]

        return overlap_matrix, n_nonzero

    def next_order(self,psi_in_dict,manifold_change,
                   *,gamma=0,new_manifold_mask = None,
                   pulse_number = 0):
        """This function connects psi^(n) to psi^(n+1) using a DFT convolution algorithm.
psi_in_dict is the input wavefunction dictionary
manifold_change is either +/-1 (up or down)
pulse_time is the arrival time of the pulse
gamma is the optical dephasing (only use with final interaction)
new_manifold_mask is optional - define the states to be considered in the next manifold
pulse_number - 0,1,2,... (for pump-probe experiments, either 0 (pump) or 1 (probe))
               can also be set to 'impulsive'
"""
        pulse_time = self.pulse_times[pulse_number]
        pulse_time_ind = np.argmin(np.abs(self.t - pulse_time))

        pulse_start_ind = pulse_time_ind - self.size//2
        pulse_end_ind = pulse_time_ind + self.size//2 + self.size%2

        t_slice = slice(pulse_start_ind, pulse_end_ind,1)
        t = self.t[t_slice]
        
        psi_in = psi_in_dict['psi'][:,t_slice].copy()
        
        m_nonzero = psi_in_dict['bool_mask']
        starting_manifold_num = psi_in_dict['manifold_num']
        next_manifold_num = starting_manifold_num + manifold_change

        exp_factor_starting = np.conjugate(self.unitary[starting_manifold_num][m_nonzero,t_slice])

        psi_in *= exp_factor_starting
        
        boolean_matrix, overlap_matrix = self.dipole_matrix(starting_manifold_num,next_manifold_num,
                                                            pulse_number)

        overlap_matrix, n_nonzero = self.mask_dipole_matrix(boolean_matrix,overlap_matrix,m_nonzero,
                                                 next_manifold_mask = new_manifold_mask)

        psi = overlap_matrix.dot(psi_in)
        
        exp_factor1 = self.unitary[next_manifold_num][n_nonzero,t_slice]
        if gamma != 0:
            exp_factor1 *= np.exp(gamma * t[np.newaxis,:])

        psi *= exp_factor1

        if pulse_number == 'impulsive':
            heavi = np.heaviside(t-pulse_time,0.5)[np.newaxis,:]
            psi = psi[:,self.size//2].copy()
            psi = psi[:,np.newaxis] * heavi
        else:
            if next_manifold_num > starting_manifold_num:
                efield = self.efields[pulse_number]
            else:
                efield = np.conjugate(self.efields[pulse_number])
            psi = self.fft_convolve2(psi * efield[np.newaxis,:],d=self.dt)

        psi *= 1j # i/hbar Straight from perturbation theory

        psi_dict = {'psi':psi,'bool_mask':n_nonzero,'manifold_num':next_manifold_num}

        psi_dict = self.extend_wavefunction(psi_dict,pulse_start_ind,pulse_end_ind,check_flag = False)

        return psi_dict
            
    def up(self,psi_in_dict,*,gamma=0,new_manifold_mask = None,pulse_number = 0):
        """This method connects psi^(n) to psi^(n+1) where the next order psi
        is one manifold above the current manifold.
        The only difference between the up and down operators is whether the 
        next manifold is 1 above, or 1 below, the starting manifold. """

        return self.next_order(psi_in_dict,1,gamma=gamma,
                               new_manifold_mask = new_manifold_mask,
                               pulse_number = pulse_number)

    def down(self,psi_in_dict,*,gamma=0,new_manifold_mask = None,pulse_number = 0):
        """This method connects psi^(n) to psi^(n+1) where the next order psi
        is one manifold below the current manifold.
        The only difference between the up and down operators is whether the 
        next manifold is 1 above, or 1 below, the starting manifold. """

        return self.next_order(psi_in_dict,-1,gamma=gamma,
                               new_manifold_mask = new_manifold_mask,
                               pulse_number = pulse_number)

    ### Tools for taking the expectation value of the dipole operator with perturbed wavepackets

    def psi_from_dict(self,psi_dict):
        """Wavefunction dictionaries do not store zero elements. This function
uncompresses the wavefunction, restoring all of the zero elements """
        manifold = psi_dict['manifold_num']
        full_length = self.eigenvalues[manifold].size
        full_psi = np.zeros((full_length,self.t.size),dtype='complex')
        n_nonzero = psi_dict['bool_mask']
        full_psi[n_nonzero,:] = psi_dict['psi']
        return full_psi
    
    def dipole_down(self,psi_in_dict,*,new_manifold_mask = None,pulse_number = -1):
        """This method is similar to the method down, but does not
involve the electric field shape or convolutions. It is the action of 
the dipole operator on a ket without TDPT effects.  It also includes
the dot product of the final electric field polarization vector."""
        psi_in = psi_in_dict['psi']
        m_nonzero = psi_in_dict['bool_mask']
        starting_manifold_num = psi_in_dict['manifold_num']
        next_manifold_num = starting_manifold_num - 1
        
        # This function is always used as the final interaction to
        # produce the polarization field, which is a vector quantity
        # However we assume that we will calculate a signal, which
        # invovles the dot product of the polarization field with the
        # local oscillator vector. We do this now to avoid carrying
        # around the cartesian coordinates of the polarization field
        
        boolean_matrix, overlap_matrix = self.dipole_matrix(starting_manifold_num,next_manifold_num,
                                                            pulse_number = pulse_number)

        overlap_matrix, n_nonzero = self.mask_dipole_matrix(boolean_matrix,overlap_matrix,m_nonzero,
                                                 next_manifold_mask = new_manifold_mask)

        psi = overlap_matrix.dot(psi_in)

        psi_dict = {'psi':psi,'bool_mask':n_nonzero,'manifold_num':next_manifold_num}

        return psi_dict

    def dipole_expectation(self,bra_dict_original,ket_dict_original,*,pulse_number = -1):
        """Given two wavefunctions, this computes the expectation value of the two with respect 
to the dipole operator.  Both wavefunctions are taken to be kets, and the one named 'bra' is
converted to a bra by taking the complex conjugate."""
        pulse_time = self.pulse_times[pulse_number]
        pulse_time_ind = np.argmin(np.abs(self.t - pulse_time))

        pulse_start_ind = pulse_time_ind - self.size//2
        gamma_end_ind = pulse_time_ind + int(6.91/self.gamma/self.dt)

        # The signal is zero before the final pulse arrives, and persists
        # until it decays. Therefore we avoid taking the sum at times
        # where the signal is zero. This is captured by t_slice
        t_slice = slice(pulse_start_ind, gamma_end_ind,None)

        bra_in = bra_dict_original['psi'][:,t_slice].copy()
        ket_in = ket_dict_original['psi'][:,t_slice].copy()
        
        manifold1_num = bra_dict_original['manifold_num']
        manifold2_num = ket_dict_original['manifold_num']

        bra_nonzero = bra_dict_original['bool_mask']
        ket_nonzero = ket_dict_original['bool_mask']
        
        exp_factor_bra = np.conjugate(self.unitary[manifold1_num][bra_nonzero,t_slice])
        exp_factor_ket = np.conjugate(self.unitary[manifold2_num][ket_nonzero,t_slice])
        
        bra_in *= exp_factor_bra
        ket_in *= exp_factor_ket

        bra_dict = {'bool_mask':bra_nonzero,'manifold_num':manifold1_num,'psi':bra_in}
        ket_dict = {'bool_mask':ket_nonzero,'manifold_num':manifold2_num,'psi':ket_in}

        if np.abs(manifold1_num - manifold2_num) != 1:
            warnings.warn('Dipole only connects manifolds 0 to 1 or 1 to 2')
            return None

        if manifold1_num > manifold2_num:
            bra_new_mask = ket_dict['bool_mask']
            bra_dict = self.dipole_down(bra_dict,new_manifold_mask = bra_new_mask,
                                        pulse_number = pulse_number)
        else:
            ket_new_mask = bra_dict['bool_mask']
            ket_dict = self.dipole_down(ket_dict,new_manifold_mask = ket_new_mask,
                                        pulse_number = pulse_number)

        bra = bra_dict['psi']
        ket = ket_dict['psi']

        exp_val = np.sum(np.conjugate(bra) * ket,axis=0)
        
        # Initialize return array with zeros
        ret_val = np.zeros(self.t.size,dtype='complex')
        # set non-zero values using t_slice
        ret_val[t_slice] = exp_val
        return ret_val

    def integrated_dipole_expectation(self,bra_dict_original,ket_dict_original,*,pulse_number = -1):
        """Given two wavefunctions, this computes the expectation value of the two with respect 
to the dipole operator.  Both wavefunctions are taken to be kets, and the one named 'bra' is
converted to a bra by taking the complex conjugate.  This assumes that the signal will be
frequency integrated."""
        pulse_time = self.pulse_times[pulse_number]
        pulse_time_ind = np.argmin(np.abs(self.t - pulse_time))

        pulse_start_ind = pulse_time_ind - self.size//2
        pulse_end_ind = pulse_time_ind + self.size//2 + self.size%2
        
        # The signal is zero before the final pulse arrives, and persists
        # until it decays. However, if no frequency information is
        # required, fewer time points are needed for this t_slice
        t_slice = slice(pulse_start_ind, pulse_end_ind,None)

        bra_in = bra_dict_original['psi'][:,t_slice].copy()
        ket_in = ket_dict_original['psi'][:,t_slice].copy()
        
        
        manifold1_num = bra_dict_original['manifold_num']
        manifold2_num = ket_dict_original['manifold_num']

        bra_nonzero = bra_dict_original['bool_mask']
        ket_nonzero = ket_dict_original['bool_mask']
        
        exp_factor_bra = np.conjugate(self.unitary[manifold1_num][bra_nonzero,t_slice])
        exp_factor_ket = np.conjugate(self.unitary[manifold2_num][ket_nonzero,t_slice])
        
        bra_in *= exp_factor_bra
        ket_in *= exp_factor_ket

        bra_dict = {'bool_mask':bra_nonzero,'manifold_num':manifold1_num,'psi':bra_in}
        ket_dict = {'bool_mask':ket_nonzero,'manifold_num':manifold2_num,'psi':ket_in}

        if np.abs(manifold1_num - manifold2_num) != 1:
            warnings.warn('Dipole only connects manifolds 0 to 1 or 1 to 2')
            return None

        if manifold1_num > manifold2_num:
            bra_new_mask = ket_dict['bool_mask']
            bra_dict = self.dipole_down(bra_dict,new_manifold_mask = bra_new_mask,
                                        pulse_number = pulse_number)
        else:
            ket_new_mask = bra_dict['bool_mask']
            ket_dict = self.dipole_down(ket_dict,new_manifold_mask = ket_new_mask,
                                        pulse_number = pulse_number)

        bra = bra_dict['psi']
        ket = ket_dict['psi']

        exp_val = np.sum(np.conjugate(bra) * ket,axis=0)
        return exp_val

    def polarization_to_signal(self,P_of_t_in,*,return_polarization=False,
                               local_oscillator_number = -1):
        """This function generates a frequency-resolved signal from a polarization field
local_oscillator_number - usually the local oscillator will be the last pulse in the list self.efields"""
        pulse_time = self.pulse_times[local_oscillator_number]
        if self.gamma != 0:
            exp_factor = np.exp(-self.gamma * self.t)
            P_of_t_in *= exp_factor
        P_of_t = P_of_t_in
        if return_polarization:
            return P_of_t

        if local_oscillator_number == 'impulsive':
            efield = np.exp(1j*self.w*(pulse_time))
        else:
            pulse_time_ind = np.argmin(np.abs(self.t - pulse_time))

            pulse_start_ind = pulse_time_ind - self.size//2
            pulse_end_ind = pulse_time_ind + self.size//2 + self.size%2

            t_slice = slice(pulse_start_ind, pulse_end_ind,None)
            efield = np.zeros(self.t.size,dtype='complex')
            efield[t_slice] = self.efields[local_oscillator_number]
            efield = fftshift(ifft(ifftshift(efield)))*len(P_of_t)*(self.t[1]-self.t[0])/np.sqrt(2*np.pi)

        if P_of_t.size%2:
            P_of_t = P_of_t[:-1]
            efield = efield[:len(P_of_t)]

        P_of_w = fftshift(ifft(ifftshift(P_of_t)))*len(P_of_t)*(self.t[1]-self.t[0])/np.sqrt(2*np.pi)
        
        signal = np.imag(P_of_w * np.conjugate(efield))
        return signal

    def polarization_to_integrated_signal(self,P_of_t,*,
                                          local_oscillator_number = -1):
        """This function generates a frequency-integrated signal from a polarization field
local_oscillator_number - usually the local oscillator will be the last pulse in the list self.efields
"""
        pulse_time = self.pulse_times[local_oscillator_number]
        pulse_time_ind = np.argmin(np.abs(self.t - self.delay_time))

        pulse_start_ind = pulse_time_ind - self.size//2
        pulse_end_ind = pulse_time_ind + self.size//2 + self.size%2

        t_slice = slice(pulse_start_ind, pulse_end_ind,1)
        t = self.t[t_slice]
        if self.gamma != 0:
            exp_factor = np.exp(-self.gamma * t)
            P_of_t *= exp_factor

        if local_oscillator_number == 'impulsive':
            signal = P_of_t[self.size//2]
        else:
            efield = self.efields[local_oscillator_number]
            signal = np.trapz(np.conjugate(efield)*P_of_t,x=t)
        
        return np.imag(signal)
