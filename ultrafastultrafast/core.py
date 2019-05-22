#Standard python libraries
import os
import warnings
import copy

#Dependencies - numpy, scipy, matplotlib, pyfftw
import numpy as np
import matplotlib.pyplot as plt
import pyfftw
from pyfftw.interfaces.numpy_fft import fft, fftshift, ifft, ifftshift, fftfreq
import scipy
import time

class HeavisideConvolve:
    """This class calculates the discrete convolution of an array with the 
        heaviside step function

    Attributes:
        size (int) : number of linear convolution points
        theta_fft (numpy.ndarray) : discrete fourier transform of the step 
            function
        a : aligned array of zeros for use with the fftw algorithm
        b : empty aligned array for use with the fftw algorithm
        c : empty aligned array for use with the fftw algorithm
        fft : method for calculating the FFT of a (stores the result in b)
        ifft : method for calculating the IFFT of b (stores the result in c)
        
"""
    def __init__(self,arr_size):
        """
        Args:
            arr_size (int) : number of points desired for the linear 
                convolution
"""
        self.size = arr_size
        self.theta_fft = self.heaviside_fft()
        # The discrete convolution is inherently circular. Therefore we
        # perform the convolution using 2N-1 points
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
            value_at_zero (float): value of the heaviside step function at 
                x = 0

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
        """This method calculates the linear convolution of an input with 
            the heaviside step function
        
        Args:
            arr (numpy.ndarray): 1d array of input function values f(x)
            d (float): spacing size of grid f(x) is evaluated on, dx

        Returns:
            numpy.ndarray: linear convolution of arr with heaviside step 
                function
"""
        self.a[:arr.size] = arr

        self.b = self.fft()
        
        self.b *= self.theta_fft

        self.b = self.b

        self.c = self.ifft()
        # Return only the results of the linear convolution
        return self.c[-arr.size:] * d

    def fft_convolve2(self,arr,*,d=1):
        """This method loops over fft_convolve in order to perform the convolution of input array with the heaviside step function along the second axis of arr

        Args:
            arr (numpy.ndarray): 2d array of input function values f_i(x), 
                where i is the 1st index of the array
            d (float): spacing size of grid f_i(x) is evaluated on, dx

        Returns:
            numpy.ndarray: 2d array of linear convolution of arr with 
                heaviside step function along the second axis of arr
"""
        size0,size1 = arr.shape
        for i in range(size0):
            arr[i,:] = self.fft_convolve(arr[i,:],d=d)
        return arr

class Wavepackets(HeavisideConvolve):
    """This class is designed to calculate perturbative wavepackets in the
        light-matter interaction given the eigenvalues of the unperturbed 
        hamiltonian and the material dipole operator evaluated in the
        eigenbasis of the unperturbed hamiltonian.

    Args:
        file_path (string): path to folder containing eigenvalues and the
            dipole operator for the system Hamiltonian
        num_conv_points (int): number of desired points for linear 
            convolution. Also number of points used to resolve all optical
            pulse shapes
        dt (float): time spacing used to resolve the shape of all optical
            pulses
        initial_state (int): index of initial state for psi^0
        total_num_time_poitns (int): total number of time points used for
            the spectroscopic calculations.

"""
    def __init__(self,file_path,*, num_conv_points=138, dt=0.1,center = 0,
                 initial_state=0, total_num_time_points = 2000):
        self.slicing_time = 0
        self.interpolation_time = 0
        self.expectation_time = 0
        self.next_order_expectation_time = 0
        self.convolution_time = 0
        self.extend_time = 0
        self.mask_time = 0
        self.dipole_time = 0
        self.base_path = file_path

        self.undersample_factor = 1

        self.set_homogeneous_linewidth(0.05)

        self.load_eigenvalues()

        self.load_mu()

        self.efield_t = np.arange(-(num_conv_points//2),num_conv_points//2+num_conv_points%2) * dt
        self.efield_w = 2*np.pi*fftshift(fftfreq(self.efield_t.size,d=dt))

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
        
        # f = fftshift(fftfreq(self.t.size-self.t.size%2,d=self.dt))
        f = fftshift(fftfreq(self.t.size,d=self.dt))
        self.w = 2*np.pi*f
        
        # Initialize unperturbed wavefunction
        self.set_psi0(initial_state)

        # Define the unitary operator for each manifold in the RWA given the rotating frequency center
        self.recenter(new_center = center)

        self.gamma_res = 6.91

    def set_psi0(self,initial_state):
        """Creates the unperturbed wavefunction. This code does not 
            support initial states that are coherent super-positions of 
            eigenstates. To perform thermal averaging, recalculate spectra 
            for each initial state that contributes to the thermal ensemble.
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
        """Calculates and stores the time-evolution operator for the 
            unperturbed hamiltonian.
            Time evolution is handled separately in each manifold, so the 
            time-evolution operator is stored as a list, called self.unitary.
"""
        self.unitary = []
        for i in range(len(self.eigenvalues)):
            E = self.eigenvalues[i]
            self.unitary.append( np.exp(-1j * E[:,np.newaxis] * self.t[np.newaxis,:]) )

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
        
        ### store original eigenvalues for recentering purposes
        self.original_eigenvalues = copy.deepcopy(self.eigenvalues)

    def load_mu(self):
        """Load the precalculated dipole overlaps.  The dipole operator must
            be stored as a .npz file, and must contain at least one array, each with three 
            indices: (upper manifold eigenfunction, lower manifold eigenfunction, 
            cartesian coordinate).  So far this code supports up to three manifolds, and
            therefore up to two dipole operators (connecting between manifolds)"""
        file_name = os.path.join(self.base_path,'mu.npz')
        file_name_pruned = os.path.join(self.base_path,'mu_pruned.npz')
        file_name_bool = os.path.join(self.base_path,'mu_boolean.npz')
        try:
            mu_archive = np.load(file_name_pruned)
            mu_boolean_archive = np.load(file_name_bool)
            self.mu_GSM_to_SEM_boolean = mu_boolean_archive['GSM_to_SEM']
            pruned = True
        except FileNotFoundError:
            mu_archive = np.load(file_name)
            pruned = False
        self.mu_GSM_to_SEM = mu_archive['GSM_to_SEM']
        if pruned == False:
            self.mu_GSM_to_SEM_boolean = np.ones(self.mu_GSM_to_SEM.shape[:2],dtype='bool')

        # Following two lines necessary for recentering and remasking based upon electric field shape
        # self.original_mu_GSM_to_SEM_boolean = copy.deepcopy(self.mu_GSM_to_SEM_boolean)
        self.original_mu_GSM_to_SEM_boolean = self.mu_GSM_to_SEM_boolean
        # self.original_mu_GSM_to_SEM = copy.deepcopy(self.mu_GSM_to_SEM)
        self.original_mu_GSM_to_SEM = self.mu_GSM_to_SEM
        
        if 'DEM' in self.manifolds:
            self.mu_SEM_to_DEM = mu_archive['SEM_to_DEM']
            if pruned == True:
                self.mu_SEM_to_DEM_boolean = mu_boolean_archive['SEM_to_DEM']
            else:
                self.mu_SEM_to_DEM_boolean = np.ones(self.mu_SEM_to_DEM.shape[:2],dtype='bool')
            # Following two lines necessary for recentering and remasking based upon electric field shape
            # self.original_mu_SEM_to_DEM_boolean = copy.deepcopy(self.mu_SEM_to_DEM_boolean)
            self.original_mu_SEM_to_DEM_boolean = self.mu_SEM_to_DEM_boolean
            # self.original_mu_SEM_to_DEM = copy.deepcopy(self.mu_SEM_to_DEM)
            self.original_mu_SEM_to_DEM = self.mu_SEM_to_DEM
    
    def trim_mu(self,masks):
        """Trims dipole operators based on a list of n masks, where n is the number 
            of manifolds.  This modifies mu and the list of eigenvalues, and can't
            be undone without reloading mu and the eigenvalues.  Wavefunctions won't
            be easily reconstructed once this is done.  This step is not necessary,
            it only serves to speed up calculations and free up some memory."""
        # Eventually I want to treat mu as a list, not with these cumbersome names
        mu = [self.original_mu_GSM_to_SEM]
        mu_boolean = [self.original_mu_GSM_to_SEM_boolean]
        if 'DEM' in self.manifolds:
            mu.append(self.original_mu_SEM_to_DEM)
            mu_boolean.append(self.original_mu_SEM_to_DEM_boolean)
        
        for i in range(len(masks)-1):
            # Using ellipsis so that if I ever convert to only working with mu_x, mu_y, mu_z at a time, this
            # should still work
            mu[i] = mu[i][:,masks[i],...]
            mu[i] = mu[i][masks[i+1],...]
            mu_boolean[i] = mu_boolean[i][:,masks[i]]
            mu_boolean[i] = mu_boolean[i][masks[i+1],:]

        self.mu_GSM_to_SEM = mu[0]
        self.mu_GSM_to_SEM_boolean = mu_boolean[0]
        if 'DEM' in self.manifolds:
            self.mu_SEM_to_DEM = mu[1]
            self.mu_SEM_to_DEM_boolean = mu_boolean[1]

        for i in range(len(masks)):
            self.eigenvalues[i] = self.eigenvalues[i][masks[i]]

        self.original_eigenvalues = copy.deepcopy(self.eigenvalues)
        
        self.original_mu_GSM_to_SEM_boolean = self.mu_GSM_to_SEM_boolean
        self.original_mu_GSM_to_SEM = self.mu_GSM_to_SEM
        if 'DEM' in self.manifolds:
            self.original_mu_SEM_to_DEM_boolean = self.mu_SEM_to_DEM_boolean
            self.original_mu_SEM_to_DEM = self.mu_SEM_to_DEM

    def recenter(self,new_center = 0):
        """Substracts new_center from the SEM eigenvalues, and 2*new_center from the DEM.
This is the same as changing the frequency-domain center of the pulse, but is more
efficient from the perspective of the code  """
        self.eigenvalues[1] = self.original_eigenvalues[1] - new_center
        if 'DEM' in self.manifolds:
            self.eigenvalues[2] = self.original_eigenvalues[2] - 2*new_center
        self.center = new_center
        self.electric_field_mask()
        self.set_U0()

    def extend_wavefunction(self,psi_dict,pulse_start_ind,pulse_end_ind,*,check_flag = False,
                            gamma_end_ind = None):
        """Perturbative wavefunctions are calculated only during the time where the given pulse
is non-zero.  This function extends the wavefunction beyond those bounds by taking all values
before the interaction to be zero, and all the values to be constant (in the interaction 
picture)"""
        t0 = time.time()
        if check_flag:
            self.asymptote(psi_dict)

        t_slice = slice(pulse_start_ind, pulse_end_ind,1)

        psi = psi_dict['psi']

        m_nonzero = psi_dict['bool_mask']
        manifold_num = psi_dict['manifold_num']

        total_psi = np.zeros((psi.shape[0],self.t.size),dtype='complex')
        total_psi[:,t_slice] = psi
        asymptote = psi_dict['psi'][:,-1]
        total_psi[:,pulse_end_ind:gamma_end_ind] = asymptote[:,np.newaxis]
        non_zero_inds = slice(pulse_start_ind,gamma_end_ind,None)
        total_psi[:,non_zero_inds] *= self.unitary[manifold_num][m_nonzero,non_zero_inds]

        psi_dict['psi'] = total_psi
        t1 = time.time()
        self.extend_time += t1-t0
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
            polarization_list (list): list of four strings, can be 'x','y' or 'z'
        Returns:
            None: sets the attribute polarization sequence
"""

        x = np.array([1,0,0])
        y = np.array([0,1,0])
        z = np.array([0,0,1])
        pol_options = {'x':x,'y':y,'z':z}

        self.polarization_sequence = [pol_options[pol] for pol in polarization_list]


    ### Tools for recursively calculating perturbed wavepackets using TDPT

    def dipole_matrix(self,starting_manifold_num,next_manifold_num,pulse_number):
        """Calculates the dipole matrix that connects eigenstates from one 
manifold to the next, using the known dipole moments and the efield 
polarization, determined by the pulse number.
Returns a boolean matrix listing which entries are nonzero (precalculated), 
and the actual overlap values as the second matrix."""
        t0 = time.time()
        pol = self.polarization_sequence[pulse_number]
        upper_manifold_num = max(starting_manifold_num,next_manifold_num)

        x = np.array([1,0,0])
        y = np.array([0,1,0])
        z = np.array([0,0,1])
        
        if abs(starting_manifold_num - next_manifold_num) != 1:
            warnings.warn('Can only move from manifolds 0 to 1 or 1 to 2')
            return None

        if upper_manifold_num == 1:
            boolean_matrix = self.mu_GSM_to_SEM_boolean
            if np.all(pol == x):
                overlap_matrix = self.mu_GSM_to_SEM[:,:,0]
            elif np.all(pol == y):
                overlap_matrix = self.mu_GSM_to_SEM[:,:,1]
            elif np.all(pol == z):
                overlap_matrix = self.mu_GSM_to_SEM[:,:,2]
            else:
                overlap_matrix = np.tensordot(self.mu_GSM_to_SEM,pol,axes=(-1,0))
        elif upper_manifold_num == 2:
            boolean_matrix = self.mu_SEM_to_DEM_boolean
            if np.all(pol == x):
                overlap_matrix = self.mu_SEM_to_DEM[:,:,0]
            elif np.all(pol == y):
                overlap_matrix = self.mu_SEM_to_DEM[:,:,1]
            elif np.all(pol == z):
                overlap_matrix = self.mu_SEM_to_DEM[:,:,2]
            else:
                overlap_matrix = np.tensordot(self.mu_SEM_to_DEM,pol,axes=(-1,0))

        if starting_manifold_num > next_manifold_num:
            # Take transpose if transition is down rather than up
            boolean_matrix =  boolean_matrix.T
            if overlap_matrix.dtype == 'complex':
                # This step can be slow for large matrices, so skip if possible
                overlap_matrix = np.conjugate(overlap_matrix.T)
            else:
                overlap_matrix = overlap_matrix.T

        t1 = time.time()
        self.dipole_time += t1-t0

        return boolean_matrix, overlap_matrix

    def electric_field_mask(self):
        """This method determines which molecular transitions will be 
supported by the electric field.  We assume that the electric field has
0 amplitude outside the minimum and maximum frequency immplied by the 
choice of dt and num_conv_points.  Otherwise we will inadvertently 
alias transitions onto nonzero electric field amplitudes.
"""
        if self.efield_t.size == 1:
            pass
        else:
            eig0 = self.eigenvalues[0]
            eig1 = self.eigenvalues[1]
            diff10 = eig1[:,np.newaxis] - eig0[np.newaxis,:]

            # The only transitions allowed by the electric field shape are
            inds_allowed10 = np.where((diff10 > self.efield_w[0]) & (diff10 < self.efield_w[-1]))
            mask10 = np.zeros(diff10.shape,dtype='bool')
            mask10[inds_allowed10] = 1
            self.mu_GSM_to_SEM_boolean = self.original_mu_GSM_to_SEM_boolean * mask10
            self.mu_GSM_to_SEM = self.original_mu_GSM_to_SEM * mask10[:,:,np.newaxis]

            if 'DEM' in self.manifolds:
                eig2 = self.eigenvalues[2]
                diff21 = eig2[:,np.newaxis] - eig1[np.newaxis,:]
                inds_allowed21 = np.where((diff21 >= self.efield_w[0]) & (diff21 <= self.efield_w[-1]))
                mask21 = np.zeros(diff21.shape,dtype='bool')
                mask21[inds_allowed21] = 1
                self.mu_SEM_to_DEM_boolean = self.original_mu_SEM_to_DEM_boolean * mask21
                self.mu_SEM_to_DEM = self.original_mu_SEM_to_DEM * mask21[:,:,np.newaxis]

    def mask_dipole_matrix(self,boolean_matrix,overlap_matrix,
                           starting_manifold_mask,*,next_manifold_mask = None):
        """Takes as input the boolean_matrix and the overlap matrix that it 
            corresponds to. Also requires the starting manifold mask, which specifies
            which states have non-zero amplitude, given the signal tolerance requested.
            Trims off unnecessary starting elements, and ending elements. If 
            next_manifold_mask is None, then the masking is done automatically
            based upon which overlap elements are nonzero. If next_manifold_mask is
            a 1D numpy boolean array, it is used as the mask for next manifold."""
        t0 = time.time()
        if np.all(starting_manifold_mask == True):
            pass
        else:
            boolean_matrix = boolean_matrix[:,starting_manifold_mask]
            overlap_matrix = overlap_matrix[:,starting_manifold_mask]

        #Determine the nonzero elements of the new psi, in the
        #eigenenergy basis, n_nonzero
        if type(next_manifold_mask) is np.ndarray:
            n_nonzero = next_manifold_mask
        else:
            n_nonzero = np.any(boolean_matrix,axis=1)
        if np.all(n_nonzero == True):
            pass
        else:
            overlap_matrix = overlap_matrix[n_nonzero,:]

        t1 = time.time()
        self.mask_time += t1-t0

        return overlap_matrix, n_nonzero

    def next_order(self,psi_in_dict,manifold_change,
                   *,gamma=0,new_manifold_mask = None,
                   pulse_number = 0):
        """This function connects psi_p to psi+pj^(*) using a DFT convolution algorithm.

        Args:
            psi_in_dict (dict): input wavefunction dictionary
            manifold_change (int): is either +/-1 (up or down)
            pulse_number (int): index of optical pulse (0,1,2,...) can also be set to
                'impulsive'
            gamma (float): optical dephasing (only use with final interaction)
            new_manifold_mask (np.ndarray): optional - define the states to be considered 
                in the next manifold
        
        Return:
            psi_dict (dict): next-order wavefunction
"""
        pulse_time = self.pulse_times[pulse_number]
        pulse_time_ind = np.argmin(np.abs(self.t - pulse_time))
        if np.allclose(self.t[pulse_time_ind],pulse_time):
            pass
        else:
            warnings.warn('Pulse time is not an integer multiple of dt, changing requested pulse time, {}, to the closest value of seft.t, {}'.format(pulse_time,self.t[pulse_time_ind]))

        pulse_start_ind = pulse_time_ind - self.size//2
        pulse_end_ind = pulse_time_ind + self.size//2 + self.size%2

        t_slice = slice(pulse_start_ind, pulse_end_ind,1)
        t = self.t[t_slice]
        
        psi_in = psi_in_dict['psi'][:,t_slice].copy()
        
        m_nonzero = psi_in_dict['bool_mask']
        starting_manifold_num = psi_in_dict['manifold_num']
        next_manifold_num = starting_manifold_num + manifold_change

        # exp_factor_starting = self.unitary[starting_manifold_num][m_nonzero,t_slice]
        # psi_in *= exp_factor_starting
        
        
        boolean_matrix, overlap_matrix = self.dipole_matrix(starting_manifold_num,next_manifold_num,
                                                            pulse_number)

        overlap_matrix, n_nonzero = self.mask_dipole_matrix(boolean_matrix,overlap_matrix,m_nonzero,
                                                                next_manifold_mask = new_manifold_mask)
        t0 = time.time()
        psi = overlap_matrix.dot(psi_in)
                
        t1 = time.time()
        self.next_order_expectation_time += t1-t0
        
        exp_factor1 = np.conjugate(self.unitary[next_manifold_num][n_nonzero,t_slice])

        if gamma != 0:
            gamma_end_ind = pulse_time_ind + int(self.gamma_res/gamma/self.dt)
        else:
            gamma_end_ind = None

        psi *= exp_factor1

        t0 = time.time()

        M = self.efield_t.size

        if M == 1:
            psi *= self.efields[pulse_number]
        else:
            if next_manifold_num > starting_manifold_num:
                efield = self.efields[pulse_number]
            else:
                efield = np.conjugate(self.efields[pulse_number])
            psi = self.fft_convolve2(psi * efield[np.newaxis,:],d=self.dt)

        t1 = time.time()
        self.convolution_time += t1-t0

        psi *= 1j # i/hbar Straight from perturbation theory

        psi_dict = {'psi':psi,'bool_mask':n_nonzero,'manifold_num':next_manifold_num}

        psi_dict = self.extend_wavefunction(psi_dict,pulse_start_ind,pulse_end_ind,check_flag = False,
                                            gamma_end_ind = gamma_end_ind)

    
        return psi_dict
            
    def up(self,psi_in_dict,*,gamma=0,new_manifold_mask = None,pulse_number = 0):
        """This method connects psi_p to psi_pj where the next order psi
            is one manifold above the current manifold.

        Args:
            psi_in_dict (dict): input wavefunction dictionary
            pulse_number (int): index of optical pulse (0,1,2,...) can also be set to
                'impulsive'
            gamma (float): optical dephasing (only use with final interaction)
            new_manifold_mask (np.ndarray): optional - define the states to be considered 
                in the next manifold

        Returns:
            (dict): output from method next_order
"""

        return self.next_order(psi_in_dict,1,gamma=gamma,
                               new_manifold_mask = new_manifold_mask,
                               pulse_number = pulse_number)

    def down(self,psi_in_dict,*,gamma=0,new_manifold_mask = None,pulse_number = 0):
        """This method connects psi_p to psi_pj^* where the next order psi
            is one manifold below the current manifold.

        Args:
            psi_in_dict (dict): input wavefunction dictionary
            pulse_number (int): index of optical pulse (0,1,2,...) can also be set to
                'impulsive'
            gamma (float): optical dephasing (only use with final interaction)
            new_manifold_mask (np.ndarray): optional - define the states to be considered 
                in the next manifold

        Returns:
            (dict): output from method next_order
"""

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

    def set_undersample_factor(self,frequency_resolution):
        """dt is set by the pulse. However, the system dynamics may not require such a 
            small dt.  Therefore, this allows the user to set a requested frequency
            resolution for any spectrally resolved signals."""
        # f = pi/dt
        dt = np.pi/frequency_resolution
        u = int(np.floor(dt/self.dt))
        self.undersample_factor = max(u,1)

    def dipole_expectation(self,bra_dict_original,ket_dict_original,*,pulse_number = -1):
        """Computes the expectation value of the two wavefunctions with respect 
            to the dipole operator.  Both wavefunctions are taken to be kets, and the one 
            named 'bra' is converted to a bra by taking the complex conjugate."""
        t0 = time.time()
        pulse_time = self.pulse_times[pulse_number]
        pulse_time_ind = np.argmin(np.abs(self.t - pulse_time))

        pulse_start_ind = pulse_time_ind - self.size//2
        pulse_end_ind = pulse_time_ind + self.size//2 + self.size%2
        if self.gamma != 0:
            gamma_end_ind = pulse_time_ind + int(self.gamma_res/self.gamma/self.dt)
        else:
            gamma_end_ind = None

        # The signal is zero before the final pulse arrives, and persists
        # until it decays. Therefore we avoid taking the sum at times
        # where the signal is zero. This is captured by t_slice
        t_slice = slice(pulse_start_ind, gamma_end_ind,None)

        u = self.undersample_factor
        if u != 1:
            t_slice1 = slice(pulse_start_ind, pulse_end_ind,None)
            t_slice2 = slice(pulse_end_ind, gamma_end_ind+u+1,u)

            t = self.t[t_slice]
            t1 = self.t[t_slice1]
            t2 = self.t[t_slice2]

            bra_in1 = bra_dict_original['psi'][:,t_slice1]#.copy()
            bra_in2 = bra_dict_original['psi'][:,t_slice2]#.copy()
            ket_in1 = ket_dict_original['psi'][:,t_slice1]#.copy()
            ket_in2 = ket_dict_original['psi'][:,t_slice2]#.copy()

            # _u is an abbreviation for undersampled
            t_u = np.hstack((t1,t2))
            bra_u = np.hstack((bra_in1,bra_in2))
            ket_u = np.hstack((ket_in1,ket_in2))
        else:
            bra_u = bra_dict_original['psi'][:,t_slice]#.copy()
            ket_u = ket_dict_original['psi'][:,t_slice]#.copy()
        t1 = time.time()
        self.slicing_time += t1-t0
        
        manifold1_num = bra_dict_original['manifold_num']
        manifold2_num = ket_dict_original['manifold_num']

        bra_nonzero = bra_dict_original['bool_mask']
        ket_nonzero = ket_dict_original['bool_mask']
        
        # exp_factor_bra = self.unitary[manifold1_num][bra_nonzero,t_slice]
        # exp_factor_ket = self.unitary[manifold2_num][ket_nonzero,t_slice]
        
        # bra_in *= exp_factor_bra
        # ket_in *= exp_factor_ket
        

        bra_dict = {'bool_mask':bra_nonzero,'manifold_num':manifold1_num,'psi':bra_u}
        ket_dict = {'bool_mask':ket_nonzero,'manifold_num':manifold2_num,'psi':ket_u}

        if np.abs(manifold1_num - manifold2_num) != 1:
            warnings.warn('Dipole only connects manifolds 0 to 1 or 1 to 2')
            return None
        t0 = time.time()
        if manifold1_num > manifold2_num:
            bra_new_mask = ket_dict['bool_mask']
            bra_dict = self.dipole_down(bra_dict,new_manifold_mask = bra_new_mask,
                                        pulse_number = pulse_number)
        else:
            ket_new_mask = bra_dict['bool_mask']
            ket_dict = self.dipole_down(ket_dict,new_manifold_mask = ket_new_mask,
                                        pulse_number = pulse_number)
        
        bra_u = bra_dict['psi']
        ket_u = ket_dict['psi']

        exp_val_u = np.sum(np.conjugate(bra_u) * ket_u,axis=0)
        t1 = time.time()
        self.expectation_time += t1-t0

        t0 = time.time()

        # Interpolate expectation value back onto the full t-grid
        if u != 1:
            exp_val_interp = scipy.interpolate.interp1d(t_u,exp_val_u,kind='cubic')
            exp_val = exp_val_interp(t)
        else:
            exp_val = exp_val_u
        # print(exp_val.size/exp_val_u.size)
        t1 = time.time()
        self.interpolation_time += t1-t0
        
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
        
        # exp_factor_bra = self.unitary[manifold1_num][bra_nonzero,t_slice]
        # exp_factor_ket = self.unitary[manifold2_num][ket_nonzero,t_slice]
        
        # bra_in *= exp_factor_bra
        # ket_in *= exp_factor_ket

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
                                local_oscillator_number = -1,undersample_factor = 1):
        """This function generates a frequency-resolved signal from a polarization field
           local_oscillator_number - usually the local oscillator will be the last pulse 
                                     in the list self.efields"""
        undersample_slice = slice(None,None,undersample_factor)
        P_of_t = P_of_t_in[undersample_slice]
        t = self.t[undersample_slice]
        dt = t[1] - t[0]
        pulse_time = self.pulse_times[local_oscillator_number]
        if self.gamma != 0:
            exp_factor = np.exp(-self.gamma * np.abs(t-pulse_time))
            P_of_t *= exp_factor
        if return_polarization:
            return P_of_t

        pulse_time_ind = np.argmin(np.abs(self.t - pulse_time))
        efield = np.zeros(self.t.size,dtype='complex')

        if self.efield_t.size == 1:
            # Impulsive limit
            efield[pulse_time_ind] = self.efields[local_oscillator_number]
            efield = fftshift(ifft(ifftshift(efield)))*efield.size/np.sqrt(2*np.pi)
        else:
            pulse_start_ind = pulse_time_ind - self.size//2
            pulse_end_ind = pulse_time_ind + self.size//2 + self.size%2

            t_slice = slice(pulse_start_ind, pulse_end_ind,None)
            
            efield[t_slice] = self.efields[local_oscillator_number]
            efield = fftshift(ifft(ifftshift(efield)))*self.t.size*(self.t[1]-self.t[0])/np.sqrt(2*np.pi)

        # if P_of_t.size%2:
        #     P_of_t = P_of_t[:-1]
        #     t = t[:-1]
        halfway = self.w.size//2
        pm = self.w.size//(2*undersample_factor)
        efield_min_ind = halfway - pm
        efield_max_ind = halfway + pm + self.w.size%2
        efield = efield[efield_min_ind:efield_max_ind]

        P_of_w = fftshift(ifft(ifftshift(P_of_t)))*len(P_of_t)*dt/np.sqrt(2*np.pi)

        signal = np.imag(P_of_w * np.conjugate(efield))
        return signal

    def polarization_to_integrated_signal(self,P_of_t,*,
                                          local_oscillator_number = -1):
        """This function generates a frequency-integrated signal from a polarization field
local_oscillator_number - usually the local oscillator will be the last pulse in the list self.efields
"""
        pulse_time = self.pulse_times[local_oscillator_number]
        pulse_time_ind = np.argmin(np.abs(self.t - pulse_time))

        pulse_start_ind = pulse_time_ind - self.size//2
        pulse_end_ind = pulse_time_ind + self.size//2 + self.size%2

        t_slice = slice(pulse_start_ind, pulse_end_ind,1)
        t = self.t[t_slice]
        P_of_t = P_of_t[t_slice]
        if self.gamma != 0:
            exp_factor = np.exp(-self.gamma * np.abs(t-pulse_time))
            P_of_t *= exp_factor

        if local_oscillator_number == 'impulsive':
            signal = P_of_t[self.size//2]
        else:
            efield = self.efields[local_oscillator_number]
            signal = np.trapz(np.conjugate(efield)*P_of_t,x=t)
        
        return np.imag(signal)
