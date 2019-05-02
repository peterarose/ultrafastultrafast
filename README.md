# ultrafastultrafast (UF2)
Code for simulating nonlinear optical spectroscopies of closed systems

# How to install
In this directory, run either  
python setup.py install  
or
pip install .

Once installed, you should be able to use  
import ultrafastultrafast as uf2  
Note: you should not need to install this code in order to run the Jupyter
notebook Examples.ipynb. This notebook can be used to get an idea of what
this code does.

# Dependencies
numpy  
matplotlib  
pyfftw  
scipy

# How to Use
See the Jupyter notebook UF2_examples.ipynb for examples of how to use
this code to generate perturbative wavepackets, and from there
the desired nonlinear spectroscopic signal

See the Jupyter notebook RKE_examples.ipynb for exmples of how to use
the RK45-Euler method included with this code to generate
perturbative wavepackets, and from there the desired nonlinear
spectroscopic signal.  The API for both the UF2 algorithm and the
RKE algorithm is the same.

The folder example_folder includes the necessary files to simulate
a two-level system coupled to a single harmonic mode of Huang-Rhys
factor 0.4^2/2 = 0.08. (UF2 only)

The folder dimer_example includes the necessary parameters file,
called simple_params.yaml, to run both the UF2 and RKE algorithms.
Documentation describing how to edit and create simple_params.yaml
files to simulate other vibronic systems will be added soon.

# Simulating with your own system

To use UF2 to calculate spectra for other systems, you must create a
folder for the system.  You can use the Jupyter notebook Examples.ipynb
to calcualte the transient absorption signal for your system by simply
changing the file_path variable to specify the folder you created,and running
the notebook.  You can also use the class core.UF2 to write your own code to
calculate any n-wave mixing process.  See TA_example.py for a 4-wave mixing
example, and TA_5th_order_example.py for a 6-wave mixing example.

The folder describing the system parameters must have the following two files:

eigenvalues.npz - a numpy archive with the following keys:  
- 'GSM': containing all eigenvalues in the ground state manifold  
- 'SEM': containing all eigenvalues in the singly excited manifold  
- 'DEM' (optional): containing all eigenvalues in the doubly excited manifold  

mu.npz - a numpy archive with the following keys:  
- 'GSM_to_SEM': containing a 3d numpy array with indicies [i,j,k] of the dipole elements
connecting the GSM eigenstates (index j) to the SEM eigenstates (index i). The thid index
k = 0,1,2 corresonds to cartesian coordinates k = x,y,z
- 'SEM_to_DEM' (optional): containing a 3d numpy array with indicies [i,j,k] of the dipole elements
connecting the SEM eigenstates (index j) to the DEM eigenstates (index i). The thid index
k = 0,1,2 corresonds to cartesian coordinates k = x,y,z

Note: RKE is currently not compatible with systems other than those that can
be described by the vibronic_eigenstates package included in this
repository. That is not a fundamental limitation, but simply a limitation of
the current implementation.  Eventually the code will be updated to make it
compatible with your own Hamiltonians, just as UF2 is already.

# Documentation
To view documentation open the html file 'doc/build/html/index.html'