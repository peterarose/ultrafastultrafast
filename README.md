# ultrafastultrafast (UF2)
Code for simulating nonlinear optical spectroscopies of closed systems

## Taking UF2 for a test drive
To try this package without installing or downloading the repository,
follow this link to see an example jupyter notebook using Google's
Colaboratory:  
https://colab.research.google.com/github/peterarose/ultrafastultrafast/blob/master/UF2_Colab.ipynb
(Note: Google's Coloaboratory gives a warning message about running Jupyter
notebooks not authored by Google. When prompted by the warning, select
"RUN ANYWAY", and then click "YES" when it asks you if you would like to
reset all runtimes)

You should be able to run the whole notebook in about 20 seconds.  The final
plot produced at the bottom is Figure 6a from our paper (JCP:
https://doi.org/10.1063/1.5094062), without the Gaussian lineshape function
(only homogeneous broadening is included in this example).  There are
currently also a few other minor discrepancies: color-scale has been
rescaled, axes are in different units, and y-axis has the optical
carrier frequency rotated away.

## How to install
You can install UF2 without downloading the source code by running  
pip install ultrafastultrafast  

(Note that UF2 is only written for python 3, so you may need to run
pip3 install ultrafastultrafast if pip points to python 2 on your machine)  

If you would like to install from the source code, you can clone this
repository, navigate the repository directory, and run either  
python setup.py install  
or
pip install .

Once installed, you should be able to use  
import ultrafastultrafast as uf2  
Note: you should not need to install this code in order to run any of the
Jupyter notebooks included with this repository. This notebook can be used to
get an idea of what this code does.

## Dependencies
numpy  
matplotlib  
pyfftw  
scipy  
pyyaml

## How to Use
To take UF2 for a test run without cloning this repository or installing
it on your system, you can follow the above link to Google Colaboraty, or
click on UF2_Colab.ipynb on the github page, and then click on the link
"Open in Colab" at the top of the document.  You should be able to run
the entire Jupyter notebook in about 20 seconds.  It produces the
isotropically averaged TA spectra shown in Fig 6a of our paper, without
the Gaussian linewidth

See the Jupyter notebook UF2_examples.ipynb for examples of how to use
this code to generate perturbative wavepackets, and from there
the desired nonlinear spectroscopic signal.

See the Jupyter notebook RKE_examples.ipynb for exmples of how to use
the RK45-Euler method included with this code to generate
perturbative wavepackets, and from there the desired nonlinear
spectroscopic signal.  The API for both the UF2 algorithm and the
RKE algorithm is the same.

The folder example_folder includes the necessary files to simulate
a two-level system coupled to a single harmonic mode of Huang-Rhys
factor 0.4^2/2 = 0.08. (UF2 only - see Examples.ipynb)

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

