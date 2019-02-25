.. ultrafastultrafast documentation master file, created by
   sphinx-quickstart on Sat Feb 23 00:41:52 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ultrafastultrafast's documentation!
==============================================
This code is based upon the algorithm Ultrafast Ultrafast
(:math:`\text{UF}^2`), detailed in http://arxiv.org/abs/1902.07854.
:math:`\text{UF}^2` calculates the perturbative wavepackets needed to
calculate n-wave mixing signals.  It works with closed systems described
by the Hamiltonian :math:`H_0`, and calculates perturbations due to
ultrafast optical pulses in the dipole approximation, so that the total
Hamiltonian is

.. math::
   H = H_0 - \boldsymbol{\mu} \cdot \boldsymbol{E}(t)

where :math:`\boldsymbol{\mu}` is the dipole operator.  Bold symbols represent
cartesian vectors in the molecular frame, where for example

.. math::
    \boldsymbol{\mu} = \mu_x\boldsymbol{x} + \mu_y\boldsymbol{y} + \mu_z\boldsymbol{z}

In order to function, :math:`\text{UF}^2` requires that the eigenvalues
:math:`\hbar\omega_i` of :math:`H_0` are known

.. math::
   H_0|i\rangle = \hbar\omega_i|i\rangle

and that the dipole operator is known in the eigenbasis of :math:`H_0`

.. math::
   \boldsymbol{\mu}_{ij} = \langle i|\boldsymbol{\mu}|j\rangle

In order to use :math:`\text{UF}^2`, you must make a folder somewhere on
your computer, and place in that folder two files - eigenvalues.npz and
mu.npz.  Both files must be numpy archives created with numpy's savez
function.  The contents of the files are as follows

eigenvalues.npz contains up to three 1d numpy arrays with keys:

- 'GSM' - sorted list of eigenvalues in the ground state manifold
- 'SEM' - sorted list of eigenvalues in the singly excited manifold  
- 'DEM' - (optional) sorted list of eigenvalues in the doubly excited manifold
  

(Note: 4-wave mixing signals, to lowest order in time dependent perturbation
theory (TDPT) can probe at most these three excitation manifolds.  So far
this code has only been written to support up to the doubly excited manifold.
It would be straightforward to extend functionality to higher excitation
manifolds (e.g., triply excited, etc.))

mu.npz contains up to two 3d numpy arrays with keys:

- 'GSM_to_SEM' - dipole operator connecting the ground and singly excited
  manifolds.  Indices are [i,j,k] where i are the indices of the singly
  excited manifold, j are the indices of the ground state manifold, and k is
  the cartesian index where :math:`k = {0,1,2}` corresponds to
  :math:`k={x,y,z}`
- 'SEM_to_DEM' - dipole operator connecting the singly and doubly excited
  manifolds.  Indices are [i,j,k] where i are the indices of the doubly
  excited manifold, j are the indices of the singly excited manifold, and k is
  the cartesian index where :math:`k = {0,1,2}` corresponds to
  :math:`k={x,y,z}`


Convolutions
------------

In the paper we show that an operator :math:`K_{j^{(*)}}` can be used to
iteratively calculate all of the perturbative wavepackets needed for a given
spectroscopic signal.  :math:`K_{j}` describes an optical excitation of the
ket by the jth pulse, whereas :math:`K_{j^*}` describes an optical
de-excitation of the ket by the jth pulse.  These operators are implemented
as methods 'up' and 'down', respectively, in the primary class of
:math:`\text{UF}^2`, which is documented here

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. automodule:: ultrafastultrafast
		
.. autoclass:: Wavepackets
   :members: up,down,next_order,dipole_expectation,polarization_to_signal

We show in our paper that :math:`K_{j^{(*)}}` is essentially a weighted sum
over dipole matrix elements, followed by a convolution with the heaviside
step function :math:`\theta`.  We implement this convolution using the FFT
and the convolution theorem.  In order to get maximum performance, we rely
on the FFTW algorithm.  The class that defines the convolution operation is
documented here

.. autoclass:: ultrafastultrafast.core.HeavisideConvolve
   :members:

Pruning Dipole Operator
-----------------------
This is an optional class which can be used to trim (or prune) the dipole
operator down to the fewest possible terms that are needed to resolve the
spectrosocopic signal.  Given a relative tolerance epsilon, this class will
save two new files to the folder being used: mu_pruned.npz and
mu_boolean.npz.  mu_pruned.npz has all unnecessary elements of the dipole
operator set to 0, and mu_boolean.npz contains 1's where the dipole
operator is non-zero, and 0's everywhere else.  This pruning is recommended
for systems where the dipole operator has many elements that are close to
0 and do not contribute to the signal appreciably.  If you use this class
to prune the dipole operator, keep in mind that your choice of epsilon is
a convergence parameter for the signals you are ultimately calculating.

.. autoclass:: ultrafastultrafast.DipolePruning
   :members:

Signals
-------
We include several exmaples of how to use :math:`\text{UF}^2` to calculate
n-wave mixing signals in the module signals.  For example, we provide a
class for calculating the Transient Absorption Signal documented here

.. autoclass:: ultrafastultrafast.signals.TransientAbsorption
   :members:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
