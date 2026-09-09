Kostogryz et al. (2026) specific intensities
=============================================

The ``kostogryz2026`` library provides the angle-dependent specific
intensities :math:`I_\lambda(\mu)` from `Kostogryz et al. (2026)
<https://arxiv.org/abs/2606.21912>`_. They were synthesized with MPS-ATLAS
from 3D radiative-MHD MURaM simulations. Unlike the ordinary 1D
:doc:`mps_atlas` grids, these are not disk-integrated flux spectra.

Loading a model
---------------

Load all native disk positions for one stellar model and magnetic state:

.. code-block:: python

   from speclib import SpecificIntensityGrid

   grid = SpecificIntensityGrid.from_library(
       "kostogryz2026",
       model="G2",
       metallicity=0.0,
       magnetic_state="ssd",
   )

   print(grid.mu)
   spec = grid.at_mu(0.5)

``grid.mu`` contains the ten native values from 0.1 to 1.0 in steps of 0.1,
where :math:`\mu=\cos\theta`. ``grid.at_mu`` accepts only these values. No
interpolation is performed.

Available models
----------------

.. list-table:: Kostogryz model selection
   :header-rows: 1
   :widths: 20 30 25

   * - ``model``
     - ``metallicity`` ([M/H])
     - :math:`\log g` (cgs)
   * - ``F3``
     - 0.0
     - 4.0
   * - ``G2``
     - -1.0, 0.0, +0.5
     - 4.438
   * - ``K0``
     - 0.0
     - 4.4
   * - ``K4``
     - 0.0
     - 4.609
   * - ``M0``
     - 0.0
     - 4.826
   * - ``M2``
     - 0.0
     - 5.0
   * - ``M4``
     - 0.0
     - 5.0

Each model has five magnetic states: ``hydro``, ``ssd``, ``B100G``,
``B200G``, and ``B300G``. ``hydro`` is non-magnetic and ``ssd`` is the
small-scale-dynamo simulation. The last three labels give the imposed vertical
field in the simulation setup; they are not local surface-field strengths.

Data properties
---------------

The spectra cover 200.50003--9980.0014 nm at 978 nonuniform vacuum
wavelengths. They use low-resolution ODF sampling, approximately
:math:`R\sim400` in the visible, rather than a constant-resolution grid.
Specific intensities are returned in
``erg / (s cm2 sr Angstrom)`` with wavelengths in Å.

``meta["teff_spectrum"]`` is the effective temperature derived from the
synthetic spectrum for the selected magnetic state. It is distinct from the
effective temperature derived directly from the MURaM simulation. The returned
metadata use [M/H] = +0.5 for the metal-rich G2 model and
:math:`\log g=4.4` for K0, following the paper's model definitions.

Models are downloaded automatically when first requested and cached under the
normal ``speclib`` library location. See :doc:`../installation` to change the
cache location or prefetch data.

Citation
--------

Please cite `Kostogryz et al. (2026)
<https://arxiv.org/abs/2606.21912>`_ and the `Edmond V1.0 data release
<https://doi.org/10.17617/3.FBTIYY>`_ when using these spectra.
