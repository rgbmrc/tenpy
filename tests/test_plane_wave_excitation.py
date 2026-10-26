"""Environment sums of the plane-wave engines: ``GMRES-reg`` against the series and GMRES."""

import numpy as np
import pytest

import tenpy
from tenpy.algorithms.dmrg import TwoSiteDMRGEngine
from tenpy.algorithms.plane_wave_excitation import (
    MultiSitePlaneWaveExcitationEngine,
    PlaneWaveExcitationEngine,
)
from tenpy.models.tf_ising import TFIChain
from tenpy.networks.mps import MPS
from tenpy.networks.uniform_mps import UniformMPS

tenpy.tools.misc.skip_logging_setup = True


@pytest.fixture(scope='module', params=[None, 'parity'])
def gs(request):
    # truncated, so H|psi> keeps a small tangent-space residual: the p = 0 sums plateau
    M = TFIChain(dict(L=2, J=1.0, g=1.5, bc_MPS='infinite', conserve=request.param))
    psi = MPS.from_product_state(
        M.lat.mps_sites(), ['up', 'up'], bc='infinite', unit_cell_width=M.lat.mps_unit_cell_width
    )
    TwoSiteDMRGEngine(psi, M, {'trunc_params': {'chi_max': 8, 'svd_min': 1.0e-6}, 'max_sweeps': 30}).run()
    psi.canonical_form_infinite2()
    return M, UniformMPS.from_MPS(psi), request.param


def gap(M, upsi, size, method, p, qtotal_change):
    opts = {'sum_method': method, 'lanczos_params': {'N_max': 100}}
    if method == 'explicit':
        opts.update(sum_iterations=100, sum_tol=1.0e-15)  # >> correlation length
    else:
        opts['GMRES_params'] = {'N_max': 50, 'restart': 10, 'res': 1.0e-13}
    if size is None:
        eng = PlaneWaveExcitationEngine(upsi, M, opts)
    else:
        eng = MultiSitePlaneWaveExcitationEngine(upsi, M, dict(opts, excitation_size=size))
    E, _, _ = eng.run(p, qtotal_change=qtotal_change)
    return E[0], eng.sum_stats


@pytest.mark.slow
@pytest.mark.parametrize('size', [None, 1, 2, 3])
def test_GMRES_reg(gs, size):
    M, upsi, conserve = gs
    q = [1] if conserve else None  # a single spin flip, odd under parity
    # p = 0: the series plateaus along a direction that drops out, GMRES-reg must match it
    E_reg, stats = gap(M, upsi, size, 'GMRES-reg', 0.0, q)
    assert stats['failed'] == 0
    E_series, _ = gap(M, upsi, size, 'explicit', 0.0, q)
    assert abs(E_reg - E_series) < 1.0e-10
    # p != 0: 1 - T is invertible, GMRES-reg changes nothing
    E_reg, stats = gap(M, upsi, size, 'GMRES-reg', 0.7, q)
    assert stats['failed'] == 0
    E_gmres, _ = gap(M, upsi, size, 'GMRES', 0.7, q)
    assert abs(E_reg - E_gmres) < 1.0e-10
