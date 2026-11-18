"""Virtual bond terms ``H = H_MPO + sum_b f_b(Q_b)``, see :meth:`tenpy.networks.mpo.MPO.set_bond_terms`.

Q_b is the raw charge label of the MPS bond b (left of site b).
Reference values come from exact diagonalization (finite), from a closed-form sum over the Schmidt
spectrum (infinite), and from an exact equivalence of linear bond terms with onsite fields.
"""
# Copyright (C) TeNPy Developers, Apache license

import functools
import itertools
import os

# small matrices: BLAS threading only slows the tests down (effective if numpy is not yet imported)
for _var in ('MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import tenpy
from tenpy.algorithms.dmrg import SingleSiteDMRGEngine, TwoSiteDMRGEngine
from tenpy.algorithms.exact_diag import ExactDiag
from tenpy.algorithms.mps_common import TwoSiteH, ZeroSiteH
from tenpy.algorithms.plane_wave_excitation import (
    MultiSitePlaneWaveExcitationEngine,
    PlaneWaveExcitationEngine,
)
from tenpy.algorithms.tdvp import SingleSiteTDVPEngine, TwoSiteTDVPEngine
from tenpy.algorithms.tebd import TEBDEngine
from tenpy.algorithms.vumps import VUMPSEngine
from tenpy.linalg import np_conserved as npc
from tenpy.models.spins import SpinChain
from tenpy.networks.mpo import MPOEnvironment, MPOEnvironmentBuilder, MPOTransferMatrix
from tenpy.networks.mps import MPS
from tenpy.networks.uniform_mps import UniformMPS

tenpy.tools.misc.skip_logging_setup = True

# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def bond_f(q, g=0.0, h=0.0, c=0.0):
    """``f(Q) = g Q^2 + h Q + c`` for the single charge ``Q = 2 Sz``."""
    Q = np.asarray(q)[:, 0].astype(float)
    return g * Q**2 + h * Q + c


def g_of(b):
    return 0.2 + 0.1 * b  # b-dependent, positive (bounded below)


def make_bt(L, finite, h=0.15, c=0.0, none_at=()):
    bt = [functools.partial(bond_f, g=g_of(b), h=h, c=c) for b in range(L)]
    for b in none_at:
        bt[b] = None
    if finite:
        bt[0] = None
    return bt


def spin_model(L, bc, hz=0.0, bt=None, Jz=1.5):
    M = SpinChain(dict(L=L, S=0.5, Jx=1.0, Jy=1.0, Jz=Jz, hz=hz, bc_MPS=bc, conserve='Sz'))
    if bt is not None:
        M.H_MPO.set_bond_terms(bt)
    return M


def neel(M, bc, chargeL=None):
    L = M.lat.N_sites
    return MPS.from_product_state(
        M.lat.mps_sites(),
        [['up', 'down'][i % 2] for i in range(L)],
        bc=bc,
        chargeL=chargeL,
        unit_cell_width=M.lat.mps_unit_cell_width,
    )


def bond_charges(psi, b):
    """Raw charge labels (flat, 1 charge) and the probability of the bond b (left of site b)."""
    q = psi.get_B(b, form='B').get_leg('vL').to_qflat()[:, 0]
    S = psi.get_SL(b)
    return q, S**2 / np.sum(S**2)


def sum_f_per_cell(psi, bt):
    """``sum_b sum_a S_{b,a}^2 f_b(Q_{b,a})`` for an infinite psi, charges and weights read directly."""
    tot = 0.0
    for b, f in enumerate(bt):
        if f is not None:
            q, w = bond_charges(psi, b)
            tot += np.sum(w * f(q[:, None]))
    return tot


def idmrg(M, engine=TwoSiteDMRGEngine, chi=16, **kw):
    psi = neel(M, 'infinite')
    opts = {
        'trunc_params': {'chi_max': chi, 'svd_min': 1e-12},
        'max_sweeps': 200,
        'max_E_err': 1e-11,
        'max_S_err': 1e-7,
        'mixer': engine is SingleSiteDMRGEngine,
    }
    opts.update(kw)
    eng = engine(psi, M, opts)
    E, _ = eng.run()
    psi.canonical_form_infinite2()
    return E, psi


PW_OPTS = {
    'sum_method': 'GMRES-reg',
    'lanczos_params': {'N_max': 150},
    'GMRES_params': {'N_max': 80, 'restart': 20, 'res': 1e-13},
}
CHI_PW = 6  # plane-wave identities hold for any canonical state: keep the ground states small


def pw_engine(M, upsi, size=None, run=False, **kw):
    """Plane-wave engine (``size=None``: single-site engine); `run`: it will call ``run`` (else no Lanczos)."""
    opts = dict(PW_OPTS, **kw)
    if not run:
        opts.pop('lanczos_params')  # unused options warn (and warnings are errors)
    if size is None:
        return PlaneWaveExcitationEngine(upsi, M, opts)
    return MultiSitePlaneWaveExcitationEngine(upsi, M, dict(opts, excitation_size=size))


_PW_CACHE = {}


def pw_energy(M, upsi, p, size=None, tag=None):
    """The two lowest excitation energies and the energy density; cached if `tag` is given."""
    key = (tag, p, size)
    if tag is not None and key in _PW_CACHE:
        return _PW_CACHE[key]
    eng = pw_engine(M, upsi, size, run=True)
    E, _, _ = eng.run(p, qtotal_change=[0], num_ev=2)
    res = (np.real(E[:2]), float(eng.energy_density))
    if tag is not None:
        _PW_CACHE[key] = res
    return res


@pytest.fixture(scope='module')
def inf_nl():
    """Infinite L=2 with nonlinear bond terms; the ground state of ``H_MPO + sum_b f_b``."""
    L = 2
    bt = make_bt(L, False, none_at=())
    M = spin_model(L, 'infinite', bt=bt)
    E, psi = idmrg(M, chi=CHI_PW)
    return dict(L=L, bt=bt, M=M, E=E, psi=psi, upsi=UniformMPS.from_MPS(psi), M0=spin_model(L, 'infinite'))


@pytest.fixture(scope='module')
def inf_lin():
    """Infinite L=2: onsite fields ``hz``, and the equivalent linear bond terms on ``hz=0``.

    With H = -sum hz_b Sz_b = -sum_b hz_b (Q_{b+1} - Q_b)/2 = sum_b a_b Q_b, ``a_b = (hz_b - hz_{b-1})/2``.
    """
    L = 2
    hz = [0.4, -0.25]
    a = [(hz[b] - hz[(b - 1) % L]) / 2 for b in range(L)]
    bt = [functools.partial(bond_f, h=ab) for ab in a]
    Mh = spin_model(L, 'infinite', hz=np.array(hz))
    Ml = spin_model(L, 'infinite', hz=0.0, bt=bt)
    E, psi = idmrg(Mh, chi=CHI_PW)
    return dict(L=L, Mh=Mh, Ml=Ml, E=E, psi=psi, upsi=UniformMPS.from_MPS(psi))


# ---------------------------------------------------------------------------------------------
# 1. API
# ---------------------------------------------------------------------------------------------


def test_mpo_api():
    L = 4
    M = spin_model(L, 'finite')
    H = M.H_MPO
    assert not H.has_bond_terms
    assert H.get_bond_term(2) is None
    with pytest.raises(ValueError):
        H.set_bond_terms(make_bt(L, False))  # finite: bt[0] must be None
    with pytest.raises(ValueError):
        H.set_bond_terms(make_bt(L - 1, True))  # wrong length
    with pytest.raises(ValueError):
        H.set_bond_terms([None, None, 1.0, None])  # not callable
    assert not H.has_bond_terms
    H.set_bond_terms([None] * L)  # all None: no bond terms
    assert not H.has_bond_terms
    bt = make_bt(L, True, none_at=(2,))
    H.set_bond_terms(bt)
    assert H.has_bond_terms
    H.test_sanity()
    # outer bonds and None entries have no term, others return (f, IdL, IdR)
    assert H.get_bond_term(0) is None and H.get_bond_term(L) is None and H.get_bond_term(2) is None
    f, IdL, IdR = H.get_bond_term(3)
    assert f is bt[3] and IdL == H.get_IdL(3) and IdR == H.get_IdR(2)
    H.set_bond_terms(None)
    assert not H.has_bond_terms

    # explicit_plus_hc, in both orders
    Mh = spin_model(L, 'finite')
    Mh.H_MPO.explicit_plus_hc = True
    with pytest.raises(NotImplementedError):
        Mh.H_MPO.set_bond_terms(bt)
    Mh = spin_model(L, 'finite', bt=bt)
    Mh.H_MPO.explicit_plus_hc = True
    with pytest.raises(NotImplementedError):
        Mh.H_MPO.get_bond_term(3)

    # infinite: bond 0 allowed, wrap-around of get_bond_term
    Mi = spin_model(2, 'infinite', bt=make_bt(2, False))
    Hi = Mi.H_MPO
    assert Hi.get_bond_term(2)[0] is Hi.bond_terms[0] and Hi.get_bond_term(-1)[0] is Hi.bond_terms[1]
    # segment bc is not supported
    Hs = spin_model(L, 'finite').H_MPO
    Hs.bc = 'segment'
    with pytest.raises(NotImplementedError):
        Hs.set_bond_terms(bt)

    # constructor kwarg, dagger, enlarge_mps_unit_cell, is_equal
    Hb = spin_model(L, 'finite', bt=bt).H_MPO
    Hd = Hb.dagger()
    assert Hd.has_bond_terms and all(f is g for f, g in zip(Hd.bond_terms, Hb.bond_terms))
    assert Hb.is_equal(Hb)
    assert not Hb.is_equal(spin_model(L, 'finite').H_MPO)
    assert not Hb.is_equal(spin_model(L, 'finite', bt=make_bt(L, True)).H_MPO)  # other callables
    Hi2 = Hi.copy().enlarge_mps_unit_cell(2) if hasattr(Hi, 'copy') else None
    if Hi2 is None:
        import copy

        Hi2 = copy.copy(Hi)
        Hi2.enlarge_mps_unit_cell(2)
    assert Hi2.L == 4 and len(Hi2.bond_terms) == 4
    assert all(Hi2.bond_terms[b] is Hi.bond_terms[b % 2] for b in range(4))
    Hi2.test_sanity()


def test_guarded_methods():
    L = 4
    M = spin_model(L, 'finite', bt=make_bt(L, True))
    H = M.H_MPO
    psi = neel(M, 'finite')
    Mi = spin_model(2, 'infinite', bt=make_bt(2, False))
    Hi = Mi.H_MPO
    for name, call in [
        ('group_sites', lambda: H.group_sites(2)),
        ('extract_segment', lambda: Hi.extract_segment(0, 5)),
        ('make_U', lambda: H.make_U(0.1)),
        ('expectation_value_power', lambda: Hi.expectation_value_power(neel(Mi, 'infinite'))),
        ('variance', lambda: H.variance(psi)),
        ('to_TermList', lambda: H.to_TermList(None)),
        ('apply', lambda: H.apply(psi, {})),
        ('apply_naively', lambda: H.apply_naively(psi)),
        ('apply_zipup', lambda: H.apply_zipup(psi, {})),
        ('plus_identity', lambda: H.plus_identity(1.0, 1.0)),
        ('__add__', lambda: H + H),
        ('save_hdf5', lambda: H.save_hdf5(None, None, '')),
        ('MPOEnvironmentBuilder', lambda: MPOEnvironmentBuilder(H, psi)),
        ('calc_H_bond_from_MPO', lambda: M.calc_H_bond_from_MPO()),
        ('ExactDiag', lambda: ExactDiag(M).build_full_H_from_mpo()),
    ]:
        with pytest.raises(NotImplementedError):
            call()


def test_guards_engines():
    L = 4
    # TEBD silently ignores H_MPO/bond terms -> must raise; ED_all changes the bond charges
    Mf = spin_model(L, 'finite', bt=make_bt(L, True))
    psi = neel(Mf, 'finite')
    with pytest.raises(NotImplementedError):
        TEBDEngine(psi, Mf, {})
    for cls in (SingleSiteTDVPEngine, TwoSiteTDVPEngine):
        with pytest.raises(NotImplementedError):
            cls(psi, Mf, {})
    with pytest.raises(NotImplementedError):
        TwoSiteDMRGEngine(psi, Mf, {'diag_method': 'ED_all'})
    Mi = spin_model(2, 'infinite', bt=make_bt(2, False))
    with pytest.raises(NotImplementedError):
        VUMPSEngine(neel(Mi, 'infinite'), Mi, {})
    # absolute charges: a non-trivial qtotal on a tensor is rejected
    psi2 = neel(Mf, 'finite')
    psi2.get_B(1).qtotal = np.array([1])
    with pytest.raises(ValueError):
        MPOEnvironment(psi2, Mf.H_MPO, psi2)


def test_plane_wave_guards(inf_nl):
    upsi, M = inf_nl['upsi'], inf_nl['M']
    for eng in (
        PlaneWaveExcitationEngine(upsi, M, {}),
        MultiSitePlaneWaveExcitationEngine(upsi, M, {'excitation_size': 2}),
    ):
        with pytest.raises(NotImplementedError):
            eng.run(0.3, qtotal_change=[1])
        with pytest.raises(NotImplementedError):
            eng.initial_guess([2])
    # a state with a shifted qtotal cannot be used
    psi = inf_nl['psi'].copy()
    psi.get_B(0).qtotal = np.array([1])
    with pytest.raises(ValueError):
        MPOEnvironment(psi, M.H_MPO, psi)


# ---------------------------------------------------------------------------------------------
# 2. finite chain against exact diagonalization
# ---------------------------------------------------------------------------------------------


def ed_ground_energy(L, bt, chargeL, Qtot=0):
    """Dense H of the plain MPO plus the diagonal bond terms, lowest eigenvalue at total charge `Qtot`."""
    M0 = spin_model(L, 'finite')
    ED = ExactDiag(M0)
    ED.build_full_H_from_mpo()
    H = ED.full_H.to_ndarray()
    pipe = ED._pipe
    qsite = [s.leg.to_qflat()[:, 0] for s in M0.lat.mps_sites()]
    dim = 2**L
    Qcum = np.zeros((dim, L + 1), dtype=int)  # charge of bond b for each basis state
    Qcum[:, 0] = chargeL
    for conf in itertools.product(range(2), repeat=L):
        idx = pipe.map_incoming_flat(list(conf))
        Qcum[idx, 1:] = chargeL + np.cumsum([qsite[j][c] for j, c in enumerate(conf)])
    diag = np.zeros(dim)
    for b in range(1, L):
        if bt[b] is not None:
            diag += bt[b](Qcum[:, b : b + 1])
    sel = np.nonzero(Qcum[:, L] - chargeL == Qtot)[0]
    Hs = H[np.ix_(sel, sel)] + np.diag(diag[sel])
    return np.linalg.eigvalsh(Hs)[0], H, Qcum, diag


@pytest.mark.parametrize('chargeL', [0, 1])
def test_finite_vs_ED(chargeL):
    L = 8
    bt = make_bt(L, True, none_at=(5,))
    E_ref, _, _, _ = ed_ground_energy(L, bt, chargeL)
    E_plain = ed_ground_energy(L, [None] * L, chargeL)[0]
    assert abs(E_ref - E_plain) > 0.05  # the bond terms matter

    M = spin_model(L, 'finite', bt=bt)
    for engine, combine, diag_method in itertools.product(
        [TwoSiteDMRGEngine, SingleSiteDMRGEngine], [False, True], ['lanczos', 'ED_block']
    ):
        psi = neel(M, 'finite', chargeL=[chargeL])
        eng = engine(
            psi,
            M,
            {
                'trunc_params': {'chi_max': 40, 'svd_min': 1e-14},
                'max_sweeps': 40,
                'max_E_err': 1e-12,
                'max_S_err': 1e-8,
                'mixer': engine is SingleSiteDMRGEngine,
                'combine': combine,
                'diag_method': diag_method,
                **({'lanczos_params': {'N_min': 5, 'N_max': 40, 'E_tol': 1e-14}} if diag_method == 'lanczos' else {}),
            },
        )
        E, _ = eng.run()
        label = f'{engine.__name__} combine={combine} {diag_method} chargeL={chargeL}'
        assert abs(E - E_ref) < 1e-9, f'{label}: DMRG {E:.12f} vs ED {E_ref:.12f}'
        # the energy of the final state, from the MPO expectation value and from every site
        assert abs(M.H_MPO.expectation_value(psi) - E_ref) < 1e-8, label
        env = MPOEnvironment(psi, M.H_MPO, psi)
        for i in range(L):
            assert abs(env.full_contraction(i) - E_ref) < 1e-8, f'{label}: full_contraction({i})'


def test_product_state_exact_finite():
    L = 8
    bt = make_bt(L, True, none_at=(5,))
    for chargeL in [0, 1]:
        M = spin_model(L, 'finite', bt=bt)
        psi = neel(M, 'finite', chargeL=[chargeL])
        E_plain = spin_model(L, 'finite').H_MPO.expectation_value(psi)
        assert abs(E_plain - (-0.25 * 1.5 * (L - 1))) < 1e-12
        E_f = sum(bt[b](psi.get_B(b).get_leg('vL').to_qflat()[:1, :]) for b in range(L) if bt[b] is not None)[0]
        assert abs(M.H_MPO.expectation_value(psi) - (E_plain + E_f)) < 1e-12
        # ED diagonal element of the same basis state
        _, H, Qcum, diag = ed_ground_energy(L, bt, chargeL)
        ED = ExactDiag(spin_model(L, 'finite'))
        idx = ED._pipe.map_incoming_flat([[1, 0][i % 2] for i in range(L)])  # up = index 1 (charge +1)
        up = ED._pipe.legs[0].to_qflat()[:, 0]
        assert [up[1], up[0]] == [1, -1]
        assert abs(H[idx, idx] + diag[idx] - (E_plain + E_f)) < 1e-12


def test_product_state_exact_infinite():
    for L in [2, 4]:
        bt = make_bt(L, False)
        M = spin_model(L, 'infinite', bt=bt)
        psi = neel(M, 'infinite')
        e_plain = spin_model(L, 'infinite').H_MPO.expectation_value(psi)
        e_f = sum(bt[b](psi.get_B(b).get_leg('vL').to_qflat()[:1, :])[0] for b in range(L)) / L
        assert abs(M.H_MPO.expectation_value(psi) - (e_plain + e_f)) < 1e-10, L


# ---------------------------------------------------------------------------------------------
# 4./5. infinite systems: transfer matrix, environments, sum identity
# ---------------------------------------------------------------------------------------------


def test_TM_vs_env_growth(inf_nl):
    M, psi, L = inf_nl['M'], inf_nl['psi'], inf_nl['L']
    H = M.H_MPO
    env_data, e, _ = MPOTransferMatrix.find_init_LP_RP(H, psi, 0, L - 1, calc_E=True)
    e = np.mean(e)
    env = MPOEnvironment(psi, H, psi, **env_data)
    LW, RW = env_data['init_LP'], env_data['init_RP']
    IdL, IdR = H.get_IdL(0), H.get_IdR(L - 1)
    # one more cell to the right adds e*L*eye to the IdR channel and nothing else
    LP = LW
    for i in range(L):
        LP = env._contract_LP(i, LP)
    diff = LP.itranspose(['wR', 'vR*', 'vR']) - LW.itranspose(['wR', 'vR*', 'vR'])
    eyeL = npc.diag(1.0, LW.get_leg('vR*'), labels=['vR*', 'vR'])
    for w in range(diff.shape[0]):
        d = diff.take_slice(w, 'wR')
        err = npc.norm(d - e * L * eyeL) if w == IdR else npc.norm(d)  # (other channels are charged)
        assert err < 1e-8, f'LP channel {w}: |diff - e L 1| = {err:.2e}'
    # same for the right environment, grown to the left
    RP = RW
    for i in reversed(range(L)):
        RP = env._contract_RP(i, RP)
    diff = RP.itranspose(['wL', 'vL', 'vL*']) - RW.itranspose(['wL', 'vL', 'vL*'])
    eye = npc.diag(1.0, RW.get_leg('vL'), labels=['vL', 'vL*'])
    for w in range(diff.shape[0]):
        d = diff.take_slice(w, 'wL')
        err = npc.norm(d - e * L * eye) if w == IdL else npc.norm(d)
        assert err < 1e-8, f'RP channel {w}: |diff - e L 1| = {err:.2e}'
    # the default-built environment (forced to the TM) agrees up to the eye ambiguity of the IdR channel
    env2 = MPOEnvironment(psi, H, psi)
    LP2 = env2.get_LP(0).itranspose(['wR', 'vR*', 'vR'])
    d = LP2 - LW.itranspose(['wR', 'vR*', 'vR'])
    for w in range(d.shape[0]):
        dw = d.take_slice(w, 'wR')
        if w == IdR:
            dw = dw - (npc.trace(dw, 'vR*', 'vR') / eyeL.shape[0]) * eyeL
        assert npc.norm(dw) < 1e-7, f'default env LP channel {w}'
    # TM energy density is the engine's
    assert abs(e - H.expectation_value(psi)) < 1e-9


def test_infinite_sum_identity(inf_nl):
    """e(H + F) = e(H) + (1/L) sum_b sum_a S_{b,a}^2 f_b(Q_{b,a}), also for a state that is not the GS of either."""
    L, bt, psi = inf_nl['L'], inf_nl['bt'], inf_nl['psi']
    e_F = inf_nl['M'].H_MPO.expectation_value(psi)
    e_0 = inf_nl['M0'].H_MPO.expectation_value(psi)
    assert abs(e_F - (e_0 + sum_f_per_cell(psi, bt) / L)) < 1e-9
    # the charge distribution is genuinely spread (otherwise the test is weak)
    assert max(len(bond_charges(psi, b)[0]) for b in range(L)) >= 3
    # a different state: ground state of the plain model, and an L=4 model
    _, psi0 = idmrg(inf_nl['M0'], chi=10, max_sweeps=20, max_E_err=1e-6, max_S_err=1e-3)
    e_F = inf_nl['M'].H_MPO.expectation_value(psi0)
    e_0 = inf_nl['M0'].H_MPO.expectation_value(psi0)
    assert abs(e_F - (e_0 + sum_f_per_cell(psi0, bt) / L)) < 1e-9
    L4, bt4 = 4, make_bt(4, False, h=-0.3, c=0.2, none_at=(1,))
    M4, M40 = spin_model(L4, 'infinite', bt=bt4), spin_model(L4, 'infinite')
    _, psi4 = idmrg(M40, chi=8, max_sweeps=20, max_E_err=1e-6, max_S_err=1e-3)
    e_F = M4.H_MPO.expectation_value(psi4)
    e_0 = M40.H_MPO.expectation_value(psi4)
    assert abs(e_F - (e_0 + sum_f_per_cell(psi4, bt4) / L4)) < 1e-9


# ---------------------------------------------------------------------------------------------
# 6. iDMRG
# ---------------------------------------------------------------------------------------------


def test_idmrg(inf_nl):
    M, L, bt = inf_nl['M'], inf_nl['L'], inf_nl['bt']
    E2, psi2 = inf_nl['E'], inf_nl['psi']
    e_TM = M.H_MPO.expectation_value(psi2)
    assert abs(E2 - e_TM) < 1e-8, f'two-site iDMRG {E2:.12f} vs TM {e_TM:.12f}'
    # at fixed chi the two engines land on different truncated states (also without bond terms): compare at chi=12
    E2_12, _ = idmrg(M, chi=12)
    E1, psi1 = idmrg(M, engine=SingleSiteDMRGEngine, chi=12)
    assert abs(E1 - E2_12) < 1e-8, f'one-site {E1:.12f} vs two-site {E2_12:.12f}'
    assert abs(M.H_MPO.expectation_value(psi1) - E1) < 1e-8
    # a constant c shifts the energy by exactly c per site (same state, TM and iDMRG)
    c = 0.37
    Mc = spin_model(L, 'infinite', bt=make_bt(L, False, c=c))
    assert abs(Mc.H_MPO.expectation_value(psi2) - (e_TM + c)) < 1e-10
    Ec, _ = idmrg(Mc, chi=CHI_PW)
    assert abs(Ec - (E2 + c)) < 1e-8


# ---------------------------------------------------------------------------------------------
# 7. effective Hamiltonians
# ---------------------------------------------------------------------------------------------


def fully_split(T):
    while any(lbl.startswith('(') for lbl in T.get_leg_labels()):
        T = T.split_legs()
    return T


@pytest.mark.parametrize('combine', [False, True])
def test_effH_unit(combine):
    L = 8
    bt = make_bt(L, True)
    M = spin_model(L, 'finite', bt=bt)
    psi = neel(M, 'finite', chargeL=[1])
    TwoSiteDMRGEngine(
        psi, M, {'trunc_params': {'chi_max': 12, 'svd_min': 1e-10}, 'max_sweeps': 3}
    ).run()
    env = MPOEnvironment(psi, M.H_MPO, psi)
    E_tot = env.full_contraction(0)
    rng = np.random.default_rng(5)
    for i0 in [0, 3, 5, 6]:
        H2 = TwoSiteH(env, i0, combine=combine)
        theta = psi.get_theta(i0, n=2)
        thc = H2.combine_theta(theta)
        # <theta|H|theta> is the total energy
        e = npc.inner(thc, H2.matvec(thc), axes='labels', do_conj=True)
        assert abs(e - E_tot) < 1e-9, f'i0={i0}: <theta|H|theta> = {e:.10f}, total energy {E_tot:.10f}'
        # random neutral theta: to_matrix is Hermitian and agrees with matvec
        th = npc.Array.from_func(rng.standard_normal, theta.legs, qtotal=theta.qtotal, labels=theta.get_leg_labels())
        thc = H2.combine_theta(th)
        out = fully_split(H2.matvec(thc))
        mat = H2.to_matrix()
        dense = mat.to_ndarray()
        assert np.max(np.abs(dense - dense.conj().T)) < 1e-12, f'i0={i0}: to_matrix not Hermitian'
        out2 = npc.tensordot(
            fully_split(mat), th, axes=[['vR', 'p0*', 'p1*', 'vL'], ['vL', 'p0', 'p1', 'vR']]
        ).ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
        out2.itranspose(out.get_leg_labels())
        assert npc.norm(out - out2) < 1e-10, f'i0={i0}: to_matrix vs matvec {npc.norm(out - out2):.2e}'
        # Hermiticity of the matvec itself
        th2 = npc.Array.from_func(rng.standard_normal, theta.legs, qtotal=theta.qtotal, labels=theta.get_leg_labels())
        th2c = H2.combine_theta(th2)
        a = npc.inner(th2c, H2.matvec(thc), axes='labels', do_conj=True)
        b = npc.inner(H2.matvec(th2c), thc, axes='labels', do_conj=True)
        assert abs(a - b) < 1e-10
    # zero-site: the bond matrix S sees the total energy
    for i0 in [1, 4, 7]:
        H0 = ZeroSiteH(env, i0)
        S = psi.get_SL(i0)
        C = npc.diag(S, psi.get_B(i0).get_leg('vL'), labels=['vL', 'vR'])
        C = C.replace_labels(['vL', 'vR'], ['vL', 'vR'])
        e = npc.inner(C, H0.matvec(C), axes='labels', do_conj=True) / np.sum(S**2)
        assert abs(e - E_tot) < 1e-9, f'ZeroSiteH(i0={i0}): {e:.10f} vs {E_tot:.10f}'
        d0 = H0.to_matrix().to_ndarray()
        assert np.max(np.abs(d0 - d0.conj().T)) < 1e-12


# ---------------------------------------------------------------------------------------------
# 8./9. plane waves
# ---------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_plane_wave_linear_equals_onsite(inf_lin):
    """Linear bond terms are exactly onsite fields: energy, TM and plane-wave energies agree."""
    Mh, Ml, psi, upsi = inf_lin['Mh'], inf_lin['Ml'], inf_lin['psi'], inf_lin['upsi']
    assert abs(Mh.H_MPO.expectation_value(psi) - Ml.H_MPO.expectation_value(psi)) < 1e-10
    # iDMRG on the bond-term model from the same product state
    El, _ = idmrg(Ml, chi=CHI_PW)
    assert abs(El - inf_lin['E']) < 1e-9, f'iDMRG bond terms {El:.12f} vs onsite {inf_lin["E"]:.12f}'
    for p, sizes in [(0.0, [None, 2]), (1.3, [None, 1, 2])]:
        for size in sizes:
            Eh, eh = pw_energy(Mh, upsi, p, size)
            El, el = pw_energy(Ml, upsi, p, size)
            assert abs(eh - el) < 1e-10, f'e0 p={p} size={size}: {eh:.12f} vs {el:.12f}'
            assert np.max(np.abs(Eh - El)) < 1e-9, f'gaps p={p} size={size}: onsite {Eh} vs bond terms {El}'


@pytest.mark.slow
def test_plane_wave_constant_shift(inf_nl):
    """A constant f = c shifts the energy density by c and leaves all excitation energies unchanged."""
    M, upsi, L, c = inf_nl['M'], inf_nl['upsi'], inf_nl['L'], 0.43
    Mc = spin_model(L, 'infinite', bt=make_bt(L, False, c=c))
    for p, sizes in [(0.0, [None, 1, 2]), (0.7, [None, 1, 2, 3])]:
        for size in sizes:
            E, e = pw_energy(M, upsi, p, size, tag='nl')
            Ec, ec = pw_energy(Mc, upsi, p, size)
            assert abs(ec - (e + c)) < 1e-10, f'e0 shift p={p} size={size}: {ec - e:.12f} vs {c}'
            assert np.max(np.abs(Ec - E)) < 1e-9, f'gaps p={p} size={size}: {E} vs {Ec}'


@pytest.mark.slow
def test_plane_wave_nontrivial(inf_nl):
    M, upsi = inf_nl['M'], inf_nl['upsi']
    for p in [0.0, 0.7]:
        E1, _ = pw_energy(M, upsi, p, None, tag='nl')
        E1m, _ = pw_energy(M, upsi, p, 1, tag='nl')
        assert np.max(np.abs(E1 - E1m)) < 1e-9, f'p={p}: single-site {E1} vs multi-site size 1 {E1m}'
        E2, _ = pw_energy(M, upsi, p, 2, tag='nl')
        assert E1[0] > -1e-7 and E2[0] > -1e-7, f'p={p}: negative gaps {E1[0]}, {E2[0]}'
        assert E2[0] <= E1[0] + 1e-9, f'p={p}: size 2 {E2[0]:.12f} above size 1 {E1[0]:.12f}'


@pytest.fixture(scope='module')
def inf_nl4():
    """L=4, nonlinear bond terms with a None entry; a loosely converged state suffices."""
    L, bt = 4, make_bt(4, False, h=-0.2, none_at=(2,))
    M = spin_model(L, 'infinite', bt=bt)
    _, psi = idmrg(M, chi=CHI_PW, max_sweeps=30, max_E_err=1e-8, max_S_err=1e-4)
    return dict(L=L, bt=bt, M=M, psi=psi, upsi=UniformMPS.from_MPS(psi))


@pytest.mark.parametrize('which', ['nl', 'nl4'])
def test_plane_wave_padded_block(which, inf_nl, inf_nl4):
    """A size-s X = X1 AR..AR is the single-site plane wave X1: the energies agree for any p.

    Hermiticity cannot see bond terms put at wrong charges inside a multi-site block, this does.
    """
    d = inf_nl if which == 'nl' else inf_nl4
    M, upsi, L = d['M'], d['upsi'], d['L']
    e1 = pw_engine(M, upsi, None)
    X1 = [x / np.sqrt(L) for x in e1.initial_guess([0])]
    for p in [0.0, 0.7]:
        E1 = e1.energy(p, X1)
        for s in [2, 3]:
            Xs = []
            for i, x in enumerate(X1):
                for m in range(1, s):
                    x = npc.tensordot(x, upsi.get_AR((i + m) % L).replace_label('p', f'p{m}'), axes=(['vR'], ['vL']))
                Xs.append(x)
            Es = pw_engine(M, upsi, s).energy(p, Xs)
            assert abs(E1 - Es) < 1e-9, f'{which} p={p} size={s}: single-site {E1:.12f} vs padded {Es:.12f}'


def rand_X(eng, seed):
    eng.options['init_guess_seed'] = seed
    X1 = eng.initial_guess([0])
    eng.options['init_guess_seed'] = seed + 1000
    X2 = eng.initial_guess([0])
    return [x1 + 1j * x2 for x1, x2 in zip(X1, X2)]


def vinner(Y, X):
    return sum(npc.inner(y, x, axes='labels', do_conj=True) for y, x in zip(Y, X))


@pytest.mark.parametrize('size', [None, 1, 2])
@pytest.mark.parametrize('p', [0.0, 0.7])
def test_plane_wave_effH_hermitian(inf_nl, size, p):
    M, upsi = inf_nl['M'], inf_nl['upsi']
    eng = pw_engine(M, upsi, size, GMRES_params={'N_max': 80, 'restart': 20, 'res': 1e-14})
    if size is None:
        Heff = [eng.aligned_H, eng.Unaligned_Effective_H(eng, p)]
    else:
        Heff = [eng.Aligned_Effective_H(eng, p), eng.Unaligned_Effective_H(eng, p)]

    def apply(X):
        a, u = Heff[0].matvec(X), Heff[1].matvec(X)
        return [x + y for x, y in zip(a, u)]

    X, Y = rand_X(eng, 1), rand_X(eng, 2)
    HX, HY = apply(X), apply(Y)
    a, b = vinner(Y, HX), vinner(HY, X)
    scale = abs(vinner(X, HX)) + abs(vinner(Y, HY))
    assert abs(a - b) < 1e-9 * max(1.0, scale), f'size={size} p={p}: <Y|H X> = {a:.10f}, <H Y|X> = {b:.10f}'
    assert abs(vinner(X, HX).imag) < 1e-9 * max(1.0, scale)
