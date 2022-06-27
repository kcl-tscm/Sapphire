import numpy as np
import ray
import scipy.special
from flare.env import AtomicEnvironment
from numba import njit


def spherical_conversion(xyz: np.array, jacobian: bool = False) -> np.array:
    """
    Parameters
    ----------
    xyz: array_like
       npoints, 3 (x, y, z)
    jacobian: bool
       if enabled return also the jacobian

    Return
    ------
    np.array: r, theta, phi as in phisical convention

    if jacobian return also:

    np.array: npoints, 3,3
        dr/dx  dtheta/dx dphi/dx
    J = dr/dy  dtheta/dy dphi/dy
        dr/dz  dtheta/dz dphi/dz
    """
    ptsnew = np.zeros(xyz.shape)
    # Adding the term to prevent numerical instability
    # when all points lie on a plane (?)
    xy = xyz[:, 0]**2 + xyz[:, 1]**2 + 1e-8
    ptsnew[:, 0] = np.sqrt(xy + xyz[:, 2]**2)
    # for elevation angle defined from Z-axis down
    ptsnew[:, 1] = np.arctan2(np.sqrt(xy), xyz[:, 2])
    ptsnew[:, 2] = np.arctan2(xyz[:, 1], xyz[:, 0])
    # computation of the jacobian
    if jacobian:
        dr_dxyz = xyz / ptsnew[:, 0][:, np.newaxis]
        num = np.zeros(xyz.shape)
        num[:, 0] = xyz[:, 0] * xyz[:, 2]
        num[:, 1] = xyz[:, 1] * xyz[:, 2]
        num[:, 2] = -xy
        den = (xy + xyz[:, 2]**2) * np.sqrt(xy)
        dtheta_dxyz = num/den[:, np.newaxis]
        dphi_dxyz = np.zeros(xyz.shape)
        dphi_dxyz[:, 0] = -xyz[:, 1] / xy
        dphi_dxyz[:, 1] = xyz[:, 0] / xy
        return ptsnew, np.dstack([dr_dxyz, dtheta_dxyz, dphi_dxyz])

    else:
        return ptsnew, []


def compute_ds_es_coefficients(ns):
    # compute the e coefficient for each n
    # shape: ns
    es = np.arange(ns)
    es = es**2 * (es + 2)**2 / (4 * (es + 1)**4 + 1)

    # recursive creation of the d coefficient for each n
    # d[0] = 1 as in the paper
    # shape: ns
    ds = [1]
    for idx, e in enumerate(es[1:]):
        ds.append(1 - e / ds[idx])
    ds = np.array(ds)

    return es, ds


def compute_F1_coefficient(ns, radial_cutoff):
    # compute the coefficient of f_n for each n
    # shape: ns
    rad_base = np.arange(ns)
    F1_coeff = (-1)**rad_base * np.sqrt(2) * np.pi / radial_cutoff**(3 / 2)
    F1_coeff *= (rad_base + 1) * (rad_base + 2) / \
        np.sqrt((rad_base+1)**2 + (rad_base+2)**2)

    return F1_coeff


def compute_F2_coefficient(ls):
    # Note: this can be optimized reusing by using part
    # of the above cell but does not have much sense for
    # now
    # first we need to compute the coefficients for
    # the symmetric (-m...m) part
    m_ = (np.arange(-ls + 1, ls, dtype=np.float))
    # [-m_max, ......., m_max]
    l_ = np.arange(ls, dtype=np.float)
    # [0, ......., l_max]
    # shape: 2ls-1, ls
    # m on row (reverse orderign), l on column
    # question, here m is m as in the paper or is |m|
    # as it is for the mirror coefficient?
    num = l_[np.newaxis, :] - m_[:, np.newaxis]
    den = l_[np.newaxis, :] + m_[:, np.newaxis]

    # useles values
    mask_top = np.tri(ls - 1, ls)[:, ::-1]
    mask_bottom = np.tri(ls, ls)[::-1, ::-1]
    mask = np.vstack([mask_top, mask_bottom])
    num *= mask
    den *= mask
    # computing the F2 coefficients
    num = scipy.special.factorial(num)
    den = scipy.special.factorial(den)
    pre_factor = (2 * l_ + 1) / 2
    pre_factor = np.tile(pre_factor[np.newaxis, :], (2 * ls - 1, 1))

    # Proposed (Claudio):
    # shape: 2ls - 1, ls
    F2_coeff = (pre_factor * num / den)**0.5

    return F2_coeff


def generate_mirror_coefficient_Legendre(ls):
    # ==construction of P and dP needed for F2 ==
    # first we need to compute the coefficients for
    # the symmetric (-m...-1) part
    m_ = (np.arange(ls, dtype=np.float))[:0:-1]
    # [m_max, ......., 1]
    l_ = np.arange(ls, dtype=np.float)
    # [0, ......., l_max]
    # shape: ls - 1, ls
    # m on row (reverse orderign), l on column
    num = l_[np.newaxis, :] - m_[:, np.newaxis]
    den = l_[np.newaxis, :] + m_[:, np.newaxis]

    # removing negative and useles values
    mask = np.tri(ls - 1, ls)[:, ::-1]
    num *= mask
    den *= mask

    m1 = (-1)**m_
    m1 = np.tile(m1[:, np.newaxis], ls)
    # shape: ls - 1, ls
    mirror_coeff = m1 * \
        scipy.special.factorial(num)/scipy.special.factorial(den) * mask

    return mirror_coeff


def precompute_coefficients(ns, ls, radial_cutoff):
    es, ds = compute_ds_es_coefficients(ns)
    F1_coeff = compute_F1_coefficient(ns, radial_cutoff)
    F2_coeff = compute_F2_coefficient(ls)
    mirror_coeff = generate_mirror_coefficient_Legendre(ls)

    return es, ds, F1_coeff, F2_coeff, mirror_coeff


def precompute_Chebyshev_polynomials(ns):
    p = [np.zeros_like, np.ones_like]

    for k in range(2, ns+1):
        coeff = np.zeros(k)
        coeff[-1] = 1
        poly = np.polynomial.chebyshev.Chebyshev(coeff)
        p.append(poly)
    return p


def compute_radial_term(rs, rad_base, es, ds, radial_cutoff,
                        F1_coeff, Tk, basis='bessel'):

    if basis == 'bessel':
        gs, dgs = radial_spherical_bessel(
            rs, rad_base, es, ds, radial_cutoff, F1_coeff)
    elif basis == 'chebyshev':
        gs, dgs = radial_chebyshev(
            rs, rad_base, radial_cutoff, Tk)
    elif basis == 'scaled_chebyshev':
        gs, dgs = radial_scaled_chebyshev(
            rs, rad_base, radial_cutoff, Tk)
    else:
        print("Basis name not recognized. Defaulting to 'bessel'")
        gs, dgs = radial_spherical_bessel(
            rs, rad_base, es, ds, radial_cutoff, F1_coeff)

    return gs, dgs


def radial_scaled_chebyshev(rs, rad_base, radial_cutoff, Tk, lam=5):
    x1 = np.exp(-lam*(rs/radial_cutoff - 1))
    x2 = np.exp(lam) - 1
    x = 1 - 2*(x1 - 1)/x2
    dx = 2*lam/radial_cutoff/x2*x1
    g2 = 1 + np.cos(np.pi*rs/radial_cutoff)
    dg2 = -np.sin(np.pi*rs/radial_cutoff)*np.pi/radial_cutoff

    gs, dgs = [], []
    for n in rad_base+1:
        if n == 1:
            gs.append(0.5*g2)
            dgs.append(0.5*dg2)
        else:
            g1 = Tk[n](x)
            gs.append(0.25*(1-g1)*g2)
            dgs.append(0.25*(1-g1)*dg2 - 0.25*g2*dx*Tk[n].deriv(1)(x))

    # shapes: ns, nat_incutoff
    gs = np.array(gs)
    dgs = np.array(dgs)

    return gs, dgs


def radial_chebyshev(rs, rad_base, radial_cutoff, Tk):
    x = 2*rs/radial_cutoff - 1
    dx = 2/radial_cutoff
    g2 = 1 + np.cos(np.pi*rs/radial_cutoff)
    dg2 = -np.sin(np.pi*rs/radial_cutoff)*np.pi/radial_cutoff

    gs, dgs = [], []
    for n in rad_base+1:
        if n == 1:
            gs.append(0.5*g2)
            dgs.append(0.5*dg2)
        else:
            g1 = Tk[n](x)
            gs.append(0.25*(1-g1)*g2)
            dgs.append(0.25*(1-g1)*dg2 - 0.25*g2*dx*Tk[n].deriv(1)(x))

    # shapes: ns, nat_incutoff
    gs = np.array(gs)
    dgs = np.array(dgs)

    return gs, dgs


def radial_spherical_bessel(rs, rad_base, es, ds, radial_cutoff, F1_coeff):
    # ==construction of gs and dgs, F1 = g ==
    # notation is consistent with the paper and support material

    # support quantities
    # shapes: nat_incutoff, ns
    a = rs[:, np.newaxis] *\
        (rad_base[np.newaxis, :] + 1) * np.pi / radial_cutoff
    b = rs[:, np.newaxis] *\
        (rad_base[np.newaxis, :] + 2) * np.pi / radial_cutoff

    # N.B.! We use sin(x)/x and not np.sinc(x) because np.sinc(x)
    # is the NORMALIZED sinc function. but we want the unnormalized one
    F1_sinc = np.sin(a) / a + np.sin(b) / b

    # Proposed (Claudio):
    dF1_sinc_a = (np.cos(a) - np.sin(a) / a) / rs[:, np.newaxis]
    dF1_sinc_b = (np.cos(b) - np.sin(b) / b) / rs[:, np.newaxis]

    gs, dgs = [], []

    for n in rad_base:
        # shapes: nat_incutoff
        f_n = F1_coeff[n] * F1_sinc[:, n]
        df_n = F1_coeff[n] * (dF1_sinc_a[:, n] + dF1_sinc_b[:, n])
        if n == 0:
            gs.append(f_n)
            dgs.append(df_n)
        else:
            tmp = np.sqrt(es[n] / ds[n - 1]) * gs[n - 1]
            dtmp = np.sqrt(es[n] / ds[n - 1]) * dgs[n - 1]
            gs.append((f_n + tmp) / np.sqrt(ds[n]))
            dgs.append((df_n + dtmp) / np.sqrt(ds[n]))

    # shapes: ns, nat_incutoff
    gs = np.array(gs)
    dgs = np.array(dgs)

    return gs, dgs


def get_a_coeff(env, ns, ls, coefficients, specie,
                compute_dgvect, basis='bessel'):

    es, ds, F1_coeff, F2_coeff, mirror_coeff, Tk = coefficients

    xyz = env.bond_array_2[:, 0][:, None]*env.bond_array_2[:, 1:]
    mask = env.etypes == specie
    # computing spherical coordinates
    centered_position_cutoff_spherical, jac = \
        spherical_conversion(xyz[mask], compute_dgvect)

    # extract meaningful quantities
    rs = centered_position_cutoff_spherical[:, 0]
    thetas = centered_position_cutoff_spherical[:, 1]
    phis = centered_position_cutoff_spherical[:, 2]

    rad_base = np.arange(ns)

    gs, dgs = compute_radial_term(rs, rad_base, es, ds, env.cutoffs['twobody'],
                                  F1_coeff, Tk, basis)
    # finally P, F2 and dP, now is easy.
    # lpmn does not support arrays, only scalars
    # so we must loop here.
    # But it compute the derivatives.
    # Note: this code mirror a matrix,
    # np.arange(9).reshape(3,3)[:0:-1]
    # 0 exclude the mirror axis

    Ps = np.empty((len(thetas), 2 * ls - 1, ls))
    dPs = np.empty((len(thetas), 2 * ls - 1, ls))

    cos_thetas = np.cos(thetas)
    for i, cos_theta in enumerate(cos_thetas):
        P, dP = scipy.special.lpmn(ls - 1, ls - 1, cos_theta)

        P_full = np.zeros([2 * ls - 1, ls])
        P_full[ls - 1:, :] = P
        P_full[:ls - 1, :] = mirror_coeff * P[:0:-1]

        dP_full = np.zeros([2 * ls - 1, ls])
        dP_full[ls - 1:, :] = dP
        dP_full[:ls - 1, :] = mirror_coeff * dP[:0:-1]
        Ps[i] = P_full
        dPs[i] = dP_full

    # shape: nat_incutoff, 2ls -1, ls
    F2 = Ps * F2_coeff[np.newaxis, :, :]
    if compute_dgvect:
        dF2 = dPs * F2_coeff[np.newaxis, :, :] * \
            (-np.sin(thetas)[:, np.newaxis, np.newaxis])

    # compute F3
    ms = np.arange(-ls + 1, ls)
    # shape: nat_incutoff, 2ls - 1
    F3 = 1 / np.sqrt(2 * np.pi) * \
        np.exp(1j * phis[:, np.newaxis] * ms[np.newaxis, :])

    dF3 = 1 / np.sqrt(2 * np.pi) * \
        np.exp(1j * phis[:, np.newaxis] * ms[np.newaxis, :]) * \
        (1j * ms[np.newaxis, :])

    # Descriptor computation, a joke now
    # shapes recap
    # gs: ns, nat_incutoff
    # F2: nat_incutoff, 2ls-1, ls
    # F3: nat_incutoff, 2ls-1

    # support variables to make the code more readable
    # create new axis and transpose to achieve shape compatible with:
    # ns, ls, 2ls-1, nat_incutoff
    sup_gs = gs[:, np.newaxis, np.newaxis, :]
    sup_F2 = F2.T[np.newaxis, :, :, :]
    sup_F3 = F3.T[np.newaxis, np.newaxis, :, :]

    # shape: ns, ls, 2ls-1
    A_nlm = np.sum(sup_gs * sup_F2 * sup_F3, axis=-1)

    if compute_dgvect:
        # shape recap
        # dgs: ns, nat_incutoff
        # dF2: nat_incutoff, 2ls-1, ls
        # dF3: nat_incutoff, 2ls-1

        # support variables
        # create new axis and transpose to achieve shape compatible with:
        # ns, ls, 2ls-1, nat_incutoff
        sup_dgs = dgs[:, np.newaxis, np.newaxis, :]
        sup_dF2 = dF2.T[np.newaxis, :, :, :]
        sup_dF3 = dF3.T[np.newaxis, np.newaxis, :, :]

        dA_nlm_dr_q = sup_dgs * sup_F2 * sup_F3
        dA_nlm_dhteta_q = sup_gs * sup_dF2 * sup_F3
        dA_nlm_dphi_q = sup_gs * sup_F2 * sup_dF3

        # shape: ns, ls, 2ls-1, nat_incutoff, 3
        dA_nlm_dpolar = np.concatenate([dA_nlm_dr_q[:, :, :, :, np.newaxis],
                                        dA_nlm_dhteta_q[:, :,
                                                        :, :, np.newaxis],
                                        dA_nlm_dphi_q[:, :, :, :, np.newaxis]],
                                       axis=-1)

        # converting the derivative to caresian
        # jacobian shape: nat_in_cutoff, 3, 3
        # dp_nl_dpolar shape: ns, ls, 2ls-1, nat_incutoff, 3
        tiled = np.tile(dA_nlm_dpolar[:, :, :, :, np.newaxis, :], (3, 1))
        # shape: ns, ls, 2ls-1, nat_incutoff, 3
        dA_nlm_dxyz_relative = np.sum(tiled * jac, axis=-1)
        # abs(delta) with procedure 1 ~ 1.1e-16.

        # 0 is the central atom index
        # first k=i, q!=i
        # dAs has dimensions nat, ns, ls, 2ls-1, nat, 3
        dA_nlm_dxyz = np.zeros(
            (ns, ls, 2*ls-1, len(env.positions), 3), dtype='complex64')
        dA_nlm_dxyz[:, :, :, env.atom,
                    :] -= np.sum(dA_nlm_dxyz_relative, axis=3)
        for i, idx in enumerate(env.bond_inds[mask]):
            dA_nlm_dxyz[:, :, :, idx, :] += dA_nlm_dxyz_relative[:, :, :, i, :]
        return A_nlm, dA_nlm_dxyz
    else:
        return A_nlm, []


def get_ace_single_atom(env, ns, ls, coefficients, species,
                        compute_dgvect, basis='bessel'):

    As = np.zeros((len(species), len(species), ns,
                   ls, 2*ls-1), dtype='complex64')
    # Index_of_central_atom_species in the species vector
    j = np.where(species == env.species[env.atom])[0][0]

    if compute_dgvect:
        dAs = np.zeros((len(species), len(species), ns, ls, 2*ls-1,
                        len(env.positions), 3), dtype='complex64')

    for i, s in enumerate(species):
        if compute_dgvect:
            As[j, i, ...], dAs[j, i, ...] = get_a_coeff(
                env, ns, ls, coefficients, s, compute_dgvect, basis)
        else:
            As[j, i, ...], _ = get_a_coeff(
                env, ns, ls, coefficients, s, compute_dgvect, basis)

    if compute_dgvect:
        return As, dAs

    else:
        return As, [0]


def get_ace(structure, ns, ls, radial_cutoff, species,
            coefficients, compute_dgvect=True, basis='bessel'):
    # Precoumpute all factors that can be precomputed

    As = np.zeros([structure.nat, len(species), len(species), ns,
                   ls, ls*2-1], dtype=np.complex64)

    if compute_dgvect:
        dAs = np.zeros([structure.nat, len(species), len(species), ns,
                        ls, ls*2 - 1, structure.nat, 3], dtype=np.complex64)
    else:
        dAs = np.zeros(1, dtype=np.complex64)

    # Obtain local atomic cluster expansion and its derivatives
    for i in np.arange(structure.nat):
        env = AtomicEnvironment(structure, i, {'twobody': radial_cutoff})
        if compute_dgvect:
            As[i, ...], dAs[i, ...] = get_ace_single_atom(
                env, ns, ls, coefficients, species, compute_dgvect, basis)
        else:
            As[i, ...], _ = get_ace_single_atom(
                env, ns, ls, coefficients, species, compute_dgvect, basis)

    return As, dAs


class Descriptor():
    def __init__(self, rc, ns, ls, species, basis='bessel'):
        self.rc = rc
        self.ns = ns
        self.ls = ls
        self.species = species  # List containin atomic numbers
        self.number_of_species = len(species)
        self.basis = basis
        self.species_dict = {key: value for (value, key) in enumerate(species)}

    def compute(self, atoms, compute_dgvect=False, ncores=1):
        if type(atoms) == list:
            if ncores == 1:
                G, dG = [], []
                for x in atoms:
                    g, dg = self.compute_single_core(x, compute_dgvect)
                    G.append(g)
                    if compute_dgvect:
                        dG.append(dg)
            else:
                G, dG = self.compute_multi_core(atoms, compute_dgvect, ncores)

            if compute_dgvect:
                for i in np.arange(len(G)):
                    G[i], dG[i] = self.add_one_body_term(G[i], dG[i], atoms[i],
                                                         True)
            else:
                for i in np.arange(len(G)):
                    G[i], _ = self.add_one_body_term(G[i], None, atoms[i],
                                                     False)

        else:
            if ncores > 1:
                print("Using more than one core for a single structure will not \
improve performance. \nSwitching to single core.")
            G, dG = self.compute_single_core(atoms, compute_dgvect)
            if compute_dgvect:
                G, dG = self.add_one_body_term(G, dG, atoms, compute_dgvect)
            else:
                G, _ = self.add_one_body_term(G, None, atoms, compute_dgvect)

        return G, dG

    def compute_single_core(self, atoms, compute_dgvect=False):
        # Overridden by descriptor-specific compute_ function
        G, dG = [], []
        return G, dG

    def compute_multi_core(self, atoms, compute_dgvect=False, ncores=2):
        # Overridden by descriptor-specific compute_ function
        G, dG = [], []
        return G, dG

    def add_one_body_term(self, g, dg, atoms, compute_dgvect):
        one_body_term = np.zeros((g.shape[0], self.number_of_species))
        for k, v in self.species_dict.items():
            matching_species = atoms.coded_species == k
            one_body_term[matching_species, v] += 1
        # g = np.append(g, np.array(one_body_term, dtype='float32'), axis=-1)
        g_ = np.copy(g)
        g_[:, -self.number_of_species:] = one_body_term
        # if compute_dgvect:
        #     dg = np.append(dg, np.zeros((dg.shape[0], self.number_of_species,
        #                                  dg.shape[2], 3), dtype='float32'),
        #                    axis=1)
        return g_, dg


class Descr3(Descriptor):
    """ B2 (3-body) Atomic Cluster Expansion descriptor

    Parameters
    ----------
      compute_dgvect : boolean
      sparse_dgvect : boolean
      species : string of comma separated values
        sequence of species eg: C,H,N,O
      pbc_directions: optional
        pbc directions to override those in the json

    Notes
    -----
    """

    def __init__(self, rc, ns, ls, species, basis='bessel'):
        Descriptor.__init__(self, rc, ns, ls, species, basis)
        self.gsize_partial = int(self.number_of_species**2 *
                                 (self.ns * (self.ns + 1))/2 * self.ls)
        self.gsize = int(self.gsize_partial + self.number_of_species)
        es, ds, F1_coeff, F2_coeff, mirror_coeff = precompute_coefficients(
            ns, ls, rc)
        Tk = precompute_Chebyshev_polynomials(ns)
        self.coefficients = [es, ds, F1_coeff, F2_coeff, mirror_coeff, Tk]

    def compute_single_core(self, structure,  compute_dgvect=True):
        # define shorter names:
        radial_cutoff = self.rc
        species = self.species
        ns = self.ns
        ls = self.ls
        coefficients = self.coefficients

        # Obtain the local atomic cluster expansions
        As, dAs = get_ace(structure, ns, ls, radial_cutoff,
                          species, coefficients, compute_dgvect, self.basis)

        # Use local atomic cluster expansion and its derivatives
        # to compute B2 descriptor and its derivatives
        parity = (-1)**np.linspace(-ls+1, ls-1, 2*ls-1)
        Gs, dGs = get_B2_from_ace(parity, As, dAs, compute_dgvect)
        del As, dAs

        # # Compute indexes to use when reducing G and dG because of symmetry
        # r_ind, c_ind = np.triu_indices(ns)
        # # Reduce size of G because of symmetry
        # Gs = Gs[:, :, :, r_ind, c_ind, ...]
        # Gs = np.reshape(Gs, (structure.nat, self.gsize_partial))

        # if compute_dgvect:
        #     # Reduce size of dG because of symmetry
        #     dGs = dGs[:, :, :, r_ind, c_ind, ...]
        #     dGs = np.reshape(
        #         dGs, (structure.nat, self.gsize_partial, structure.nat, 3))
        return Gs, dGs

    def compute_env(self, env, compute_dgvect=True):
        # Make sure the environment cutoff matches the
        # one set for the descritptor
        env.cutoffs = {'twobody': self.rc}
        # define shorter names:
        species = self.species
        ns = self.ns
        ls = self.ls
        coefficients = self.coefficients
        # Compute indexes to use when reducing G and dG because of symmetry
        # r_ind, c_ind = np.triu_indices(ns)

        # Obtain the local atomic cluster expansions
        As, dAs = get_ace_single_atom(
            env, ns, ls, coefficients, species, compute_dgvect)

        # Use local atomic cluster expansion and its derivatives to
        # compute B2 descriptor and its derivatives
        parity = (-1)**np.linspace(-ls+1, ls-1, 2*ls-1)
        Gs, dGs = get_B2_from_ace_single_atom(parity, As, dAs, compute_dgvect)
        del As, dAs
        return Gs, dGs

        # # Reduce size of G because of symmetry
        # Gs = Gs[:, :, r_ind, c_ind, :]
        # Gs = np.reshape(Gs, (self.gsize_partial))

        # if compute_dgvect:
        #     # Reduce size of dG because of symmetry
        #     dGs = dGs[:, :, r_ind, c_ind, ...]
        #     dGs = np.reshape(
        #         dGs, (self.gsize_partial, len(env.positions), 3))

    def compute_multi_core(self, structures, compute_dgvect=True, ncores=4):
        ray.init(num_cpus=ncores)
        data_ref = ray.put(structures)
        g_, dg_ = [], []
        for i in np.arange(len(structures)):
            g__, dg__ = compute_multicore_helper_b2.remote(
                self.rc, self.ns, self.ls, self.species,
                self.coefficients, self.basis, self.gsize_partial,
                i, data_ref, compute_dgvect)
            g_.append(g__)
            dg_.append(dg__)

        g = ray.get(g_)
        if compute_dgvect:
            dg = ray.get(dg_)
        else:
            dg = None
        ray.shutdown()
        return g, dg


@ray.remote(num_returns=2)
def compute_multicore_helper_b2(radial_cutoff, ns, ls, species, coefficients,
                                basis, gsize, i, data_ref, compute_dgvect):
    structure = data_ref[i]
    # Obtain the local atomic cluster expansions
    As, dAs = get_ace(structure, ns, ls, radial_cutoff,
                      species, coefficients, compute_dgvect, basis)

    # Use local atomic cluster expansion and its derivatives
    # to compute B2 descriptor and its derivatives
    parity = (-1)**np.linspace(-ls+1, ls-1, 2*ls-1)
    Gs, dGs = get_B2_from_ace(parity, As, dAs, compute_dgvect)
    del As, dAs
    return Gs, dGs

    # # Compute indexes to use when reducing G and dG because of symmetry
    # r_ind, c_ind = np.triu_indices(ns)
    # # Reduce size of G because of symmetry
    # Gs = Gs[:, :, :, r_ind, c_ind, ...]
    # Gs = np.reshape(Gs, (structure.nat, gsize))

    # if compute_dgvect:
    #     # Reduce size of dG because of symmetry
    #     dGs = dGs[:, :, :, r_ind, c_ind, ...]
    #     dGs = np.reshape(
    #         dGs, (structure.nat, gsize, structure.nat, 3))

# def get_B2_from_ace(As, dAs=None, compute_dgvect=False):
#     """ This is following the implementation from Drautz_2019.
#     An implementation using complex conjugate following the
#     Spherical Bessel paper is equivalent but ~20% slower.
#     """
#     ls = As.shape[4]
#     parity = (-1)**np.linspace(-ls+1, ls-1, 2*ls-1)
#     # Shape is (nat, nsp, nsp, ns, ns, ls, 2ls-1) and we sum over 2ls-1
#     B_nnl = np.sum(parity*As[:, :, :, :, None, :, :] *
#                    As[:, :, :, None, :, :, ::-1], axis=-1).real
#     B_nnl = np.array(B_nnl, dtype='float32')

#     if compute_dgvect:
#         # Shape is (nat, nsp, nsp, ns, ns, ls, 2ls-1, nat, 3)
#         # and we sum over 2ls-1
#         dB_nnl = (np.sum(parity[None, None, None, None,
#                                 None, None, :, None, None] *
#                          dAs[:, :, :, :, None, ...] *
#                          As[:, :, :, None, :, :, ::-1, None, None],
#                          axis=-3).real +
#                   np.sum(parity[None, None, None, None,
#                                 None, None, :, None, None]
#                   * As[:, :, :, :, None, :, :, None, None] *
#                   dAs[:, :, :, None, :, :, ::-1,  ...], axis=-3).real)
#         dB_nnl = np.array(dB_nnl, dtype='float32')
#         del As, dAs, parity
#         return B_nnl, dB_nnl
#     else:
#         del As, parity
#         return B_nnl, []


# @njit(fastmath=False)
def get_B2_from_ace(parity, As, dAs=None, compute_dgvect=False):
    """ This is following the implementation from Drautz_2019.
    An implementation using complex conjugate following the
    Spherical Bessel paper is equivalent but ~20% slower.
    """
    ls = As.shape[4]
    ns = As.shape[3]
    nat = As.shape[0]
    nsp = As.shape[1]
    r_ind, c_ind = np.triu_indices(ns)
    le = ls*nsp*nsp
    B_nnl = np.zeros((nat, le*len(r_ind)+nsp), dtype=np.float64)
    if compute_dgvect:
        dB_nnl = np.zeros((nat, le*len(r_ind)+nsp, nat, 3), dtype=np.float64)
    else:
        dB_nnl = np.zeros(1)
    for i in np.arange(nat):
        j = 0
        for r, c in zip(r_ind, c_ind):
            B_nnl[i, j*le:(j+1)*le] = np.ravel(np.sum(As[i, :, :, r, :, :]*As[i, :, :, c, :, ::-1]*parity, axis=3).real)
            j += 1
    if compute_dgvect:
        for i in np.arange(nat):
            j = 0
            for r, c in zip(r_ind, c_ind):
                for k in np.arange(nat):
                    for s in np.arange(3):
                        dB_nnl[i, j*le:(j+1)*le, k, s] = np.ravel(np.sum(dAs[i, :, :, r, :, :, k, s] * As[i, :, :, c, :, ::-1] * parity, axis=3).real)
                        dB_nnl[i, j*le:(j+1)*le, k, s] += np.ravel(np.sum(As[i, :, :, r, :, :] * dAs[i, :, :, c, :, ::-1, k, s] * parity, axis=3).real)
                j += 1

    return B_nnl, dB_nnl


# @njit(fastmath=False)
def get_B2_from_ace_single_atom(parity, As, dAs=None, compute_dgvect=False):
    ls = As.shape[3]
    ns = As.shape[2]
    nsp = As.shape[0]
    r_ind, c_ind = np.triu_indices(ns)
    le = ls*nsp*nsp
    B_nnl = np.zeros((le*len(r_ind)+nsp), dtype=np.float64)
    j = 0
    for r, c in zip(r_ind, c_ind):
        B_nnl[j*le:(j+1)*le] = np.ravel(np.sum(As[:, :, r, :, :]*As[:, :, c, :, ::-1]*parity, axis = -1).real)
        j += 1

    if compute_dgvect:
        nat = dAs.shape[-2]
        dB_nnl = np.zeros((le*len(r_ind)+nsp, nat, 3), dtype=np.float64)
        j = 0
        for r, c in zip(r_ind, c_ind):
            for k in np.arange(nat):
                for s in np.arange(3):
                    dB_nnl[j*le:(j+1)*le, k, s] = np.ravel(np.sum(dAs[:, :, r, :, :, k, s] * As[:, :, c, :, ::-1]*parity, axis=-1).real)
                    dB_nnl[j*le:(j+1)*le, k, s] += np.ravel(np.sum(dAs[:, :, c, :, ::-1, k, s] * As[:, :, r, :, :]*parity, axis=-1).real)    
            j += 1
    return B_nnl, dB_nnl


class Descr25(Descriptor):
    """ Spherical Bessel descriptor (2.5-body)

    Parameters
    ----------
      compute_dgvect : boolean
      sparse_dgvect : boolean
      species : string of comma separated values
        sequence of species eg: C,H,N,O
      pbc_directions: optional
        pbc directions to override those in the json

    Notes
    -----
    """

    def __init__(self, rc, ns, ls, species, basis='bessel'):
        Descriptor.__init__(self, rc, ns, ls, species, basis)
        self.gsize_partial = int(self.number_of_species *
                                 self.number_of_species * self.ns * self.ls)
        self.gsize = int(self.gsize_partial + self.number_of_species)
        es, ds, F1_coeff, F2_coeff, mirror_coeff = precompute_coefficients(
            ns, ls, rc)
        Tk = precompute_Chebyshev_polynomials(ns)
        self.coefficients = [es, ds, F1_coeff, F2_coeff, mirror_coeff, Tk]

    def compute_single_core(self, structure,  compute_dgvect=True):

        # define shorter names:
        radial_cutoff = self.rc
        species = self.species
        ns = self.ns
        ls = self.ls
        coefficients = self.coefficients

        # Obtain the local atomic cluster expansions
        As, dAs = get_ace(structure, ns, ls, radial_cutoff,
                          species, coefficients, compute_dgvect, self.basis)

        # Use local atomic cluster expansion and its derivatives
        #  to compute B2 descriptor and its derivatives
        Gs, dGs = get_SB_from_ace(As, dAs, compute_dgvect)
        del As, dAs

        # Reduce size of G because of symmetry
        Gs = np.reshape(Gs, (structure.nat, self.gsize_partial))

        if compute_dgvect:
            dGs = np.reshape(
                dGs, (structure.nat, self.gsize_partial, structure.nat, 3))

        return Gs, dGs

    def compute_multi_core(self, structures, compute_dgvect=True, ncores=4):
        ray.init(num_cpus=ncores)
        data_ref = ray.put(structures)
        g_, dg_ = [], []
        for i in np.arange(len(structures)):
            g__, dg__ = compute_multicore_helper_sb.remote(
                self.rc, self.ns, self.ls, self.species,
                self.coefficients, self.basis, self.gsize_partial,
                i, data_ref, compute_dgvect)
            g_.append(g__)
            dg_.append(dg__)

        g = ray.get(g_)
        if compute_dgvect:
            dg = ray.get(dg_)
        else:
            dg = None
        ray.shutdown()
        return g, dg


@ray.remote(num_returns=2)
def compute_multicore_helper_sb(radial_cutoff, ns, ls, species, coefficients,
                                basis, gsize, i, data_ref, compute_dgvect):
    structure = data_ref[i]
    # Obtain the local atomic cluster expansions
    As, dAs = get_ace(structure, ns, ls, radial_cutoff,
                      species, coefficients, compute_dgvect, basis)

    # Use local atomic cluster expansion and its derivatives
    #  to compute B2 descriptor and its derivatives
    Gs, dGs = get_SB_from_ace(As, dAs, compute_dgvect)
    del As, dAs

    # Reduce size of G because of symmetry
    Gs = np.reshape(Gs, (structure.nat, gsize))

    if compute_dgvect:
        dGs = np.reshape(
            dGs, (structure.nat, gsize, structure.nat, 3))

    return Gs, dGs


def compute_env(self, env, compute_dgvect=True):
    # Make sure the environment cutoff matches the
    #  one set for the descritptor
    env.cutoffs = {'twobody': self.rc}
    # define shorter names:
    species = self.species
    ns = self.ns
    ls = self.ls
    coefficients = self.coefficients

    # Obtain the local atomic cluster expansions
    As, dAs = get_ace_single_atom(
        env, ns, ls, coefficients, species, compute_dgvect)

    # Use local atomic cluster expansion and its derivatives
    #  to compute B2 descriptor and its derivatives
    Gs, dGs = get_SB_from_ace_single_atom(As, dAs, compute_dgvect)
    del As, dAs

    Gs = np.reshape(Gs, (self.gsize_partial))

    if compute_dgvect:

        dGs = np.reshape(
            dGs, (self.gsize_partial, len(env.positions), 3))

    return Gs, dGs


def get_SB_from_ace(As, dAs=None, compute_dgvect=False):
    """ This is following the implementation of B2,
    but only taking elements where n = n'.
    """
    ls = As.shape[4]
    parity = (-1)**np.linspace(-ls+1, ls-1, 2*ls-1)
    # Shape is (nat, nsp, nsp, ns, ls, 2ls-1) and we sum over 2ls-1
    B_nl = np.sum(parity*As*As[..., ::-1], axis=-1).real
    if compute_dgvect:
        # Shape is (nat, nsp, nsp, ns, ls, 2ls-1, nat, 3) and we sum over 2ls-1
        dB_nl = (
                 np.sum(parity[None, None, None, None, None, :, None, None] *
                        dAs*As[..., ::-1, None, None], axis=-3).real +
                 np.sum(parity[None, None, None, None, None, :, None, None] *
                        As[..., None, None]*dAs[..., ::-1, :, :],
                        axis=-3).real)
        del As, dAs, parity
        return B_nl, dB_nl
    else:
        del As, parity
        return B_nl, []


def get_SB_from_ace_single_atom(As, dAs=None, compute_dgvect=False):
    ls = As.shape[2]
    parity = (-1)**np.linspace(-ls+1, ls-1, 2*ls-1)
    # Shape is (nsp, ns, ns, ls, 2ls-1) and we sum over 2ls-1
    B_nl = np.sum(parity*As*As[..., ::-1], axis=-1).real
    if compute_dgvect:
        # Shape is (nsp, ns, ls, 2ls-1, nat, 3) and we sum over 2ls-1
        dB_nl = (np.sum(parity[None, None, None, :, None, None] *
                 dAs*As[..., ::-1, None, None], axis=-3).real +
                 np.sum(parity[None, None, None, :, None, None] *
                 As[..., None, None]*dAs[..., ::-1, :, :], axis=-3).real)
        del As, dAs, parity
        return B_nl, dB_nl
    else:
        del As, parity
        return B_nl, []


class Descr23(Descriptor):
    """ Gvector contiaing explicit 2 and 3-body factors.

    Parameters
    ----------
      compute_dgvect : boolean
      sparse_dgvect : boolean
      species : string of comma separated values
        sequence of species eg: C,H,N,O
      pbc_directions: optional
        pbc directions to override those in the json

    Notes
    -----
    """

    def __init__(self, rc, ns2, ns3, ls, species, basis='bessel'):
        Descriptor.__init__(self, rc, ns3, ls, species, basis)
        self.ns2 = ns2
        self.ns3 = ns3
        self.gsize_3 = int(self.number_of_species**2 *
                           (self.ns3 * (self.ns3 + 1))/2 * self.ls)
        self.gsize_2 = int(self.number_of_species**2 * self.ns2)
        self.gsize_partial = self.gsize_2 + self.gsize_3
        self.gsize = int(self.gsize_partial + self.number_of_species)
        es, ds, F1_coeff, F2_coeff, mirror_coeff = precompute_coefficients(
            ns3, ls, rc)
        Tk = precompute_Chebyshev_polynomials(ns3)
        self.coefficients_3 = [es, ds, F1_coeff, F2_coeff, mirror_coeff, Tk]

        es, ds, F1_coeff, F2_coeff, mirror_coeff = precompute_coefficients(
            ns2, 1, rc)
        Tk = precompute_Chebyshev_polynomials(ns2)
        self.coefficients_2 = [es, ds, F1_coeff, F2_coeff, mirror_coeff, Tk]

    def compute_single_core(self, structure, compute_dgvect=True):

        # define shorter names:
        radial_cutoff = self.rc
        species = self.species
        ns3 = self.ns3
        ls = self.ls
        ns2 = self.ns2
        coefficients_3 = self.coefficients_3
        coefficients_2 = self.coefficients_2

        # Compute indexes to use when reducing G and dG because of symmetry
        r_ind, c_ind = np.triu_indices(ns3)

        # Obtain the local atomic cluster expansions
        As3, dAs3 = get_ace(structure, ns3, ls, radial_cutoff,
                            species, coefficients_3, compute_dgvect,
                            self.basis)

        # Use local atomic cluster expansion and its derivatives
        #  to compute B2 descriptor and its derivatives
        Gs3, dGs3 = get_B2_from_ace(As3, dAs3, compute_dgvect)
        del As3, dAs3
        # Reduce size of G because of symmetry
        Gs3 = Gs3[:, :, :, r_ind, c_ind, ...]
        Gs3 = np.reshape(Gs3, (structure.nat, self.gsize_3))

        # Obtain the local atomic cluster expansions
        As2, dAs2 = get_ace(structure, ns2, 1, radial_cutoff,
                            species, coefficients_2, compute_dgvect,
                            self.basis)

        Gs2 = As2[:, :, :, :, 0, 0].real
        if compute_dgvect:
            dGs2 = dAs2[:, :, :, :, 0, 0, :, :].real
        del As2, dAs2
        Gs2 = np.reshape(Gs2, (structure.nat, self.gsize_2))
        Gs = np.concatenate((Gs2, Gs3), axis=1)
        del Gs2, Gs3

        if compute_dgvect:
            dGs3 = dGs3[:, :, :, r_ind, c_ind, ...]
            dGs3 = np.reshape(
                dGs3, (structure.nat, self.gsize_3, structure.nat, 3))
            dGs2 = np.reshape(
                dGs2, (structure.nat, self.gsize_2, structure.nat, 3))
            dGs = np.concatenate((dGs2, dGs3), axis=1)
            del dGs2, dGs3
        else:
            dGs = []
        return Gs, dGs

    def compute_multi_core(self, structures, compute_dgvect=True, ncores=4):
        ray.init(num_cpus=ncores)
        data_ref = ray.put(structures)
        g_, dg_ = [], []
        for x in structures:
            g__, dg__ = compute_multicore_helper_mix.remote(
                self.rc, self.ns2, self.ns3, self.ls, self.species,
                self.coefficients_2, self.coefficients_3, self.basis,
                self.gsize_2, self.gsize_3, data_ref, i, compute_dgvect)
            g_.append(g__)
            dg_.append(dg__)

        g = ray.get(g_)
        if compute_dgvect:
            dg = ray.get(dg_)
        else:
            dg = []
        ray.shutdown()
        return g, dg


@ray.remote(num_returns=2)
def compute_multicore_helper_mix(radial_cutoff, ns2, ns3, ls, species,
                                 coefficients_2, coefficients_3, basis,
                                 gsize_2, gsize_3, data_ref, i,
                                 compute_dgvect):
    structure = data_ref[i]
    # Obtain the local atomic cluster expansions
    As3, dAs3 = get_ace(structure, ns3, ls, radial_cutoff,
                        species, coefficients_3, compute_dgvect, basis)

    # Use local atomic cluster expansion and its derivatives
    #  to compute B2 descriptor and its derivatives
    Gs3, dGs3 = get_B2_from_ace(As3, dAs3, compute_dgvect)
    del As3, dAs3

    # Compute indexes to use when reducing G and dG because of symmetry
    r_ind, c_ind = np.triu_indices(ns3)
    # Reduce size of G because of symmetry
    Gs3 = Gs3[:, :, :, r_ind, c_ind, ...]
    Gs3 = np.reshape(Gs3, (structure.nat, gsize_3))

    # Obtain the local atomic cluster expansions
    As2, dAs2 = get_ace(structure, ns2, 1, radial_cutoff,
                        species, coefficients_2, compute_dgvect, basis)
    Gs2 = As2[:, :, :, :, 0, 0].real
    if compute_dgvect:
        dGs2 = dAs2[:, :, :, :, 0, 0, :, :].real
    del As2, dAs2

    Gs2 = np.reshape(Gs2, (structure.nat, gsize_2))
    Gs = np.concatenate((Gs2, Gs3), axis=1)
    del Gs2, Gs3

    if compute_dgvect:
        dGs3 = dGs3[:, :, :, r_ind, c_ind, ...]
        dGs3 = np.reshape(
            dGs3, (structure.nat, gsize_3, structure.nat, 3))
        dGs2 = np.reshape(
            dGs2, (structure.nat, gsize_2, structure.nat, 3))
        dGs = np.concatenate((dGs2, dGs3), axis=1)
        del dGs2, dGs3
    else:
        dGs = []

    return Gs, dGs
