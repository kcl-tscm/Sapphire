from pathlib import Path

import numpy as np
from flare.env import AtomicEnvironment
from scipy.stats import gaussian_kde

from . import compute_descriptors as cd
from . import utils as ut


def compute_descriptor(structures, descr, ncores):
    """ Auxiliary function that calls the correct compute_descriptor
    function from the two available above (flare and ase).
    Args:
        structures (list): List of N structures
        descr (object): descriptor class from compute_descriptors

    Returns:
        Gs (np.array): Array of shape (N, M, S) S is the size of the descriptor
            and M the number of atoms in each snapshot

    """
    Gs, _ = descr.compute(structures, compute_dgvect=False, ncores=ncores)
    Gs = np.array(Gs)
    # Multiplying by 1000 because float 32
    # and low values hinder precision othersie
    Gs = np.reshape(Gs, (Gs.shape[0] * Gs.shape[1], Gs.shape[2]))
    return Gs


def cluster_gvect(G, k, z, clustering):
    """ Auxiliary function that calls the correct clustering
        algorithm. Options are: kmeans clustering and advanced
        density peaks clustering. If the latter is chosen, the
        adp python package must be installed first.
    Args:
        G (np.array): Descriptor vector for each atom in each structure
        k (float): number of clusters for clustering = 'kmeans'
        z (loat): nvalue of Z for clustering == 'adp'
        clustering (str): clustering algorithm to use, either "kmeans" or "adp"

    Returns:
        labels (np.array): cluster (int) assigned
                           to each atom in each structure

    """
    if clustering == 'kmeans':
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k).fit(G)
        labels = kmeans.labels_
    elif clustering == 'adp':
        try:
            from adpy import data
            adp = data.Data(G)
            adp.compute_distances(maxk=max(len(G) // 100, 100), njobs=2)
            adp.compute_id()
            adp.compute_optimal_k()
            adp.compute_density_kNN(int(np.median(adp.kstar)))
            print("Selected k is : %i" % (int(np.median(adp.kstar))))
            adp.compute_clustering_optimised(Z=z, halo=False)
            labels = adp.labels
        except ModuleNotFoundError:
            print("WARNING: ADP package required to perform adp clustering.\
                   Defaulting to kmeans clustering.")
            labels = cluster_gvect(G, k, z, "kmeans")
    return labels


def set_cutoff_radius(snapshot):
    """ Auxiliary function used to automatically select the cutoff radius
    for the descriptor by setting it to 1.75 times the first peak in the
    radial distribution function of the first frame of the trajectory.

    Args:
        snapshot (ASE atom object): snapshot used to calculate the p.d.f.

    Returns:
        cut (float): cutoff radius in Angstorm

    """
    dists = [AtomicEnvironment(snapshot, i, {
        'twobody': 3.5}).bond_array_2[:, 0]
        for i in range(snapshot.nat)]
    dists = np.array([item for sublist in dists for item in sublist])
    t = np.linspace(1.5, 3.5, 100)
    cut = t[np.argmax(gaussian_kde(dists)(t))]*1.75

    return cut


def trajectory_cluster(filename, k=6, z=1.65, ns=4, ls=4, cut=None,
                       index=':', clustering='kmeans', ncores=1):
    """ Function that reads a trajectory file, computes the G vector for each atom
    and each snapshot, then uses Kmeans to cluster data into k clusters.
    The clusters are stored in a xyz file in the 'tag' column.
    The G vectors are also saved in .npy format in order to speed
    up repeated clustering
    with different k.

    Args:
        filename (str): path to the trajectory file
        k (float): number of clusters for clustering = 'kmeans'
        z (loat): nvalue of Z for clustering == 'adp'
        ns (int): number of radial basis for the
                  local atomic environment descriptor.
        ls (int): number of angular basis for the
                  local atomic environment descriptor.
        cut (float): cutoff of the descriptor in Angstrom, if None it is
                     automatically chosen.
        index (str, int or slice): used to slice the trajectory
        clustering (str): clustering algorithm to use, either "kmeans" or "adp"
        ncores (int): how many cores to use for Gvector calculation

    """
    filename_path = Path(filename)
    data = ut.load_structures(filename, index=index)
    all_species = list(set(data[0].coded_species))

    if cut is None:
        cut = set_cutoff_radius(data[0])

    if clustering == 'kmeans':
        outfile_name = str(filename_path.parent / str(
            filename_path.stem + "_clustered_k=%i_cut=%.2f.xyz" % (k, cut)))
    elif clustering == 'adp':
        outfile_name = str(filename_path.parent / str(
            filename_path.stem + "_clustered_z=%.2f_cut=%.2f.xyz" % (z, cut)))
    else:
        print("WARNING: clustering type not understood.\
              Defaulting to 'kmeans'.")

    gvec_name = str(filename_path.parent / str(filename_path.stem
                                               + "_G_cut_%.2f.npy" % (cut)))

    descr = cd.Descr3(rc=cut, ns=ns, ls=ls,
                      species=all_species)

    try:
        G = np.load(gvec_name, allow_pickle=True)
        print("Imported G Vector")
    except FileNotFoundError:
        G = compute_descriptor(data, descr, ncores)

    labels = cluster_gvect(G, k, z, clustering)

    last = 0
    for i in np.arange(len(data)):
        nat = data[i].nat
        if i == 0:
            data[i].local_energy_stds = labels[0:nat]
        else:
            data[i].local_energy_stds = labels[last:last + nat]
        last += nat

    ut.save_as_xyz(data, outfile=outfile_name, labels=True)
    np.save(gvec_name, G)

    print("Finished. The Labeled trajectory file\
          can be found at %s" % (outfile_name))
