import json
from pathlib import Path

import numpy as np
from ase.io import read
from ase.io.lammpsrun import read_lammps_dump_text
from flare.struc import Structure
from flare.utils.element_coder import NumpyEncoder, Z_to_element


def extract_info(struc):
    f = np.array([x.forces for x in struc])
    e = np.array([x.energy for x in struc])

    return f, e


def reshape_forces(Y):
    # reshape training forces if needed
    if Y is not None:
        if len(Y.shape) != 1 or Y.shape[-1] != 3:
            # Y must have shape Nenv*3
            try:
                Y = Y.reshape((np.prod(Y.shape[:-1]) * 3))
            except TypeError:
                Y_ = []
                for a in Y:
                    Y_.extend(a)
                Y = np.array(Y_)
                Y = Y.reshape((np.prod(Y.shape[:-1]) * 3))

    return Y


def xyz_to_traj(infile, outfile=None,
                force_name='forces', energy_name='energy', index=':'):
    """
    Transform from xyz data format to the .json data
    format used in FLARE,
    which is used throughout the Raffy
    package for handling atomic environments.

    The FLARE package can be found here:
    https://github.com/mir-group/flare
    """

    if outfile is None:
        infile_path = Path(infile)
        outfile = str(infile_path.parent / str(infile_path.stem + ".json"))
    if infile.endswith("xyz"):
        ase_traj = read(infile, index=index)
    elif infile.endswith("dump"):
        with open(infile) as f:
            ase_traj = read_lammps_dump_text(f, index=index)
    for atoms in ase_traj:
        if not atoms.cell:
            atoms.set_cell([[200, 0, 0], [0, 200, 0], [0, 0, 200]])
            atoms.set_pbc([False, False, False])

    idx = np.arange(len(ase_traj))
    trajectory = []
    for i in idx:
        # forces are not imported with this method!
        struct = Structure.from_ase_atoms(ase_traj[i])
        try:
            struct.forces = ase_traj[i].arrays[force_name]
        except:
            pass
        try:
            struct.energy = ase_traj[i].info[energy_name]
        except:
            pass
        trajectory.append(struct.as_dict())

    with open(outfile, 'w') as fp:
        fp.write(
            '\n'.join(json.dumps(trajectory[i], cls=NumpyEncoder
                                 ) for i in np.arange(len(trajectory))))

    return trajectory


def load_structures(filename, force_name='forces', energy_name='energy',
                    index=':'):
    """
    """
    filename_path = Path(filename)
    if filename_path.suffix == ".json":
        structures = Structure.from_file(filename)
    elif filename_path.suffix == ".xyz" or filename_path.suffix == ".dump":
        _ = xyz_to_traj(filename, str(
            filename_path.parent) + "flare_structures.json",
            force_name=force_name, energy_name=energy_name, index=index)
        structures = Structure.from_file(str(
            filename_path.parent) + "flare_structures.json")
    else:
        print("""Data format not recognized.
Supported formats are .xyz and FLARE .json format""")

    return structures


def save_as_xyz(structures, outfile="trajectory.xyz",
                energy=True, labels=False):
    """
    """
    if energy:
        for i in np.arange(len(structures)):
            if i == 0:
                to_xyz(structures[i], write_file=outfile, append=False,
                       dft_energy=structures[i].energy,
                       dft_forces=structures[i].forces, labels=labels)
            else:
                to_xyz(structures[i], write_file=outfile, append=True,
                       dft_energy=structures[i].energy,
                       dft_forces=structures[i].forces, labels=labels)
    else:
        for i in np.arange(len(structures)):
            if i == 0:
                to_xyz(structures[i], write_file=outfile, append=False,
                       dft_forces=structures[i].forces, labels=labels)
            else:
                to_xyz(structures[i], write_file=outfile, append=True,
                       dft_forces=structures[i].forces, labels=labels)


def to_xyz(struct, extended_xyz: bool = True, print_stds: bool = False,
           print_forces: bool = False, print_max_stds: bool = False,
           print_energies: bool = False, predict_energy=None,
           dft_forces=None, dft_energy=None, timestep=-1,
           write_file: str = '', append: bool = False, labels=None) -> str:
    """
    Function taken from the FLARE python package by Vandermause et al. at:
    https://github.com/mir-group/flare

    Reference:

    Vandermause, J., Torrisi, S. B., Batzner, S., Xie, Y., Sun, L., Kolpak,
    A. M. & Kozinsky, B.
    On-the-fly active learning of interpretable Bayesian force fields for
    atomistic rare events. npj Comput Mater 6, 20 (2020).
    https://doi.org/10.1038/s41524-020-0283-z

    Convenience function which turns a structure into an extended .xyz
    file; useful for further input into visualization programs like VESTA
    or Ovito. Can be saved to an output file via write_file.

    :param print_stds: Print the stds associated with the structure.
    :param print_forces:
    :param extended_xyz:
    :param print_max_stds:
    :param write_file:
    :return:
    """
    species_list = [Z_to_element(x) for x in struct.coded_species]
    xyz_str = ''
    xyz_str += f'{len(struct.coded_species)} \n'

    # Add header line with info about lattice and properties if extended
    #  xyz option is called.
    if extended_xyz:
        cell = struct.cell

        xyz_str += f'Lattice="{cell[0,0]} {cell[0,1]} {cell[0,2]}'
        xyz_str += f' {cell[1,0]} {cell[1,1]} {cell[1,2]}'
        xyz_str += f' {cell[2,0]} {cell[2,1]} {cell[2,2]}"'
        if timestep > 0:
            xyz_str += f' Timestep={timestep}'
        if predict_energy:
            xyz_str += f' PE={predict_energy}'
        if dft_energy is not None:
            xyz_str += f' DFT_PE={dft_energy}'
        xyz_str += ' Properties=species:S:1:pos:R:3'

        if print_stds:
            xyz_str += ':stds:R:3'
            stds = struct.stds
        if print_forces:
            xyz_str += ':forces:R:3'
            forces = struct.forces
        if print_max_stds:
            xyz_str += ':max_std:R:1'
            stds = struct.stds
        if labels:
            xyz_str += ':tags:R:1'
            clustering_labels = struct.local_energy_stds
        if print_energies:
            if struct.local_energies is None:
                print_energies = False
            else:
                xyz_str += ':local_energy:R:1'
                local_energies = struct.local_energies
        if dft_forces is not None:
            xyz_str += ':dft_forces:R:3'
        xyz_str += '\n'
    else:
        xyz_str += '\n'

    for i, pos in enumerate(struct.positions):
        # Write positions
        xyz_str += f"{species_list[i]} {pos[0]} {pos[1]} {pos[2]}"

        # If extended XYZ: Add in extra information
        if print_stds and extended_xyz:
            xyz_str += f" {stds[i,0]} {stds[i,1]} {stds[i,2]}"
        if print_forces and extended_xyz:
            xyz_str += f" {forces[i,0]} {forces[i,1]} {forces[i,2]}"
        if print_energies and extended_xyz:
            xyz_str += f" {local_energies[i]}"
        if print_max_stds and extended_xyz:
            xyz_str += f" {np.max(stds[i,:])} "
        if labels and extended_xyz:
            xyz_str += f" {clustering_labels[i]} "
        if dft_forces is not None:
            xyz_str += f' {dft_forces[i, 0]} {dft_forces[i,1]} ' \
                f'{dft_forces[i, 2]}'
        if i < (len(struct.positions) - 1):
            xyz_str += '\n'

    # Write to file, optionally
    if write_file:
        if append:
            fmt = 'a'
        else:
            fmt = 'w'
        with open(write_file, fmt) as f:
            f.write(xyz_str)
            f.write("\n")

    return xyz_str
