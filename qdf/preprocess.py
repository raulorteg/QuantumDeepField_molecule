#!/usr/bin/env python3

from collections import defaultdict
import os
import pickle
from pathlib import Path

import numpy as np
from scipy import spatial

from qdf.definitions import ATOMICNUMBER_DICT


def load_dict(filename: str):
    with open(filename, 'rb') as f:
        dict_load = pickle.load(f)
        dict_default = defaultdict(lambda: max(dict_load.values())+1)
        for k, v in dict_load.items():
            dict_default[k] = v
    return dict_default

def create_sphere(radius: float, grid_interval: float):
    """Create the sphere to be placed on each atom of a molecule."""
    xyz = np.arange(-radius, radius+1e-3, grid_interval)
    sphere = [[x, y, z] for x in xyz for y in xyz for z in xyz
              if (x**2 + y**2 + z**2 <= radius**2) and [x, y, z] != [0, 0, 0]]
    return np.array(sphere)


def create_field(sphere, coords):
    """Create the grid field of a molecule."""
    field = [f for coord in coords for f in sphere+coord]
    return np.array(field)


def create_orbitals(orbitals, orbital_dict):
    """Transform the atomic orbital types (e.g., H1s, C1s, N2s, and O2p)
    into the indices (e.g., H1s=0, C1s=1, N2s=2, and O2p=3) using orbital_dict.
    """
    orbitals = [orbital_dict[o] for o in orbitals]
    return np.array(orbitals)


def create_distancematrix(coords1, coords2):
    """Create the distance matrix from coords1 and coords2,
    where coords = [[x_1, y_1, z_1], [x_2, y_2, z_2], ...].
    For example, when coords1 is field_coords and coords2 is atomic_coords
    of a molecule, each element of the matrix is the distance
    between a field point and an atomic position in the molecule.
    Note that we transform all 0 elements in the distance matrix
    into a large value (e.g., 1e6) because we use the Gaussian:
    exp(-d^2), where d is the distance, and exp(-1e6^2) becomes 0.
    """
    distance_matrix = spatial.distance_matrix(coords1, coords2)
    return np.where(distance_matrix == 0.0, 1e6, distance_matrix)


def create_potential(distance_matrix, atomic_numbers):
    """Create the Gaussian external potential used in Brockherde et al., 2017,
    Bypassing the Kohn-Sham equations with machine learning.
    """
    Gaussians = np.exp(-distance_matrix**2)
    return -np.matmul(Gaussians, atomic_numbers)


def create_dataset(dir_dataset, filename, basis_set,
                   radius, grid_interval, orbital_dict, property=True):

    """Directory of a preprocessed dataset."""
    if property:
        dir_preprocess = (dir_dataset + "/" + filename + '_' + basis_set + '_' +
                          str(radius) + 'sphere_' +
                          str(grid_interval) + 'grid/')
    else:  # For demo.
        dir_preprocess = filename + '/'
    os.makedirs(dir_preprocess, exist_ok=True)

    """Basis set."""
    inner_outer = [int(b) for b in basis_set[:-1].replace('-', '')]
    inner, outer = inner_outer[0], sum(inner_outer[1:])

    """A sphere for creating the grid field of a molecule."""
    sphere = create_sphere(radius, grid_interval)

    """Load a dataset."""
    with open(Path(dir_dataset, filename+".txt"), 'r') as f:
        dataset = f.read().strip().split('\n\n')

    N = len(dataset)
    percent = 10

    for n, data in enumerate(dataset):

        if 100*n/N >= percent:
            print(str(percent) + '％ has finished.')
            percent += 40

        """Index of the molecular data."""
        data = data.strip().split('\n')
        idx = data[0]

        """Multiple properties (e.g., homo and lumo) can also be processed
        at a time (i.e., the model output has two dimensions).
        """
        if property:
            atom_xyzs = data[1:-1]
            property_values = data[-1].strip().split()
            property_values = np.array([[float(p) for p in property_values]])
        else:
            atom_xyzs = data[1:]

        atoms = []
        atomic_numbers = []
        N_electrons = 0
        atomic_coords = []
        atomic_orbitals = []
        orbital_coords = []
        quantum_numbers = []

        """Load the 3D molecular structure data."""
        for atom_xyz in atom_xyzs:
            atom, x, y, z = atom_xyz.split()
            atoms.append(atom)
            atomic_number = ATOMICNUMBER_DICT[atom]
            atomic_numbers.append([atomic_number])
            N_electrons += atomic_number
            xyz = [float(v) for v in [x, y, z]]
            atomic_coords.append(xyz)

            """Atomic orbitals (basis functions)
            and principle quantum numbers (q=1,2,...).
            """
            if atomic_number <= 2:
                aqs = [(atom+'1s' + str(i), 1) for i in range(outer)]
            elif atomic_number >= 3:
                aqs = ([(atom+'1s' + str(i), 1) for i in range(inner)] +
                       [(atom+'2s' + str(i), 2) for i in range(outer)] +
                       [(atom+'2p' + str(i), 2) for i in range(outer)])
            for a, q in aqs:
                atomic_orbitals.append(a)
                orbital_coords.append(xyz)
                quantum_numbers.append(q)

        """Create each data with the above defined functions."""
        atomic_coords = np.array(atomic_coords)
        atomic_orbitals = create_orbitals(atomic_orbitals, orbital_dict)
        field_coords = create_field(sphere, atomic_coords)
        distance_matrix = create_distancematrix(field_coords, atomic_coords)
        atomic_numbers = np.array(atomic_numbers)
        potential = create_potential(distance_matrix, atomic_numbers)
        distance_matrix = create_distancematrix(field_coords, orbital_coords)
        quantum_numbers = np.array([quantum_numbers])
        N_electrons = np.array([[N_electrons]])
        N_field = len(field_coords)  # The number of points in the grid field.

        """Save the above set of data."""
        data = [idx,
                atomic_orbitals.astype(np.int64),
                distance_matrix.astype(np.float32),
                quantum_numbers.astype(np.float32),
                N_electrons.astype(np.float32),
                N_field]

        if property:
            data += [property_values.astype(np.float32),
                     potential.astype(np.float32)]

        data = np.array(data, dtype=object)
        np.save(dir_preprocess + idx, data)
