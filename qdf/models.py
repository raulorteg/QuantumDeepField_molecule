
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class QuantumDeepField(nn.Module):
    def __init__(self, device, N_orbitals,
                 dim, layer_functional, operation, N_output,
                 hidden_HK, layer_HK,):
        super(QuantumDeepField, self).__init__()

        """All learning parameters of the QDF model."""
        self.coefficient = nn.Embedding(N_orbitals, dim)
        self.zeta = nn.Embedding(N_orbitals, 1)  # Orbital exponent.
        nn.init.ones_(self.zeta.weight)  # Initialize each zeta with one.
        self.W_functional = nn.ModuleList([nn.Linear(dim, dim)
                                           for _ in range(layer_functional)])
        self.W_property = nn.Linear(dim, N_output)
        self.W_density = nn.Linear(1, hidden_HK)
        self.W_HK = nn.ModuleList([nn.Linear(hidden_HK, hidden_HK)
                                   for _ in range(layer_HK)])
        self.W_potential = nn.Linear(hidden_HK, 1)

        self.device = device
        self.dim = dim
        self.layer_functional = layer_functional
        self.operation = operation
        self.layer_HK = layer_HK

    def list_to_batch(self, xs, dtype=torch.FloatTensor, cat=None, axis=None):
        """Transform the list of numpy data into the batch of tensor data."""
        xs = [dtype(x).to(self.device) for x in xs]
        if cat:
            return torch.cat(xs, axis)
        else:
            return xs  # w/o cat (i.e., the list (not batch) of tensor data).

    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0 and large value) for batch processing.
        For example, given a list of matrices [A, B, C],
        this function returns a new matrix [A00, 0B0, 00C],
        where 0 is the zero matrix (i.e., a block diagonal matrix).
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        pad_matrices = torch.full((M, N), pad_value, device=self.device)
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            matrix = torch.FloatTensor(matrix).to(self.device)
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices

    def basis_matrix(self, atomic_orbitals,
                     distance_matrices, quantum_numbers):
        """Transform the distance matrix into a basis matrix,
        in which each element is d^(q-1)*e^(-z*d^2), where d is the distance,
        q is the principle quantum number, and z is the orbital exponent.
        We note that this is a simplified Gaussian-type orbital (GTO)
        in terms of the spherical harmonics.
        We simply normalize the GTO basis matrix using F.normalize in PyTorch.
        """
        zetas = torch.squeeze(self.zeta(atomic_orbitals))
        GTOs = (distance_matrices**(quantum_numbers-1) *
                torch.exp(-zetas*distance_matrices**2))
        GTOs = F.normalize(GTOs, 2, 0)
        # print(torch.sum(torch.t(GTOs)[0]**2))  # Normalization check.
        return GTOs

    def LCAO(self, inputs):
        """Linear combination of atomic orbitals (LCAO)."""

        """Inputs."""
        (atomic_orbitals, distance_matrices,
         quantum_numbers, N_electrons, N_fields) = inputs

        """Cat or pad each input data for batch processing."""
        atomic_orbitals = self.list_to_batch(atomic_orbitals, torch.LongTensor)
        distance_matrices = self.pad(distance_matrices, 1e6)
        quantum_numbers = self.list_to_batch(quantum_numbers, cat=True, axis=1)
        N_electrons = self.list_to_batch(N_electrons)

        """Normalize the coefficients in LCAO."""
        coefficients = []
        for AOs in atomic_orbitals:
            coefs = F.normalize(self.coefficient(AOs), 2, 0)
            # print(torch.sum(torch.t(coefs)[0]**2))  # Normalization check.
            coefficients.append(coefs)
        coefficients = torch.cat(coefficients)
        atomic_orbitals = torch.cat(atomic_orbitals)

        """LCAO."""
        basis_matrix = self.basis_matrix(atomic_orbitals,
                                         distance_matrices, quantum_numbers)
        molecular_orbitals = torch.matmul(basis_matrix, coefficients)

        """We simply normalize the molecular orbitals
        and keep the total electrons of the molecule
        in learning the molecular orbitals.
        """
        split_MOs = torch.split(molecular_orbitals, N_fields)
        normalized_MOs = []
        for N_elec, MOs in zip(N_electrons, split_MOs):
            MOs = torch.sqrt(N_elec/self.dim) * F.normalize(MOs, 2, 0)
            # print(torch.sum(MOs**2), N_elec)  # Total electrons check.
            normalized_MOs.append(MOs)

        return torch.cat(normalized_MOs)

    def functional(self, vectors, layers, operation, axis):
        """DNN-based energy functional."""
        for l in range(layers):
            vectors = torch.relu(self.W_functional[l](vectors))
        if operation == 'sum':
            vectors = [torch.sum(vs, 0) for vs in torch.split(vectors, axis)]
        if operation == 'mean':
            vectors = [torch.mean(vs, 0) for vs in torch.split(vectors, axis)]
        return torch.stack(vectors)

    def HKmap(self, scalars, layers):
        """DNN-based Hohenberg--Kohn map."""
        vectors = self.W_density(scalars)
        for l in range(layers):
            vectors = torch.relu(self.W_HK[l](vectors))
        return self.W_potential(vectors)

    def forward(self, data, train=False, target=None, predict=False):
        """Forward computation of the QDF model
        using the above defined functions.
        """

        idx, inputs, N_fields = data[0], data[1:6], data[5]

        if predict:  # For demo.
            with torch.no_grad():
                molecular_orbitals = self.LCAO(inputs)
                final_layer = self.functional(molecular_orbitals,
                                              self.layer_functional,
                                              self.operation, N_fields)
                E_ = self.W_property(final_layer)
                return idx, E_

        elif train:
            molecular_orbitals = self.LCAO(inputs)
            if target == 'E':  # Supervised learning for energy.
                E = self.list_to_batch(data[6], cat=True, axis=0)  # Correct E.
                final_layer = self.functional(molecular_orbitals,
                                              self.layer_functional,
                                              self.operation, N_fields)
                E_ = self.W_property(final_layer)  # Predicted E.
                loss = F.mse_loss(E, E_)
            if target == 'V':  # Unsupervised learning for potential.
                V = self.list_to_batch(data[7], cat=True, axis=0)  # Correct V.
                densities = torch.sum(molecular_orbitals**2, 1)
                densities = torch.unsqueeze(densities, 1)
                V_ = self.HKmap(densities, self.layer_HK)  # Predicted V.
                loss = F.mse_loss(V, V_)
            return loss

        else:  # Test.
            with torch.no_grad():
                E = self.list_to_batch(data[6], cat=True, axis=0)
                molecular_orbitals = self.LCAO(inputs)
                final_layer = self.functional(molecular_orbitals,
                                              self.layer_functional,
                                              self.operation, N_fields)
                E_ = self.W_property(final_layer)
                return idx, E, E_