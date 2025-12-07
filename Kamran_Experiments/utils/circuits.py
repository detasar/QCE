"""
Quantum Circuits for QKD Experiments

This module provides:
- Bell state preparation circuits
- Measurement basis rotation
- CHSH angle configurations
- QKD circuit generation

Author: Davut Emre Tasar
Date: December 2024
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, List, Dict, Optional

# Check Qiskit availability
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    QuantumCircuit = None


class MeasurementMode(Enum):
    """Measurement mode for QKD protocol."""
    KEY_GENERATION = "key"
    SECURITY_TEST = "security"


@dataclass
class BasisChoice:
    """Single measurement basis choice for one Bell pair."""
    mode: MeasurementMode
    alice_basis: str
    bob_basis: str
    alice_angle: float  # radians
    bob_angle: float    # radians
    alice_setting: int = 0  # 0 or 1 for CHSH
    bob_setting: int = 0    # 0 or 1 for CHSH


@dataclass
class BellStateConfig:
    """Bell state configuration."""
    name: str
    correlation: int  # +1 for same bits, -1 for opposite

    @classmethod
    def phi_plus(cls) -> 'BellStateConfig':
        """(|00> + |11>)/sqrt(2) - BBM92 default, correlated outcomes."""
        return cls('phi_plus', correlation=+1)

    @classmethod
    def phi_minus(cls) -> 'BellStateConfig':
        """(|00> - |11>)/sqrt(2) - correlated outcomes."""
        return cls('phi_minus', correlation=+1)

    @classmethod
    def psi_plus(cls) -> 'BellStateConfig':
        """(|01> + |10>)/sqrt(2) - anti-correlated outcomes."""
        return cls('psi_plus', correlation=-1)

    @classmethod
    def psi_minus(cls) -> 'BellStateConfig':
        """(|01> - |10>)/sqrt(2) - E91 original, anti-correlated outcomes."""
        return cls('psi_minus', correlation=-1)


# Standard CHSH angles for maximum violation
# Alice: 0 and 45 degrees, Bob: 22.5 and 67.5 degrees
CHSH_ANGLES = {
    'alice': {0: 0.0, 1: np.pi/4},           # 0 and 45 degrees
    'bob': {0: np.pi/8, 1: 3*np.pi/8}         # 22.5 and 67.5 degrees
}

# Key generation basis angles
# For QKD: Z basis = 0, X basis = pi/4 (45 degrees)
# This gives cos^2(0) = 1 for matching bases, cos^2(pi/4) = 0.5 for non-matching
KEY_BASIS_ANGLES = {
    'Z': 0.0,                                  # Computational basis (0 degrees)
    'X': np.pi/4                               # Hadamard basis (45 degrees from Z)
}


def choose_measurement_settings(key_fraction: float = 0.90,
                                  rng: np.random.RandomState = None) -> BasisChoice:
    """
    Choose measurement settings for one Bell pair.

    With probability key_fraction, use Z/X basis for key generation.
    Otherwise, use CHSH angles for security testing.

    Args:
        key_fraction: Fraction of samples for key generation (default 90%)
        rng: Random state for reproducibility

    Returns:
        BasisChoice with mode, bases, and angles
    """
    if rng is None:
        rng = np.random.RandomState()

    if rng.random() < key_fraction:
        # KEY GENERATION MODE: Z or X basis
        alice_basis = rng.choice(['Z', 'X'])
        bob_basis = rng.choice(['Z', 'X'])

        alice_angle = KEY_BASIS_ANGLES[alice_basis]
        bob_angle = KEY_BASIS_ANGLES[bob_basis]

        return BasisChoice(
            mode=MeasurementMode.KEY_GENERATION,
            alice_basis=alice_basis,
            bob_basis=bob_basis,
            alice_angle=alice_angle,
            bob_angle=bob_angle
        )
    else:
        # SECURITY TEST MODE: CHSH angles
        alice_setting = rng.choice([0, 1])
        bob_setting = rng.choice([0, 1])

        alice_angle = CHSH_ANGLES['alice'][alice_setting]
        bob_angle = CHSH_ANGLES['bob'][bob_setting]

        alice_names = {0: 'a1=0', 1: 'a2=45'}
        bob_names = {0: 'b1=22.5', 1: 'b2=67.5'}

        return BasisChoice(
            mode=MeasurementMode.SECURITY_TEST,
            alice_basis=alice_names[alice_setting],
            bob_basis=bob_names[bob_setting],
            alice_angle=alice_angle,
            bob_angle=bob_angle,
            alice_setting=alice_setting,
            bob_setting=bob_setting
        )


def create_bell_circuit(bell_config: BellStateConfig = None) -> 'QuantumCircuit':
    """
    Create Bell state preparation circuit.

    Args:
        bell_config: Bell state type (default: Phi+)

    Returns:
        QuantumCircuit that prepares the Bell state
    """
    if not HAS_QISKIT:
        raise ImportError("Qiskit not installed. Run: pip install qiskit qiskit-aer")

    if bell_config is None:
        bell_config = BellStateConfig.phi_plus()

    qc = QuantumCircuit(2, 2)

    # Base Bell state preparation: H on qubit 0, CNOT 0->1
    qc.h(0)
    qc.cx(0, 1)

    # Modify for different Bell states
    if bell_config.name == 'phi_minus':
        qc.z(0)
    elif bell_config.name == 'psi_plus':
        qc.x(1)
    elif bell_config.name == 'psi_minus':
        qc.z(0)
        qc.x(1)

    return qc


def create_qkd_circuit(basis_choice: BasisChoice,
                        bell_config: BellStateConfig = None) -> 'QuantumCircuit':
    """
    Create complete QKD measurement circuit.

    Args:
        basis_choice: Measurement settings
        bell_config: Bell state to prepare (default Phi+)

    Returns:
        QuantumCircuit ready for execution
    """
    if not HAS_QISKIT:
        raise ImportError("Qiskit not installed. Run: pip install qiskit qiskit-aer")

    if bell_config is None:
        bell_config = BellStateConfig.phi_plus()

    qc = create_bell_circuit(bell_config)

    # Measurement basis rotation
    # RY(2*theta) rotates measurement basis by angle theta
    qc.ry(2 * basis_choice.alice_angle, 0)
    qc.ry(2 * basis_choice.bob_angle, 1)

    # Measure both qubits
    qc.measure([0, 1], [0, 1])

    return qc


def create_noise_model(p_single: float, p_two: float = None,
                        p_readout: float = 0.02) -> 'NoiseModel':
    """
    Create realistic noise model for simulation.

    Args:
        p_single: Single-qubit gate error rate
        p_two: Two-qubit gate error rate (default: 2 * p_single)
        p_readout: Readout error rate

    Returns:
        NoiseModel for Aer simulator
    """
    if not HAS_QISKIT:
        raise ImportError("Qiskit not installed")

    if p_two is None:
        p_two = 2 * p_single

    noise_model = NoiseModel()

    # Single-qubit gates (Qiskit 1.x names)
    if p_single > 0:
        single_gates = ['x', 'sx', 'rz', 'ry', 'h', 'id']
        for gate in single_gates:
            error = depolarizing_error(p_single, 1)
            noise_model.add_all_qubit_quantum_error(error, gate)

    # Two-qubit gates
    if p_two > 0:
        two_gates = ['cx', 'cz', 'ecr']
        for gate in two_gates:
            error = depolarizing_error(p_two, 2)
            noise_model.add_all_qubit_quantum_error(error, gate)

    # Readout error
    if p_readout > 0:
        p01 = p_readout * 0.6  # P(1|0)
        p10 = p_readout * 1.4  # P(0|1) - typically higher
        readout_err = ReadoutError([[1 - p01, p01], [p10, 1 - p10]])
        noise_model.add_all_qubit_readout_error(readout_err)

    return noise_model


class QKDSimulator:
    """
    QKD experiment simulator using Qiskit Aer.

    Generates Bell pairs with configurable:
    - Noise model
    - Key/security fraction
    - Number of pairs
    """

    def __init__(self, noise_model: 'NoiseModel' = None, seed: int = 42):
        """
        Initialize QKD simulator.

        Args:
            noise_model: Optional noise model (None = ideal)
            seed: Random seed for reproducibility
        """
        if not HAS_QISKIT:
            raise ImportError("Qiskit not installed. Run: pip install qiskit qiskit-aer")

        self.noise_model = noise_model
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Create simulator backend
        self.backend = AerSimulator(method='statevector')
        if noise_model is not None:
            self.backend = AerSimulator(noise_model=noise_model)

    def generate_qkd_data(self, n_pairs: int = 10000,
                           key_fraction: float = 0.90,
                           shots_per_circuit: int = 1) -> Dict:
        """
        Generate QKD measurement data.

        Args:
            n_pairs: Number of Bell pairs to generate
            key_fraction: Fraction for key generation (rest for security)
            shots_per_circuit: Shots per circuit (1 for single-shot QKD)

        Returns:
            Dictionary with measurement results
        """
        results = {
            'modes': [],
            'alice_bases': [],
            'bob_bases': [],
            'alice_bits': [],
            'bob_bits': [],
            'alice_settings': [],  # For CHSH calculation
            'bob_settings': []
        }

        for i in range(n_pairs):
            # Choose measurement settings
            basis = choose_measurement_settings(key_fraction, self.rng)

            # Create and run circuit
            qc = create_qkd_circuit(basis)
            job = self.backend.run(qc, shots=shots_per_circuit, seed_simulator=self.seed + i)
            counts = job.result().get_counts()

            # Extract outcome (most likely for shots=1)
            outcome = max(counts, key=counts.get)
            bob_bit = int(outcome[0])  # Qiskit bit order is reversed
            alice_bit = int(outcome[1])

            # Store results
            results['modes'].append(basis.mode.value)
            results['alice_bases'].append(basis.alice_basis)
            results['bob_bases'].append(basis.bob_basis)
            results['alice_bits'].append(alice_bit)
            results['bob_bits'].append(bob_bit)
            results['alice_settings'].append(basis.alice_setting)
            results['bob_settings'].append(basis.bob_setting)

        # Convert to numpy arrays
        for key in ['alice_bits', 'bob_bits', 'alice_settings', 'bob_settings']:
            results[key] = np.array(results[key])

        return results


class IdealBellSimulator:
    """
    Fast ideal Bell state simulator without Qiskit.

    Uses analytical probability formulas for Bell state measurements.
    Suitable for large-scale simulations.
    """

    def __init__(self, visibility: float = 1.0, seed: int = 42):
        """
        Initialize ideal simulator.

        Args:
            visibility: State visibility (1.0 = perfect, <1 = mixed with noise)
            seed: Random seed
        """
        self.visibility = visibility
        self.rng = np.random.RandomState(seed)

    def _quantum_probability(self, theta_a: float, theta_b: float) -> float:
        """
        Calculate probability of same outcome for Bell state.

        For |Phi+> = (|00> + |11>)/sqrt(2):
            P(same) = cos^2((theta_a - theta_b))

        With visibility v:
            P(same) = v * cos^2(theta_a - theta_b) + (1-v) * 0.5
        """
        p_same = np.cos(theta_a - theta_b) ** 2
        return self.visibility * p_same + (1 - self.visibility) * 0.5

    def generate_qkd_data(self, n_pairs: int = 10000,
                           key_fraction: float = 0.90) -> Dict:
        """
        Generate QKD measurement data using analytical formulas.

        Args:
            n_pairs: Number of Bell pairs
            key_fraction: Fraction for key generation

        Returns:
            Dictionary with measurement results
        """
        results = {
            'modes': [],
            'alice_bases': [],
            'bob_bases': [],
            'alice_bits': [],
            'bob_bits': [],
            'alice_settings': [],
            'bob_settings': []
        }

        for _ in range(n_pairs):
            # Choose measurement settings
            basis = choose_measurement_settings(key_fraction, self.rng)

            # Calculate probability of same outcome
            p_same = self._quantum_probability(basis.alice_angle, basis.bob_angle)

            # Generate correlated outcomes
            if self.rng.random() < p_same:
                # Same outcome
                bit = self.rng.randint(0, 2)
                alice_bit, bob_bit = bit, bit
            else:
                # Different outcome
                bit = self.rng.randint(0, 2)
                alice_bit, bob_bit = bit, 1 - bit

            # Store results
            results['modes'].append(basis.mode.value)
            results['alice_bases'].append(basis.alice_basis)
            results['bob_bases'].append(basis.bob_basis)
            results['alice_bits'].append(alice_bit)
            results['bob_bits'].append(bob_bit)
            results['alice_settings'].append(basis.alice_setting)
            results['bob_settings'].append(basis.bob_setting)

        # Convert to numpy arrays
        for key in ['alice_bits', 'bob_bits', 'alice_settings', 'bob_settings']:
            results[key] = np.array(results[key])

        return results


def get_simulator(use_qiskit: bool = True, noise_model: 'NoiseModel' = None,
                  visibility: float = 1.0, seed: int = 42):
    """
    Get appropriate simulator based on availability and requirements.

    Args:
        use_qiskit: Whether to use Qiskit (if available)
        noise_model: Noise model for Qiskit simulator
        visibility: Visibility for ideal simulator
        seed: Random seed

    Returns:
        QKDSimulator or IdealBellSimulator instance
    """
    if use_qiskit and HAS_QISKIT:
        return QKDSimulator(noise_model=noise_model, seed=seed)
    else:
        return IdealBellSimulator(visibility=visibility, seed=seed)
