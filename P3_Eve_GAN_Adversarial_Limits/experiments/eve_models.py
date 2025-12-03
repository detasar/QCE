#!/usr/bin/env python3
"""
================================================================================
EVE-GAN MODELS FOR QUANTUM DATA GENERATION
================================================================================

Neural network models for adversarial quantum data generation.
Eve-GAN learns to generate fake Bell measurement data that mimics
real quantum hardware distributions.

ARCHITECTURE:
-------------
- Generator: Takes noise + settings (x,y) -> P(a,b|x,y)
- Discriminator: Distinguishes real vs fake joint distributions
- Training: Adversarial + CHSH constraint + KL divergence

KEY INSIGHT:
------------
Despite achieving:
- Correct CHSH value
- Low KL divergence
- Matching marginals

Eve data remains detectable via conformal prediction (TARA).

Author: Davut Emre Tasar
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List


def get_device():
    """Get best available compute device."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


class EveGenerator(nn.Module):
    """
    Eve-GAN Generator V1.

    Outputs joint distribution P(a,b|x,y) for quantum outcomes.

    Architecture:
        Input: latent (16) + x_onehot (2) + y_onehot (2) = 20
        Hidden: 128 -> 128 -> 64 (LayerNorm + ReLU + Dropout)
        Output: 4-dim softmax (joint probabilities)
    """

    def __init__(self, latent_dim: int = 16, hidden_dims: List[int] = [128, 128, 64]):
        super().__init__()
        self.latent_dim = latent_dim

        input_dim = latent_dim + 4  # latent + one-hot settings

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.LayerNorm(h),
                nn.Dropout(0.1)
            ])
            prev_dim = h

        # Output: 4 joint probabilities for (a,b) in {(0,0), (0,1), (1,0), (1,1)}
        layers.append(nn.Linear(prev_dim, 4))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Generate joint probability distribution.

        Args:
            z: (batch, latent_dim) noise vector
            x, y: (batch,) measurement settings in {0, 1}

        Returns:
            probs: (batch, 4) joint probabilities P(a,b|x,y)
        """
        x_oh = F.one_hot(x.long(), 2).float()
        y_oh = F.one_hot(y.long(), 2).float()
        inp = torch.cat([z, x_oh, y_oh], dim=1)
        logits = self.net(inp)
        return F.softmax(logits, dim=1)

    def sample(self, z: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample (a, b) outcomes from joint distribution."""
        probs = self.forward(z, x, y)
        idx = torch.multinomial(probs, 1).squeeze(1)
        a = idx // 2  # 0,1,2,3 -> 0,0,1,1
        b = idx % 2   # 0,1,2,3 -> 0,1,0,1
        return a, b


class EveGeneratorV2(nn.Module):
    """
    Eve-GAN Generator V2: Improved version with setting-specific heads.

    Key improvements over V1:
    - Separate output heads for each (x,y) setting
    - Better captures anti-correlation in (1,1) setting
    - LayerNorm and residual-like structure

    This architecture better mimics the setting-dependent structure
    of quantum correlations.
    """

    def __init__(self, latent_dim: int = 16, hidden_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
        )

        # Setting-specific output heads
        self.heads = nn.ModuleDict({
            '00': nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 4)),
            '01': nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 4)),
            '10': nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 4)),
            '11': nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 4)),
        })

    def forward(self, z: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate joint probability distribution with setting-specific heads."""
        h = self.encoder(z)
        probs = torch.zeros(z.shape[0], 4, device=z.device)

        for xi in [0, 1]:
            for yi in [0, 1]:
                mask = (x == xi) & (y == yi)
                if mask.sum() > 0:
                    key = f'{xi}{yi}'
                    logits = self.heads[key](h[mask])
                    probs[mask] = F.softmax(logits, dim=-1)

        return probs

    def sample(self, z: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample (a, b) outcomes from joint distribution."""
        probs = self.forward(z, x, y)
        idx = torch.multinomial(probs, 1).squeeze(-1)
        a = idx // 2
        b = idx % 2
        return a, b


class EveDiscriminator(nn.Module):
    """
    Discriminator to distinguish real vs fake quantum data.

    Input: Joint probabilities + one-hot settings
    Output: Probability of being real data
    """

    def __init__(self, hidden_dims: List[int] = [64, 32]):
        super().__init__()

        # Input: probs (4) + x_onehot (2) + y_onehot (2) = 8
        input_dim = 8

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1)
            ])
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, probs: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Discriminate real vs fake.

        Args:
            probs: (batch, 4) outcome probabilities
            x, y: (batch,) measurement settings

        Returns:
            score: (batch, 1) probability of being real
        """
        x_oh = F.one_hot(x.long(), 2).float()
        y_oh = F.one_hot(y.long(), 2).float()
        inp = torch.cat([probs, x_oh, y_oh], dim=1)
        return self.net(inp)


def compute_chsh_torch(data: Dict[str, torch.Tensor]) -> float:
    """Compute CHSH value from torch tensors."""
    x = data['x'].cpu().numpy()
    y = data['y'].cpu().numpy()
    a = data['a'].cpu().numpy()
    b = data['b'].cpu().numpy()

    a_pm = 2 * a - 1  # Convert to +/-1
    b_pm = 2 * b - 1

    E = {}
    for xi in [0, 1]:
        for yi in [0, 1]:
            mask = (x == xi) & (y == yi)
            if mask.sum() > 0:
                E[(xi, yi)] = (a_pm[mask] * b_pm[mask]).mean()
            else:
                E[(xi, yi)] = 0.0

    return E[(0, 0)] + E[(0, 1)] + E[(1, 0)] - E[(1, 1)]


def compute_target_distributions(ibm_data: Dict[str, np.ndarray]) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Compute target P(a,b|x,y) distributions from IBM quantum data.

    Args:
        ibm_data: Dict with 'x', 'y', 'a', 'b' arrays

    Returns:
        Dict mapping (x, y) to 4-dim probability vector
    """
    x, y, a, b = ibm_data['x'], ibm_data['y'], ibm_data['a'], ibm_data['b']
    targets = {}

    for xi in [0, 1]:
        for yi in [0, 1]:
            mask = (x == xi) & (y == yi)
            if mask.sum() > 0:
                counts = np.zeros(4)
                for ai in [0, 1]:
                    for bi in [0, 1]:
                        idx = ai * 2 + bi
                        counts[idx] = np.sum((a[mask] == ai) & (b[mask] == bi))
                counts += 1e-6  # Smoothing
                targets[(xi, yi)] = counts / counts.sum()
            else:
                targets[(xi, yi)] = np.ones(4) / 4

    return targets


def kl_divergence_loss(eve_probs: torch.Tensor,
                       target_probs: Dict[Tuple[int, int], np.ndarray],
                       x: torch.Tensor,
                       y: torch.Tensor,
                       device: torch.device) -> torch.Tensor:
    """
    Compute setting-conditional KL divergence loss.

    Args:
        eve_probs: (batch, 4) generated probabilities
        target_probs: Target distributions per setting
        x, y: Setting tensors
        device: Compute device

    Returns:
        Average KL divergence across settings
    """
    total_kl = torch.tensor(0.0, device=device)
    count = 0

    for xi in [0, 1]:
        for yi in [0, 1]:
            mask = (x == xi) & (y == yi)
            if mask.sum() > 0:
                eve_dist = eve_probs[mask].mean(dim=0)
                target = torch.tensor(target_probs[(xi, yi)], device=device, dtype=torch.float32)
                kl = F.kl_div(torch.log(eve_dist + 1e-10), target, reduction='sum')
                total_kl = total_kl + kl
                count += 1

    return total_kl / max(count, 1)


def generate_eve_samples(generator: nn.Module,
                         n_samples: int,
                         device: Optional[torch.device] = None) -> Dict[str, np.ndarray]:
    """
    Generate quantum data samples from trained Eve model.

    Args:
        generator: Eve generator model
        n_samples: Number of samples to generate
        device: Torch device (auto-detected if None)

    Returns:
        Dict with 'x', 'y', 'a', 'b' numpy arrays
    """
    if device is None:
        device = next(generator.parameters()).device

    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, generator.latent_dim, device=device)
        x = torch.randint(0, 2, (n_samples,), device=device)
        y = torch.randint(0, 2, (n_samples,), device=device)
        a, b = generator.sample(z, x, y)

    return {
        'x': x.cpu().numpy(),
        'y': y.cpu().numpy(),
        'a': a.cpu().numpy(),
        'b': b.cpu().numpy()
    }


def load_eve_model(filepath: str,
                   model_class: str = 'v1',
                   device: Optional[torch.device] = None) -> nn.Module:
    """
    Load trained Eve model from checkpoint.

    Args:
        filepath: Path to .pt checkpoint file
        model_class: 'v1' or 'v2' architecture
        device: Torch device

    Returns:
        Loaded generator model in eval mode
    """
    if device is None:
        device = get_device()

    if model_class == 'v1':
        generator = EveGenerator(latent_dim=16, hidden_dims=[128, 128, 64])
    else:
        generator = EveGeneratorV2(latent_dim=16, hidden_dim=128)

    generator = generator.to(device)
    ckpt = torch.load(filepath, map_location=device, weights_only=False)
    generator.load_state_dict(ckpt['generator'])
    generator.eval()

    return generator


def train_eve_gan(ibm_data: Dict[str, np.ndarray],
                  epochs: int = 2000,
                  batch_size: int = 1024,
                  lr_g: float = 1e-3,
                  lr_d: float = 1e-4,
                  lambda_chsh: float = 10.0,
                  lambda_kl: float = 20.0,
                  target_S: float = 2.5,
                  model_version: str = 'v2',
                  save_path: Optional[str] = None,
                  verbose: bool = True) -> Tuple[nn.Module, Dict]:
    """
    Train Eve-GAN model.

    The training objective combines:
    1. GAN loss: Fool the discriminator
    2. CHSH loss: Match target Bell violation
    3. KL loss: Match per-setting distributions

    Args:
        ibm_data: Training data from quantum hardware
        epochs: Number of training epochs
        batch_size: Training batch size
        lr_g, lr_d: Learning rates for generator/discriminator
        lambda_chsh: Weight for CHSH constraint
        lambda_kl: Weight for KL divergence loss
        target_S: Target CHSH value (default 2.5)
        model_version: 'v1' or 'v2' architecture
        save_path: Path to save trained model
        verbose: Print training progress

    Returns:
        Tuple of (trained_generator, training_history)
    """
    device = get_device()
    if verbose:
        print(f"Training on device: {device}")
        print(f"Model version: {model_version}")

    # Target distributions from IBM data
    target_probs = compute_target_distributions(ibm_data)

    # Initialize models
    if model_version == 'v2':
        generator = EveGeneratorV2(latent_dim=16, hidden_dim=128).to(device)
    else:
        generator = EveGenerator(latent_dim=16, hidden_dims=[128, 128, 64]).to(device)

    discriminator = EveDiscriminator(hidden_dims=[64, 32]).to(device)

    # Optimizers
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

    # Prepare data tensors
    n = len(ibm_data['x'])
    x_all = torch.tensor(ibm_data['x'], device=device)
    y_all = torch.tensor(ibm_data['y'], device=device)
    a_all = torch.tensor(ibm_data['a'], device=device)
    b_all = torch.tensor(ibm_data['b'], device=device)

    # Real probabilities as one-hot
    real_probs = torch.zeros(n, 4, device=device)
    for i in range(n):
        idx = a_all[i] * 2 + b_all[i]
        real_probs[i, idx] = 1.0

    history = {'epoch': [], 'g_loss': [], 'd_loss': [], 'chsh': [], 'kl': []}

    for epoch in range(epochs):
        # Sample batch
        idx = torch.randint(0, n, (batch_size,))
        x_batch = x_all[idx]
        y_batch = y_all[idx]
        real_batch = real_probs[idx]

        # Train Discriminator
        z = torch.randn(batch_size, generator.latent_dim, device=device)
        fake_probs = generator(z, x_batch, y_batch)

        real_score = discriminator(real_batch, x_batch, y_batch)
        fake_score = discriminator(fake_probs.detach(), x_batch, y_batch)

        d_loss = -torch.mean(torch.log(real_score + 1e-8)) - torch.mean(torch.log(1 - fake_score + 1e-8))

        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()

        # Train Generator
        z = torch.randn(batch_size, generator.latent_dim, device=device)
        fake_probs = generator(z, x_batch, y_batch)
        fake_score = discriminator(fake_probs, x_batch, y_batch)

        # GAN loss
        g_loss_gan = -torch.mean(torch.log(fake_score + 1e-8))

        # CHSH loss
        a_gen, b_gen = generator.sample(z, x_batch, y_batch)
        sample_data = {'x': x_batch, 'y': y_batch, 'a': a_gen, 'b': b_gen}
        current_chsh = compute_chsh_torch(sample_data)
        chsh_loss = (current_chsh - target_S) ** 2

        # KL loss
        kl_loss = kl_divergence_loss(fake_probs, target_probs, x_batch, y_batch, device)

        # Total generator loss
        g_loss = g_loss_gan + lambda_chsh * chsh_loss + lambda_kl * kl_loss

        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()

        # Logging
        if epoch % 100 == 0:
            with torch.no_grad():
                z_test = torch.randn(10000, generator.latent_dim, device=device)
                x_test = torch.randint(0, 2, (10000,), device=device)
                y_test = torch.randint(0, 2, (10000,), device=device)
                test_probs = generator(z_test, x_test, y_test)
                full_kl = kl_divergence_loss(test_probs, target_probs, x_test, y_test, device).item()

            history['epoch'].append(epoch)
            history['g_loss'].append(g_loss.item())
            history['d_loss'].append(d_loss.item())
            history['chsh'].append(current_chsh)
            history['kl'].append(full_kl)

            if verbose:
                print(f"Epoch {epoch}: G={g_loss.item():.4f}, CHSH={current_chsh:.4f}, KL={full_kl:.4f}")

            # Early stopping
            if full_kl < 0.1 and current_chsh > target_S:
                if verbose:
                    print(f"Target reached at epoch {epoch}!")
                break

    # Save model
    if save_path:
        torch.save({
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'history': history,
            'target_probs': {str(k): v.tolist() for k, v in target_probs.items()},
            'config': {
                'model_version': model_version,
                'target_S': target_S,
                'lambda_chsh': lambda_chsh,
                'lambda_kl': lambda_kl
            }
        }, save_path)
        if verbose:
            print(f"Model saved to {save_path}")

    return generator, history


if __name__ == '__main__':
    # Quick test
    print("Testing Eve-GAN models...")

    device = get_device()
    print(f"Device: {device}")

    # Test V1
    gen_v1 = EveGenerator().to(device)
    z = torch.randn(100, 16, device=device)
    x = torch.randint(0, 2, (100,), device=device)
    y = torch.randint(0, 2, (100,), device=device)
    probs = gen_v1(z, x, y)
    print(f"V1 output shape: {probs.shape}")

    # Test V2
    gen_v2 = EveGeneratorV2().to(device)
    probs_v2 = gen_v2(z, x, y)
    print(f"V2 output shape: {probs_v2.shape}")

    # Test sampling
    data = generate_eve_samples(gen_v2, 1000)
    print(f"Generated data: {len(data['x'])} samples")

    print("All tests passed!")
