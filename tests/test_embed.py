import numpy as np
import pytest
import torch

from neurotools.embed import MDScale


def toroidal_model(metric):
    model = MDScale(2, struct="toroid", toroid_metric=metric)
    with torch.no_grad():
        model.log_rad_phi.copy_(torch.log(torch.tensor(2.0)))
        model.log_rad_theta.copy_(torch.log(torch.tensor(3.0)))
    return model


def test_toroid_surface_distance_uses_wrapped_intrinsic_metric():
    model = toroidal_model("surface")
    angles = torch.tensor([[0.0, 0.0], [1.5 * np.pi, np.pi]])

    distance = model.torus_distances(angles)

    # Wrapped deltas are (pi / 2, pi), scaled by radii (2, 3).
    expected = np.pi * np.sqrt(10.0)
    assert torch.allclose(distance, torch.tensor([expected], dtype=distance.dtype), atol=1e-6)


def test_toroid_ambient_distance_uses_r4_clifford_torus_chords():
    model = toroidal_model("ambient")
    angles = torch.tensor([[0.0, 0.0], [1.5 * np.pi, np.pi]])

    distance = model.torus_distances(angles)

    # Chords are 2*r_phi*sin(pi/4) and 2*r_theta*sin(pi/2).
    expected = np.sqrt(44.0)
    assert torch.allclose(distance, torch.tensor([expected], dtype=distance.dtype), atol=1e-6)


def test_toroid_predict_returns_two_wrapped_angles():
    model = MDScale(2, struct="toroid")
    model.right_latent = torch.nn.Parameter(
        torch.tensor([[-0.5 * np.pi, 5.0 * np.pi], [2.0 * np.pi, -2.0 * np.pi]])
    )

    coordinates = model.predict()

    assert coordinates.shape == (2, 2)
    assert torch.all(coordinates >= 0.0)
    assert torch.all(coordinates < 2.0 * np.pi)
    assert torch.allclose(coordinates[0], torch.tensor([1.5 * np.pi, np.pi]), atol=1e-6)
    assert torch.allclose(coordinates[1], torch.tensor([0.0, 0.0]), atol=1e-6)


def test_toroid_requires_two_angles_and_a_supported_metric():
    with pytest.raises(ValueError, match="embed_dims=2"):
        MDScale(3, embed_dims=3, struct="toroid")

    with pytest.raises(ValueError, match="toroid_metric"):
        MDScale(3, struct="toroid", toroid_metric="invalid")


def test_toroid_initialization_covers_phase_domain_and_scales_radii_to_targets():
    torch.manual_seed(4)
    model = MDScale(512, initialization="xavier", struct="toroid")
    target = torch.linspace(0.1, 1.0, 512 * 511 // 2)

    angles = model._initialize_toroid_embedding(target)

    assert torch.all(angles >= 0.0)
    assert torch.all(angles < 2.0 * np.pi)
    assert torch.all(angles.std(dim=0) > 1.0)
    expected_radius = torch.median(target) / torch.pi
    assert torch.allclose(model.rad_phi, expected_radius)
    assert torch.allclose(model.rad_theta, expected_radius)


def test_toroid_uses_supplied_initial_angles_and_reports_normalized_stress():
    levels = 5
    axis = torch.arange(levels) * (2.0 * torch.pi / levels)
    phi, theta = torch.meshgrid(axis, axis, indexing="ij")
    initial_angles = torch.stack([phi.flatten(), theta.flatten()], dim=1)

    source = MDScale(levels ** 2, struct="toroid")
    with torch.no_grad():
        source.log_rad_phi.copy_(torch.log(torch.tensor(0.2)))
        source.log_rad_theta.copy_(torch.log(torch.tensor(0.35)))
    target_vector = source.torus_distances(initial_angles).detach()

    model = MDScale(
        levels ** 2,
        struct="toroid",
        initial_angles=initial_angles,
        lr=0.05,
    )
    model.embed(target_vector, max_iter=200)

    expected_stress = model.stress_history[-1] / torch.linalg.vector_norm(target_vector).item()
    assert model.normalized_stress == pytest.approx(expected_stress)
    assert model.normalized_stress < 5e-3
