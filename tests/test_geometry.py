import numpy as np
import pytest
import torch

from neurotools import geometry, stats
from neurotools.embed import MDScale


def test_hellinger_statistics_normalize_inputs_and_match_known_values():
    p = torch.tensor([2.0, 0.0])
    q = torch.tensor([1.0, 1.0])

    similarity = stats.hellinger_similarity(p, q)
    distance = stats.hellinger_distance(p, q)

    expected_similarity = 1.0 / np.sqrt(2.0)
    assert torch.allclose(similarity, torch.tensor(expected_similarity, dtype=similarity.dtype))
    assert torch.allclose(distance, torch.sqrt(1.0 - similarity))


def test_hellinger_statistics_reject_invalid_distributions():
    with pytest.raises(ValueError, match="nonnegative"):
        stats.hellinger_distance(torch.tensor([1.0, -0.1]), torch.tensor([1.0, 0.0]))

    with pytest.raises(ValueError, match="positive mass"):
        stats.hellinger_similarity(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 0.0]))


def test_hellinger_geometry_constructs_similarity_and_distance_matrices():
    membership = torch.tensor([
        [1.0, 0.0],
        [0.0, 3.0],
        [1.0, 1.0],
    ])

    similarity = geometry.hellinger_similarity_matrix(membership)
    distance = geometry.hellinger_distance_matrix(membership)

    assert similarity.shape == (3, 3)
    assert distance.shape == (3, 3)
    assert torch.allclose(similarity, similarity.T)
    assert torch.allclose(distance, distance.T)
    assert torch.equal(torch.diag(similarity), torch.ones(3))
    assert torch.equal(torch.diag(distance), torch.zeros(3))
    assert similarity[0, 1] == 0.0
    assert distance[0, 1] == 1.0
    assert torch.allclose(distance, torch.sqrt(torch.clamp(1.0 - similarity, min=0.0)))

    batched_distance = geometry.hellinger_distance_matrix(torch.stack([membership, membership]))
    assert batched_distance.shape == (2, 3, 3)
    assert torch.allclose(batched_distance[0], distance)


def test_hellinger_dissimilarity_supports_condensed_and_mds_inputs():
    membership = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])
    distance_matrix = geometry.hellinger_distance_matrix(membership)

    condensed = geometry.dissimilarity(membership, metric="hellinger")
    checked_matrix = MDScale(3).check_dists(distance_matrix)
    checked_condensed = MDScale(3).check_dists(condensed.squeeze(0))

    expected = distance_matrix[torch.triu_indices(3, 3, offset=1).unbind()]
    assert condensed.shape == (1, 3)
    assert torch.allclose(condensed.squeeze(0), expected)
    assert torch.allclose(checked_matrix[0], expected)
    assert torch.allclose(checked_condensed[0], expected)
