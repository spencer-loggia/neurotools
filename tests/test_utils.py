from neurotools import util
import torch
import pytest


def test_pearson_corr_equivilance():
    x1 = torch.Tensor([1., 2., 3.])
    x2 = torch.Tensor([1., 2., 3.])
    corr = util.pearson_correlation(x1, x2)
    assert corr == 1.


def test_pearson_corr_fullcorr():
    x1 = torch.Tensor([1., 2., 3.])
    x2 = torch.Tensor([.5, 1., 1.5])
    corr = util.pearson_correlation(x1, x2)
    assert corr == 1.


def test_pearson_uncorr():
    x1 = torch.Tensor([1., 2., 3.])
    x2 = torch.Tensor([1., 1.2, 1.])
    corr = util.pearson_correlation(x1, x2)
    assert corr == 0.0


def test_pearson_anticorr():
    x1 = torch.Tensor([1., 2., 3.])
    x2 = torch.Tensor([3., 2., 1.])
    corr = util.pearson_correlation(x1, x2)
    assert corr == -1.
