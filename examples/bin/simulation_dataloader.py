import matplotlib.pyplot as plt
import torch
import numpy as np
from neurotools import util
from scipy.ndimage import affine_transform, gaussian_filter

n_classes = 12

def embed_unit_circles(n, k, d, generator=None):
    """
    Embed n points along a unit circle on k parallel planes in a d-dimensional space.

    Parameters:
        n (int): Number of points to embed along a unit circle per plane.
        k (int): Number of parallel planes.
        d (int): Dimensionality of the ambient space.

    Returns:
        np.ndarray: Array of shape (n * k, d) containing the embedded points.
    """
    if d < 2:
        raise ValueError("The dimensionality 'd' must be at least 2.")

    if generator is None:
        generator = np.random.default_rng(np.random.randint(0, 100000))

    # Randomly choose a 2D subspace (basis vectors) # bias toward including all d in basis
    basis = generator.normal(size=(d, 2), loc=.5, scale=.5) * (generator.integers(0, 1, size=(d, 2)) * 2 - 1)
    basis, _ = np.linalg.qr(basis)  # Orthonormalize the basis

    # Generate unit circle points in the 2D subspace
    offset = generator.random() * 2 * np.pi
    angles = np.linspace(offset, 2 * np.pi + offset, n, endpoint=False)
    angles = np.stack([angles + .2 * generator.random() + o * .05 for o in range(k)], axis=1)
    mod = (generator.random(2) + 2)
    mod = 2 * mod / np.sum(mod)
    circle_points = np.stack((mod[0] * np.cos(angles), mod[1] * np.sin(angles)), axis=2)  # Shape (n, k, 2)

    # Embed circle points in the ambient d-dimensional space
    embedded_points = circle_points @ basis.T  # Shape (n, k, d)
    # Generate k parallel planes by translating the 2D subspace in a random direction
    translation_vector = generator.normal(size=d)
    # translation_vector = translation_vector + .3 * np.sign(translation_vector)
    translation_vector -= np.dot(translation_vector, basis[:, 0]) * basis[:, 0]
    translation_vector -= np.dot(translation_vector, basis[:, 1]) * basis[:, 1]

    # Normalize the translation vector
    translation_vector /= np.linalg.norm(translation_vector)
    translation_vector = translation_vector.reshape((1, 1, d))

    # Create offsets for the parallel planes
    offsets = (.3 * generator.random() + .5) * np.linspace(-1, 1, k).reshape((1, -1, 1))  # Shape (1, k, 1)

    # Generate points on all planes
    plane_points = embedded_points + offsets * translation_vector

    # interleave parallel planes
    return plane_points.reshape((n, k, d))  # Shape (n, k, d)


np.random.seed(42) # set a default seed for all non-generator np calls.


class SimpleDataLoader:
    """
    generates cross decodable and non-cross docadable 2d rois. Only tests the simple case where cross decodable rois
    directly overlap with their template ROIs
    """

    def __init__(self, spatial=16, epochs=500, num_examples_per_class=100, difficulty=1, motion=True,
                 structured_noise=True, n_classes=8, seed=42, stable_seed=23567, debug=False):
        self.spatial = spatial
        # a only, b patterns that appear on a trials, a patterns that appear on b trial, b only
        self.roi_n = [2, 1,]
        self.classes_per_modality = n_classes
        self.batch_size = 70
        self.n_classes = n_classes
        self.seed = seed
        self.epochs = epochs
        self.warp = motion
        self.sessions = ["s1", "s2"]
        self.difficulty = difficulty
        if self.difficulty < 0 or self.difficulty >= 20:
            raise ValueError("The difficulty '{}' must be between 0 and 20".format(difficulty))


        self.origins = [[[1, 1, 8], [3, 11, 8]],
                        [[10, 9, 8]],]

        self.template_patterns = []
        gen = np.random.default_rng(seed=stable_seed)
        for s, n in enumerate(self.roi_n):
            xyz = gen.random(size=(3,))
            xyz = 0 * xyz / np.sum(np.abs(xyz))
            rxyz = gen.random(size=(3,)) - .5
            rxyz = 40 * (rxyz / np.sum(np.abs(rxyz)))
            sxyz = (gen.random(size=(3,)))
            sxyz = .9 * (sxyz / np.sum(np.abs(sxyz))) + .45
            m = np.power((1 / np.prod(sxyz)), (1 / 3))
            sxyz = sxyz * m + .1
            xyz = tuple(xyz.tolist())
            rxyz = tuple(rxyz.tolist())
            sxyz = tuple(sxyz.tolist())

            # generate patterns for each class
            class_pattern_structured = [embed_unit_circles(n_classes // 2, 2, 125, generator=gen).reshape((n_classes, 5, 5, 5), order="f") for _ in range(n)]
            class_pattern_structured = [self.transform_volume(v, xyz, rxyz, sxyz) for v in class_pattern_structured]
            class_patterns = 1.38 * np.stack(class_pattern_structured, axis=0)[None, ...]
            class_patterns = np.tile(class_patterns, (2, 1, 1, 1, 1, 1))
            class_patterns_sep = (gen.random((2, n, n_classes, 5, 5, 5)) - .5)
            # rand fraction scale with difficulty
            r_frac = .05 * self.difficulty
            self.template_patterns.append((1 - r_frac) * class_patterns + r_frac * class_patterns_sep)
        # np.random.seed(s + 2)
        gen = np.random.default_rng(seed=seed)
        self.a_noise_template = gen.random((spatial, spatial, spatial))
        # np.random.seed(s + 3)
        self.b_noise_template = gen.random((spatial, spatial, spatial)) + .1

        self.data = {}
        self.targets = []
        for m in ["a", "b"]:
            self.data[m] = []
            for c in range(n_classes):
                stim, _ = self.generate_example(batch_size=num_examples_per_class,
                                                noise=.15 * self.difficulty,
                                                stim_type=m, class_id=c, use_pattern=True)
                if m == "b":
                    stim *= (1 + difficulty * .02 * c)
                    stim += difficulty * .02 * c
                self.data[m].append(stim)
            template = np.concatenate(self.data[m], axis=0)
            template = (template - template.mean(axis=0)[None, ...]) / template.std(axis=0)[None, :]
            self.data[m] = template
        for c in range(n_classes):
            self.targets += [c] * num_examples_per_class
        self.targets = np.array(self.targets)

    def __getitem__(self, item: int):
        if item == 0:
            return self.batch_iterator(batch_size=self.batch_size, num_batches=self.epochs, modality="a")
        if item == 1:
            return self.batch_iterator(batch_size=self.batch_size, num_batches=self.epochs, modality="b")
        else:
            raise IndexError

    def transform_volume(self, volume, xyz, rxyz, sxyz):
        if volume.ndim != 4:
            raise IndexError
        affine = util.affine_from_params(rxyz, sxyz, xyz)
        # affine = np.linalg.inv(affine)
        # convert push to pull transform

        affine[:3, :3] = np.linalg.inv(affine[:3, :3])
        # Get the shape of the input volume
        b, w, h, d = volume.shape
        c_in = np.array([w, h, d]) / 2
        c_out = c_in
        offset = c_in - c_out.dot(affine[:3, :3])

        for i in range(b):
            volume[i] = affine_transform(volume[i], affine[:3, :3].T, offset=offset, order=1)
        return volume

    def generate_example(self, batch_size, noise: float, stim_type: str, class_id=None, use_pattern=True, warp=True):
        gen = np.random.default_rng(self.seed + 42)
        template = np.zeros((batch_size, self.spatial, self.spatial, self.spatial), dtype=float)
        if class_id is None:
            class_labels = gen.integers(low=0, high=self.classes_per_modality, size=(batch_size,))
        else:
            class_labels = np.ones((batch_size,), dtype=int) * class_id

        for i, type_origins in enumerate(self.origins):
            if stim_type == "a":
                ind=0
                template += self.a_noise_template * .3
                if i not in [0, 1, 2]:
                    continue
            elif stim_type == "b":
                ind=1
                template += self.b_noise_template * .7
                if i not in [1, 2, 3]:
                    continue
            else:
                ind = 0
                def warper(x):
                    return x
            for j, origin in enumerate(type_origins):
                pattern = self.template_patterns[i][ind, j, class_labels, :, :, :] * 1
                if use_pattern:
                    kern = np.zeros((5, 5, 5))
                    kern[2, 2, 2] = 1.0
                    kern = gaussian_filter(kern, sigma=.25 * 7)
                    m = 1 / np.max(kern)
                    kern = kern * m
                    code = pattern
                    code = code * kern[None, ...]
                else:
                    code = i + 1
                template[:, origin[0]:origin[0] + pattern.shape[1],
                origin[1]:origin[1] + pattern.shape[2],
                origin[2]:origin[2] + pattern.shape[3]] += code
        # np.random.seed(None)
        for i in range(batch_size):
            trans_xyz = 0 * gen.normal(loc=0, scale=self.difficulty * 0.0, size=(3,))
            scale_xyz = 1. + 0 * gen.normal(loc=0, scale=.05 * self.difficulty, size=(3,))
            rot_xyz = 0*  gen.normal(loc=0, scale=.1 * self.difficulty, size=(3,))
            if warp:
                template[i] = self.transform_volume(template[i][None, ...], trans_xyz, rot_xyz,
                                                    scale_xyz).squeeze()  # warper(torch.from_numpy(template[0][None, ...])).numpy()[0]
                pass

        if noise > 0.:
            template = template + gen.normal(0., noise, size=template.shape) + noise * gen.random(size=template.shape)
            pass

        if stim_type == "a" or stim_type == "b":
            template = (template - np.expand_dims(template.mean(axis=(1, 2, 3)), (1, 2, 3))) / np.expand_dims(
                template.std(axis=(1, 2, 3)), (1, 2, 3))
            template[np.isnan(template)] = 0
        return template.squeeze(), class_labels

    def plot_template_pattern(self):
        fig, ax = plt.subplots(1)
        template, _ = self.generate_example(1, 0.0, "all", use_pattern=False)
        template = template.squeeze()
        plt.imshow(template[:, :, 8])
        fig.show()

    def plot_circle_embedding(self):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection='3d')
        test = embed_unit_circles(n_classes // 2, 2, 3).reshape((-1, 3), order="f")
        ax.scatter(test[:, 0], test[:, 1], test[:, 2], color=["blue"] * 6 + ["red"] * 6)
        plt.show()

    def batch_iterator(self, modality: str, batch_size=70, num_batches=500):
        gen = np.random.default_rng(self.seed + 10249)
        for epoch in range(num_batches):
            examples = gen.integers(0, len(self.targets), size=batch_size)
            targets = self.targets[examples]
            data = self.data[modality][examples]
            data = data[:, None, :, :]
            yield data, targets