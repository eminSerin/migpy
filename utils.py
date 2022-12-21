# Load libraries
import os.path as op
from functools import wraps

import nibabel as nib
import numpy as np
from nilearn.masking import apply_mask
from scipy.sparse.linalg import svds


def _to_float32(func):
    """Decorator to convert input to float32."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        return np.array(func(*args, **kwargs), dtype=np.float32)

    return wrapper


def _demean(d, axis=0):
    """Demean data."""
    return d - d.mean(axis=axis, keepdims=True)


def _var_normalize(d, axis=0, demean=True):
    """Subject level normalization over time course."""
    if demean:
        d = _demean(d, axis=axis)

    # Compute std over time quickly using SVD method
    u, s, vh = svds(d, k=np.minimum(30, d.shape[0] - 1))
    vh[np.abs(vh) < 2.3 * np.std(vh)] = 0
    std_devs = np.maximum(np.std(d - u.dot(np.diag(s)).dot(vh), axis=axis), 1e-3)
    return d / std_devs


@_to_float32
def _flatten(d, t_axis=-1):
    """Flatten data."""
    return d.reshape(-1, d.shape[t_axis])


def _assert_affine(a, b):
    """Check if affine matrices are equal."""
    if not np.all(a.affine == b.affine):
        raise ValueError("Affine matrices of given images are not equal.")


def _merge_data(data_list, axis=0):
    """Merge data."""
    return np.concatenate(data_list, axis=axis)


@_to_float32
def _get_data(img, mask=None):
    """Get data from image.
    Returns (ts x voxel/vertex) array."""
    if isinstance(img, str):
        if img.endswith((".nii.gz", ".nii", ".func.gii", ".dtseries.nii")):
            img = nib.load(img)
        elif img.endswith(".npy"):
            img = np.load(img)
        else:
            raise ValueError("Input file type is not supported.")

    if isinstance(img, np.ndarray):
        return img

    if mask is None:
        if isinstance(img, nib.Cifti2Image):
            return img.get_fdata()
        elif isinstance(img, (nib.Nifti1Image, nib.Nifti2Image)):
            return _flatten(img.get_fdata()).T
        elif isinstance(img, nib.GiftiImage):
            return np.array([d.data for d in img.darrays])
        else:
            raise TypeError("Input image type is not supported.")
    else:
        _assert_affine(img, mask)
        return apply_mask(img, mask)


def _check_img_type(img):
    """Check image type."""
    if isinstance(img, nib.Cifti2Image):
        return "cifti"
    elif isinstance(img, (nib.Nifti1Image, nib.Nifti2Image)):
        return "nifti"
    elif isinstance(img, nib.GiftiImage):
        return "gifti"
    else:
        raise TypeError("Input image type is not supported.")


def _assert_symmetric(a, tol=1e-8):
    """Check if matrix is symmetric."""
    if not np.all(np.abs(a - a.T) < tol):
        raise ValueError("Matrix is not symmetric")


def _save_as_img(
    data,
    target_img,
    out_dir,
    name,
):
    # TODO: Test for GIFTI and CIFTI files.
    """Save data as image."""
    img_type = _check_img_type(target_img)

    if img_type == "cifti":
        nib.Cifti2Image(data, target_img.header.get_axis(0).xyz_transform).to_filename(
            op.join(out_dir, f"{name}.dtseries.nii")
        )
    elif img_type == "nifti":
        nib.Nifti1Image(data, target_img.affine).to_filename(
            op.join(out_dir, f"{name}.nii.gz")
        )
    elif img_type == "gifti":
        nib.GiftiImage(data, target_img.affine).to_filename(
            op.join(out_dir, f"{name}.func.gii")
        )
