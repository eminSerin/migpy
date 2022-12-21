# Emin Serin
"""
Python port of MIGP.

It is purely on Python and does not require MATLAB, or FSL.

This library is highly inspired from: 
    - https://git.fmrib.ox.ac.uk/seanf/pymigp
    - Original MATLAB scripts to compute MIGP on HCP dataset.
        See: https://www.humanconnectome.org/storage/app/media/documentation/s1200/HCP1200-DenseConnectome+PTN+Appendix-July2017.pdf

Reference:
----------
    Smith, S.M., Hyvärinen, A., Varoquaux, G., Miller, K.L., & Beckmann, C.F. (2014). Group-PCA for
    very large fMRI datasets. Neuroimage, 101, 738-749.
"""

import gc
import json
import os
import os.path as op
from collections import OrderedDict

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse.linalg import eigsh
from tqdm import tqdm

from utils import _assert_symmetric, _demean, _get_data, _merge_data, _var_normalize


def _reduce_timecourse(A, dPCA_int, var_normalize=False, demean=False):
    """Reduce timecourse of given images.

    Parameters
    ----------
    A : np.ndarray
        Data matrix of shape (n_timepoints, n_voxels).
    dPCA_int : np.int
        Number of internal dimensions to reduce to.
    var_normalize : bool, optional
        Subject specific variance normalization over time course,
        by default False
    demean : bool, optional
        Demeaning over time course., by default False

    Returns
    -------
    np.ndarray
        Reduced data matrix of shape (dPCA_int, n_voxels).
    """

    if var_normalize:
        A = _var_normalize(A, axis=0, demean=True)
    if demean:
        A = _demean(A, axis=0)
    if A.shape[0] - 10 > dPCA_int:
        A_cov = A.dot(A.T)

        # Compute eigenvalues and eigenvectors
        # Sparse solution (better with large matrices)
        _assert_symmetric(A_cov)
        _, v = eigsh(A_cov, k=dPCA_int, which="LM")

        # Return spatial eigenvectors
        return v.T.dot(A)
    else:
        Warning(
            "Image has less timepoints than dPCA_int. Therefore, skipping decomposition!"
        )
        return A


def merge_migpy(migp_files, out_dir, dPCA_int=4700, dPCA_out=None):
    """Merge MIGP files

    This function must be run after all batch steps have been processed.
    This is only for final merging!

    Parameters
    ----------
    migp_files : np.ndarray
        MIGP files to merge.
        MIGP files contain the reduced timecourse of each image.
    out_dir : str
        Output directory.
    dPCA_int : int, optional
        Number of internal dimensions to reduce to, by default 4700
    dPCA_out : _type_, optional
        Number of final dimensions to reduce to, by default None
        If dPCA_out < dPCA_int, then the time courses with the
        largest eigenvalues are selected.

    Returns:
    --------
        np.ndarray:
            Final merged MIGP file.
            This file is now ready for further processing (e.g., ICA).
    """
    if dPCA_out is None:
        dPCA_out = dPCA_int

    print("Merging MIGP files...")
    for i, migp_file in tqdm(enumerate(migp_files)):
        data = _get_data(migp_file)
        if i == 0:
            W = _reduce_timecourse(data, dPCA_int)
        else:
            W = _merge_data([W, data])
            W = _reduce_timecourse(W, dPCA_int)
    W = W[:dPCA_out, :]
    print("Saving MIGP...")
    np.save(op.join(out_dir, f"migp_dPCA{dPCA_out}.npy"), W)


def migpy(
    img_list,
    out_dir,
    step=None,
    mask=None,
    dPCA_int=4700,
    batch_num=None,
    verbose=False,
):
    """Python adaptation of MIGP.

    IMPORTANT: Step number is important for batch processing as it determines
    whether input images are raw or reduced. Please see the parameters below.

    Parameters
    ----------
    img_list : list
        List of images to process.
    out_dir : str
        Output directory.
    step : int
        Step number.
        !!!This is important, as it determines
        whether the input images are raw or already reduced!!!
        If step = 0, then then input images are raw,
        if step > 0, then the input images are already reduced.
    mask : str or Nibabel instance, optional
        Mask file, by default None
        Using a mask is strongly recommended as
        it will speed up the processing and removes
        non-brain voxels.
    dPCA_int : int, optional
        Number of internal dimensions to reduce to, by default 4700
    batch_num : int, optional
        Bactch number, by default None
        If the dataset is split into batches to reduce memory usage,
        and for parallelization, then this number is used to
        save the intermediate MIGP files.


    Raises
    ------
    ValueError
        If provided mask is not Nifti, Cifti or GIfTI.
    """
    # Mask data
    if mask is not None:
        # TODO: Add support for GIFTI mask
        if isinstance(mask, str):
            mask = nib.load(mask)
        if isinstance(mask, (nib.Nifti1Image, nib.Nifti2Image, nib.Cifti2Image)):
            mask_out = op.join(out_dir, "migpy_mask.nii.gz")
        elif isinstance(mask, nib.GiftiImage):
            mask_out = op.join(out_dir, "migpy_mask.func.gii")
        else:
            raise ValueError("Mask type is not supported.")

        if not op.exists(mask_out):
            mask.to_filename(mask_out)
        else:
            Warning(
                "Mask already exists. Skipping saving mask into the working directory."
            )
    else:
        Warning("No mask provided. Using all voxels. This may take a long time...")

    batch_dir = op.join(out_dir, "batches")
    if not op.exists(batch_dir):
        os.makedirs(batch_dir)

    # Reduce timecourse
    if verbose:
        print("Reducing timecourse...")
    for i, img in tqdm(enumerate(img_list)):
        if "migpy_batch" not in img:
            ## If input image is a raw image, not a batch file,
            ## then standardize!
            data = _demean(_var_normalize(_get_data(img, mask)))
        else:
            data = _get_data(img, mask)
            Warning(
                "Not standardizing data. Raw images (i.e., non migp batches) must be standardized!."
            )
        if i == 0:
            W = _reduce_timecourse(data, dPCA_int)
        else:
            W = _merge_data([W, data])
            W = _reduce_timecourse(W, dPCA_int)
        gc.collect()

    if verbose:
        print("Saving MIGP...")
    if batch_num is not None:
        out_file = op.join(batch_dir, f"s{step}_migpy_batch_{batch_num}.npy")
    else:
        out_file = op.join(batch_dir, f"s{step}_migpy.npy")
    np.save(out_file, W)


def get_batch(file_list, batch_size, work_dir):
    """Generates hierarchical dictionary of batches.

    Batches are a way to parallelize the computation of the MIGP.
    Batch size must be decided based on the available memory. e.g.,
    if want to get 4700 x 235375 matrix (brain-masked 2-mm MNI volume
    image), then you will need 4700 x 235375 x 8 bytes = 1.1 GB of memory
    for output data and approx. 10GB for loading the input data. Therefore,
    we suggest at least 12GB of memory for batch for brain images of such
    size.

    You should also consider the length of time series in each image,
    and desired number of internal dimensions. e.g., if you have
    1200 time points and want to get 4700 internal dimensions, then
    you set batch_size to at least 4. Otherwise, dimension reduction
    in the first step would be useless as it only reduces from 4800
    to 4700.


    Parameters
    ----------
    file_list : list of str
        List of file paths.
    batch_size : int
        Number of files in a batch.
        Batch size must be decided based on the available memory.
    work_dir : str
        Where to save the batches.

    Returns
    -------
    .json
        Saves a .json file with the batches.
    batches : OrderedDict
        Hierarchical dictionary of batches.
    """
    batches = OrderedDict()
    step = 0

    if not isinstance(file_list, list):
        try:
            if op.isfile(file_list):
                file_list = pd.read_csv(file_list, header=None)[0].tolist()
        except TypeError:
            raise TypeError(
                "file_list must be a list of file paths, or a path to a csv file."
            )
    n_sub = len(file_list)
    while batch_size < n_sub:
        tmp_chunk = OrderedDict()
        if step == 0:
            for i, x in enumerate(range(0, len(file_list), batch_size)):
                tmp_chunk[f"s{step}_migpy_batch_{i}"] = file_list[x : x + batch_size]
        else:
            for i, x in enumerate(range(0, len(file_list), batch_size)):
                tmp_chunk[f"s{step}_migpy_batch_{i}"] = [
                    op.join(op.dirname(f), "batches", op.basename(f) + ".npy")
                    for f in file_list[x : x + batch_size]
                ]
        batches[f"step_{step}"] = tmp_chunk
        step += 1
        file_list = [op.join(work_dir, x) for x in list(tmp_chunk.keys())]
        n_sub = len(file_list)

    json.dump(batches, open(op.join(work_dir, "batches.json"), "w"), indent=4)
    return batches


def migp_parallel(
    file_list, out_dir, mask=None, dPCA_int=4700, batch_size=10, n_jobs=1
):
    """Run MIGP parallel.

    This will parallelize MIGP by dividing
    the list of brain images into batches
    and run each batch in parallel.

    Parameters
    ----------
    file_list : list
        List of dicectories for all brain images to process.
    out_dir : str
        Output directory.
    mask : str or Nibabel instance, optional
        Brain mask to use, by default None
    dPCA_int : int, optional
        Number of internal dimensions to reduce to, by default 4700
    batch_size : int, optional
        Number of files in a batch, by default 10
    n_jobs : int, optional
        N cores to utilize, by default 1
        Depending on datasize MIGP requires
        at least 10GB of memory per core.

    See also migpy.migpy
    """
    # Use batches!
    batches = get_batch(file_list, batch_size, out_dir)
    for s, step in enumerate(batches.keys()):
        print(f"Processing step {s}/{len(batches.keys())}")
        Parallel(n_jobs=n_jobs)(
            delayed(migpy)(batches[step][batch], out_dir, s, mask, dPCA_int, b)
            for b, batch in enumerate(tqdm(batches[step].keys()))
        )
