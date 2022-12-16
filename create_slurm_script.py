"""Generates Slurm script for running MIGP on a cluster.
Adapted from https://github.com/HALFpipe/HALFpipe/blob/main/halfpipe/cluster.py"""

import os
import os.path as op
import sys

import click

from migpy import get_batch


def create_slurm_script(
    file_list,
    out_dir,
    mask=None,
    dPCA_int=4700,
    batch_size=5,
    mem_per_task=16384,
    cpu_per_task=2,
    task_time="24:00:00",
):
    """Create a slurm script for running MIGP on a cluster.

    Parameters
    ----------
    file_list : list
        List of file paths.
    out_dir : str
        Output directory.
    mask : str or Nibabel instance, optional
        Brain mask, by default None
    dPCA_int : int, optional
        Number of internal dimensions to reduce to, by default 4700
    batch_size : int, optional
        Number of files in each batch, by default 5
    mem_per_task : int, optional
        Memory to allocate per each cluster node, by default 16384
    cpu_per_task : int, optional
        Number of cpus to allocate per each cluster node, by default 2
    task_time : str, optional
        How long to allocate cluster node, by default "24:00:00"

    Raises
    ------
    FileNotFoundError
        If output directory does not exists.
    """
    if mask is None:
        mask = ""

    if not op.exists(out_dir):
        raise FileNotFoundError(f"{out_dir} does not exist.")

    slurm_dir = op.join(out_dir, "slurm")
    if not op.exists(slurm_dir):
        os.makedirs(slurm_dir)

    batches = get_batch(file_list, batch_size, out_dir)

    slurm_config = """#!/bin/bash
    
#SBATCH --job-name=migpy
#SBATCH --output={out_dir:s}/slurm.out/migpy_%j.out
#SBATCH --time={task_time:s}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpu_per_task:d}
#SBATCH --mem={mem_per_task:d}
#SBATCH --array=1-{n_batches:d}

{python_exec:s} \\
    {migpy_func:s} \\
    {batch_file:s} \\
    {out_dir:s} \\
    {s:d} \\
    {dPCA_int:d} \\
    ${{{array_index_var}}}
    --mask {mask:s}
"""
    if not op.exists(op.join(out_dir, "slurm.out")):
        os.makedirs(op.join(out_dir, "slurm.out"))

    for s, step in enumerate(batches.keys()):
        n_batches = len(list(batches[step].keys()))
        args = dict(
            migpy_func=f"{os.path.join(__file__, 'migpy_cluster.py')}",
            array_index_var="SLURM_ARRAY_TASK_ID",
            task_time=task_time,
            cpu_per_task=cpu_per_task,
            mem_per_task=mem_per_task,
            n_batches=n_batches,
            python_exec=sys.executable,
            batch_file=op.join(out_dir, "batches.json"),
            s=s,
            out_dir=out_dir,
            mask=mask,
            dPCA_int=dPCA_int,
        )

        # print(slurm_config.format(**args))
        with open(op.join(slurm_dir, f"slurm_step_{s}.sh"), "w") as f:
            f.write(slurm_config.format(**args))


@click.command()
@click.argument("file_list", type=click.Path(exists=True))
@click.argument("out_dir", type=click.Path(exists=True))
@click.option(
    "--dPCA_int",
    type=int,
    default=4700,
    help="Number of internal dimensions to reduce to.",
)
@click.option("--batch_size", type=int, default=8, help="Number of files in a batch.")
@click.option(
    "--mem_per_task", type=int, default=16384, help="Memory to use for each task (mb)."
)
@click.option(
    "--cpu_per_task",
    type=int,
    default=2,
    help="Number of cores to utilize for each task.",
)
@click.option(
    "--task_time", type=str, default="24:00:00", help="Time to run each task."
)
@click.option(
    "--mask", type=click.Path(exists=True), default=None, help="Path to brain mask."
)
def main(
    file_list,
    out_dir,
    dpca_int,
    batch_size,
    mem_per_task,
    cpu_per_task,
    task_time,
    mask,
):
    return create_slurm_script(
        file_list,
        out_dir,
        mask,
        dpca_int,
        batch_size,
        mem_per_task,
        cpu_per_task,
        task_time,
    )


if __name__ == "__main__":
    main()
