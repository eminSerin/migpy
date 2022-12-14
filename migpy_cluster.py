import json
import os.path as op
from collections import OrderedDict

import click
from migpy import migpy


def migpy_cluster(batch_file, out_dir, step, dPCA_int, batch_num, mask=None):
    """Run MIGP in cluster.

    See also: migpy.migpy for parameters."""
    # Check output directory
    if not op.exists(out_dir):
        raise FileNotFoundError("Output directory does not exist.")

    # Check batch file
    if not op.exists(batch_file):
        raise FileNotFoundError("Batch file does not exist.")

    batches = json.load(open(batch_file, "r"), object_pairs_hook=OrderedDict)

    input_files = batches[f"step_{step}"][f"s{step}_migpy_batch_{batch_num}"]

    # Check files in input_files
    for f in input_files:
        if not op.exists(f):
            raise FileNotFoundError(f"File {f} does not exist.")

    print(f"Batch {batch_num} starts running...")
    return migpy(input_files, out_dir, step, mask, dPCA_int, batch_num)


@click.command()
@click.argument("batch_file", type=click.Path(exists=True), help="Batch file.")
@click.argument("out_dir", type=click.Path(exists=True), help="Output directory.")
@click.argument("step", type=int, help="Step number.")
@click.argument("dPCA_int", type=int, help="Number of internal dimensions to reduce to")
@click.argument("batch_num", type=int, help="Batch number.")
@click.option(
    "--mask", type=click.Path(exists=True), help="Path to mask file.", default=None
)
def main(batch_file, out_dir, step, dPCA_int, batch_num, mask):
    if mask == "":
        mask = None
    return migpy_cluster(batch_file, out_dir, step, mask, dPCA_int, batch_num)