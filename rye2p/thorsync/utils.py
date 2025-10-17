"""
Utility functions for working with ThorSync .h5 files.
"""
import h5py


def read_gctr(h5_file):
    """Get global counter from .h5 file"""
    with h5py.File(h5_file, 'r') as f:
        gctr = f['Global']['GCtr']
    return gctr


def preview_hdf5(h5_file):
    """Preview .h5 contents (datasets by group)

    Args:
        h5_file (Union[Path, str]): h5 file
    """

    with h5py.File(h5_file, 'r') as f:
        for grp in f.keys():
            print(grp)

            for item in f[grp]:
                print(f"\t{f[grp][item]}")
