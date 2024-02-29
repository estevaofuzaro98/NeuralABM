import logging
import sys
from os.path import dirname as up

from dantro._import_tools import import_module_from_path

sys.path.append(up(up(up(__file__))))

base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")
import h5py as h5
import numpy as np
import torch

log = logging.getLogger(__name__)


def load_data(*, data_cfg: dict, h5group: h5.Group) -> torch.Tensor:
    """Returns the training data for the SIR model. If a directory is passed, the
    data is loaded from that directory. Otherwise, synthetic training data is generated, either from an ABM,
    or by iteratively solving the temporal ODE system.
    """

    # Get the time selector, if provided. This index selects a specific frame of the data to learn.
    # If a sweep over ``time_isel`` is configured, the cost matrix for multiple frames can be learned
    time_isel = data_cfg["load_from_dir"]["time_isel"]

    # Scale the data, if given
    sf = data_cfg["load_from_dir"].get("scale_factor", 1.0)

    # Load the data
    with h5.File(data_cfg["load_from_dir"]["path"], "r") as f:
        mu = sf * torch.from_numpy(np.array(f["Migration"]["net_migration"])).float()[time_isel]
        # Save to a h5 file
        dset = h5group.create_dataset(
            "net_migration",
            mu.shape,
            maxshape=mu.shape,
            chunks=True,
            compression=3,
            dtype=float,
        )
        dset.attrs["dim_names"] = ["i"]
        dset.attrs["coords_mode__i"] = f["Migration"]["net_migration"].attrs["coords_mode__i"]
        dset.attrs["coords__i"] = f["Migration"]["net_migration"].attrs["coords__i"]
        dset[:] = mu

    return mu

