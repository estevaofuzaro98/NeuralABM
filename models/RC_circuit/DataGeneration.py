import h5py as h5
import logging
import sys
import torch
from os.path import dirname as up
import numpy as np

sys.path.append(up(up(up(__file__))))

from dantro._import_tools import import_module_from_path

base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------------------------------------
# Data loading and generation utilities
# ----------------------------------------------------------------------------------------------------------------------

def generate_synthetic_data(cfg: dict, *, dt: float) -> torch.Tensor:
    """ Function that generates synthetic time series of length num_steps for T_in, T_out, Q_H, Q_O.

    :param cfg: configuration of data settings
    :param device: device to use for training
    :param dt: time differential for the numerical solver (Euler in this case)
    :return: torch.Tensor of the time series for T_in, T_out, Q_H, Q_O. Tensor has shape (4, num_steps)
    """

    # Draw an initial condition for the data using the prior defined in the config
    # Uses the random_tensor function defined in include.utils
    initial_condition = torch.Tensor(
        [base.random_tensor(cfg["initial_conditions"]["T_in"], size=(1,)),
         base.random_tensor(cfg["initial_conditions"]["T_out"], size=(1,)),
         base.random_tensor(cfg["initial_conditions"]["Q_H"], size=(1,)),
         base.random_tensor(cfg["initial_conditions"]["Q_O"], size=(1,))]
    )

    data = [initial_condition]

    # Generate some synthetic time series
    for i in range(cfg['num_steps']):

        # Solve the equation for T_in and generate a random time series for T_out and the Q values
        data.append(
            torch.Tensor([
                # T_in
                data[-1][0] + dt / cfg["C"] * ((data[-1][1] - data[-1][0]) / cfg["R"] + data[-1][2] + data[-1][-1]),

                #T_out: fluctuates randomly with variance 'T_out_std'
                torch.normal(data[-1][1], cfg["T_out_std"]),

                # Q_H: fluctuates randomly with variance 'Q_std'
                torch.normal(data[-1][2], cfg["Q_std"]),

                # Q_O: same as Q_H
                torch.normal(data[-1][3], cfg["Q_std"])
            ])
        )

    return torch.reshape(torch.stack(data), (len(data), 4, 1))


def get_RC_circuit_data(*, data_cfg: dict, h5group: h5.Group):
    """Returns the training data for the RC_circuit model. If a directory is passed, the
    data is loaded from that directory (not yet implemented). Otherwise, synthetic training data is generated
    by iteratively solving the temporal ODE system.

    :param data_cfg: dictionary of config keys
    :param h5group: h5.Group to write the training data to
    :return: torch.Tensor of training data

    """
    if "load_from_dir" in data_cfg.keys():
        with h5.File(data_cfg["load_from_dir"], "r") as f:
            data = torch.from_numpy(np.array(f["RC_circuit"]["RC_data"])).float()

    elif "synthetic_data" in data_cfg.keys():

        data = generate_synthetic_data(
            cfg=data_cfg["synthetic_data"],
            dt=data_cfg["dt"]
        )

    else:
        raise ValueError(
            f"You must supply one of 'load_from_dir' or 'synthetic data' keys!"
        )

    # Store the synthetically generated data in an h5 file
    dset = h5group.create_dataset(
        "RC_data",
        data.shape,
        maxshape=data.shape,
        chunks=True,
        compression=3,
        dtype=float,
    )

    dset.attrs["dim_names"] = ["time", "kind", "dim_name__0"]
    dset.attrs["coords_mode__time"] = "trivial"
    dset.attrs["coords_mode__kind"] = "values"
    dset.attrs["coords__kind"] = [
        "T_in",
        "T_out",
        "Q_H",
        "Q_O"
    ]
    dset.attrs["coords_mode__dim_name__0"] = "trivial"

    dset[:, :] = data

    return data

