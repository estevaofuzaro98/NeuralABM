import logging
from typing import Tuple
import sys
from os.path import dirname as up

from dantro._import_tools import import_module_from_path

sys.path.append(up(up(up(__file__))))

base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")
import h5py as h5
import numpy as np
import torch

log = logging.getLogger(__name__)


def generate_synthetic_data(
    cfg: dict = None, *, epsilon: float, **__
) -> dict:
    """
    Generates a synthetic cost matrix, transport map, and marginals. The marginals can be unbalanced by adding a
    random perturbation to the balanced marginals.

    :param cfg: configuration entry
    :param epsilon: regularisation parameter
    :returns: a dictionary of the cost matrix, transport plan, and marginals
    """

    # Structure of the cost matrix
    nw_structure: dict = cfg.get("network", "uniform")

    # Number of origin and destination zones
    M, N = cfg["M"], cfg["N"]

    # Banded l^p distance or random matrix
    if nw_structure['distribution'].lower() == "l1_distance":
        iota = 1e-2
        C = torch.tensor(
            [[iota + np.abs(i - j) for j in range(N)] for i in range(M)],
            dtype=torch.float,
        )
    elif nw_structure['distribution'].lower() == "l2_distance":
        iota = 1e-2
        C = torch.tensor(
            [[iota + np.abs(i - j)**2 for j in range(N)] for i in range(M)],
            dtype=torch.float,
        )
    else:
        # Random cost matrix
        C = base.random_tensor(nw_structure, size=(M, N))

    # Normalise the cost matrix row sums, if given
    if cfg.get('normalize_cost_rows'):
        C /= C.sum(dim=1, keepdim=True)

    # Get random Lagrange multipliers
    d1 = torch.diag(base.random_tensor(cfg["mu"], size=(M, )))
    d2 = torch.diag(base.random_tensor(cfg["nu"], size=(N, )))

    # Generate the transport plan
    T = torch.matmul(torch.matmul(d1, torch.exp(-C / epsilon)), d2)

    # Calculate the marginals, which have shapes (M, 1) and (1, N) respectively
    mu = T.sum(dim=1, keepdim=True)
    nu = T.sum(dim=0, keepdim=True)

    # Unbalance the marginals, if given
    mu = torch.normal(
        mu, cfg.get("unbalancing", {}).get("mu", 0) * torch.ones(mu.shape)
    ).abs()
    nu = torch.normal(
        nu, cfg.get("unbalancing", {}).get("nu", 0) * torch.ones(nu.shape)
    ).abs()

    return dict(C=C, T=T, mu=mu, nu=nu)


def get_data(*, data_cfg: dict, h5group: h5.Group, **kwargs) -> dict:
    """Returns the training data for the SIR model. If a directory is passed, the
    data is loaded from that directory. Otherwise, synthetic training data is generated, either from an ABM,
    or by iteratively solving the temporal ODE system.
    """
    if "load_from_dir" in data_cfg.keys():

        with h5.File(data_cfg["load_from_dir"]["path"], "r") as f:

            # Get the time selector, if provided. This index selects a specific frame of the data to learn.
            # If a sweep over ``time_isel`` is configured, the cost matrix for multiple frames can be learned
            time_isel = data_cfg["load_from_dir"].get("time_isel", None)

            # Scale the data, if given
            sf = data_cfg["load_from_dir"].get("scale_factor", 1.0)

            mu = sf * torch.from_numpy(np.array(f["IOT"]["mu"])).float().unsqueeze(-1)
            nu = sf * torch.from_numpy(np.array(f["IOT"]["nu"])).float().unsqueeze(-2)

            C = sf * torch.from_numpy(np.array(f["IOT"]["C"])).float() if "C" in f["IOT"].keys() else None
            T = sf * torch.from_numpy(np.array(f["IOT"]["T"])).float()

            data = dict(C=C, T=T, mu=mu, nu=nu) if C else dict(T=T, mu=mu, nu=nu)
            if time_isel is not None:
                data = dict((k, v[time_isel]) for k, v in data.items())
                data["time_isel"] = time_isel

            data.update(**f["IOT"]["T"].attrs)

    else:
        # Load synthetic data
        data = generate_synthetic_data(data_cfg["synthetic_data"], **kwargs)

    # Store the data in seperate datasets
    if "C" in data.keys():
        dset_C = h5group.create_dataset(
            "C",
            data["C"].shape,
            maxshape=data["C"].shape,
            chunks=True,
            compression=3,
            dtype=float,
        )
        dset_C.attrs["dim_names"] = ["i", "j"]
        dset_C.attrs["coords_mode__i"] = data.get("coords_mode__i", "trivial")
        if data.get("coords_mode__i", "trivial") == "values":
            dset_C.attrs["coords__i"] = data["coords__i"]
        dset_C.attrs["coords_mode__j"] = data.get("coords_mode__j", "trivial")
        if data.get("coords_mode__j", "trivial") == "values":
            dset_C.attrs["coords__j"] = data["coords__j"]

        dset_C[:, :] = data["C"]

    dset_T = h5group.create_dataset(
        "T",
        data["T"].shape,
        maxshape=data["T"].shape,
        chunks=True,
        compression=3,
        dtype=float,
    )
    dset_T.attrs["dim_names"] = ["i", "j"]
    dset_T.attrs["coords_mode__i"] = data.get("coords_mode__i", "trivial")
    if data.get("coords_mode__i", "trivial") == "values":
        dset_T.attrs["coords__i"] = data["coords__i"]
    dset_T.attrs["coords_mode__j"] = data.get("coords_mode__j", "trivial")
    if data.get("coords_mode__j", "trivial") == "values":
        dset_T.attrs["coords__j"] = data["coords__j"]
    dset_T[:, :] = data["T"]

    dset_mu = h5group.create_dataset(
        "mu",
        data["mu"].squeeze(-1).shape,
        maxshape=data["mu"].squeeze(-1).shape,
        chunks=True,
        compression=3,
        dtype=float,
    )
    dset_mu.attrs["dim_names"] = ["i"]
    dset_mu.attrs["coords_mode__i"] = data.get("coords_mode__i", "trivial")
    if data.get("coords_mode__i", "trivial") == "values":
        dset_mu.attrs["coords__i"] = data["coords__i"]
    dset_mu[:] = data["mu"].squeeze(-1)

    dset_nu = h5group.create_dataset(
        "nu",
        data["nu"].squeeze(-2).shape,
        maxshape=data["nu"].squeeze(-2).shape,
        chunks=True,
        compression=3,
        dtype=float,
    )
    dset_nu.attrs["dim_names"] = ["i"]
    dset_nu.attrs["coords_mode__i"] = data.get("coords_mode__i", "trivial")
    if data.get("coords_mode__i", "trivial") == "values":
        dset_nu.attrs["coords__i"] = data["coords__i"]
    dset_nu[:] = data["nu"].squeeze(-2)

    # Return the tensors, which have shapes (M, N) (C), (M, N) (T), (M, 1) (mu), and (1, N) (nu) respectively.
    # The time dimension is dropped, since the model only learns the cost matrix for a single time frame
    return data
