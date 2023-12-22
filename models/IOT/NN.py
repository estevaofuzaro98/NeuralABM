import sys
from os.path import dirname as up

import h5py as h5
import numpy as np
import torch
from dantro import logging
from dantro._import_tools import import_module_from_path

sys.path.append(up(up(__file__)))
sys.path.append(up(up(up(__file__))))

Covid = import_module_from_path(mod_path=up(up(__file__)), mod_str="Covid")
base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------------------------------------
# Model implementation
# ----------------------------------------------------------------------------------------------------------------------

class IOT_NN:
    def __init__(
            self,
            *,
            rng: np.random.Generator,
            h5group: h5.Group,
            neural_netC: base.NeuralNet,
            neural_netM: base.NeuralNet,
            write_every: int = 1,
            write_start: int = 1,
            training_data: dict,
            epsilon: float,
            loss_function: dict,
            device: str,
            **__,
    ):
        """Initialize the model instance with a previously constructed RNG and
        HDF5 group to write the output data to.

        :param rng (np.random.Generator): The shared RNG
        :param h5group (h5.Group): The output file group to write data to
        :param neural_netC: The neural network for the cost matrix
        :param neural_netM: The neural network for the marginals
        :param write_every: write every nth epoch
        :param write_start: epoch at which to start writing
        :param training_data: dictionary training data to use, containing the marginals and transport map
        :param epsilon: regularisation parameter
        :param loss_function (dict): the loss function to use
        """
        self._h5group = h5group
        self._rng = rng

        # Neural network for the cost matrix
        self.neural_netC = neural_netC
        self.neural_netC.optimizer.zero_grad()

        # Neural network for the marginals
        self.neural_netM = neural_netM
        if self.neural_netM:
            self.neural_netM.optimizer.zero_grad()

        # Get the loss function
        self.loss_function = base.LOSS_FUNCTIONS[loss_function.get("name").lower()](
            loss_function.get("args", None), **loss_function.get("kwargs", {})
        )

        # Neural net loss
        self.current_loss: torch.Tensor = torch.zeros(3, dtype=torch.float)

        # Get the training data
        self.mu, self.nu = training_data["mu"], training_data["nu"]
        self.T = training_data["T"]

        # Origin zone and destination zone sizes
        self.N: int = self.T.shape[0]
        self.M: int = self.T.shape[1]
        self.epsilon: torch.tensor = torch.tensor(epsilon, dtype=torch.float)

        # Store the current predictions
        self.current_marginals: torch.Tensor = torch.zeros(self.M + self.N)
        self.current_C: torch.Tensor = torch.zeros(self.M, self.N)
        self.current_T: torch.Tensor = torch.zeros(self.M, self.N)

        # Epochs processed
        self._time = 0
        self._write_every = write_every
        self._write_start = write_start
        self.device = device

        # --- Set up chunked dataset to store the state data in --------------------------------------------------------
        # Write the loss after every epoch
        self._dset_loss = self._h5group.create_dataset(
            "loss",
            (0, 3),
            maxshape=(None, 3),
            chunks=True,
            compression=3,
        )
        self._dset_loss.attrs["dim_names"] = ["epoch", "kind"]
        self._dset_loss.attrs["coords_mode__epoch"] = "start_and_step"
        self._dset_loss.attrs["coords__epoch"] = [write_start, write_every]
        self._dset_loss.attrs["coords_mode__kind"] = "values"
        self._dset_loss.attrs["coords__kind"] = ["C", "mu", "nu"]

        # Write the computation time of every epoch
        self._dset_time = self._h5group.create_dataset(
            "computation_time",
            (0,),
            maxshape=(None,),
            chunks=True,
            compression=3,
        )
        self._dset_time.attrs["dim_names"] = ["epoch"]
        self._dset_time.attrs["coords_mode__epoch"] = "trivial"

        # Marginals
        self._dset_mu = self._h5group.create_dataset(
            "predicted_mu",
            (0, self.M),
            maxshape=(None, self.M),
            chunks=True,
            compression=3,
        )
        self._dset_mu.attrs["dim_names"] = ["epoch", "i"]
        self._dset_mu.attrs["coords_mode__epoch"] = "start_and_step"
        self._dset_mu.attrs["coords__epoch"] = [write_start, write_every]
        self._dset_mu.attrs["coords_mode__i"] = "trivial"

        self._dset_nu = self._h5group.create_dataset(
            "predicted_nu",
            (0, self.N),
            maxshape=(None, self.N),
            chunks=True,
            compression=3,
        )
        self._dset_nu.attrs["dim_names"] = ["epoch", "i"]
        self._dset_nu.attrs["coords_mode__epoch"] = "start_and_step"
        self._dset_nu.attrs["coords__epoch"] = [write_start, write_every]
        self._dset_nu.attrs["coords_mode__i"] = "trivial"

        # Cost matrix
        self._dset_C = self._h5group.create_dataset(
            "predicted_C",
            (0, self.M, self.N),
            maxshape=(None, self.N, self.M),
            chunks=True,
            compression=3,
        )
        self._dset_C.attrs["dim_names"] = ["epoch", "i", "j"]
        self._dset_C.attrs["coords_mode__epoch"] = "start_and_step"
        self._dset_C.attrs["coords__epoch"] = [write_start, write_every]
        self._dset_C.attrs["coords_mode__i"] = "trivial"
        self._dset_C.attrs["coords_mode__j"] = "trivial"

        # Transport plan
        self._dset_T = self._h5group.create_dataset(
            "predicted_T",
            (0, self.M, self.N),
            maxshape=(None, self.N, self.M),
            chunks=True,
            compression=3,
        )
        self._dset_T.attrs["dim_names"] = ["epoch", "i", "j"]
        self._dset_T.attrs["coords_mode__epoch"] = "start_and_step"
        self._dset_T.attrs["coords__epoch"] = [write_start, write_every]
        self._dset_T.attrs["coords_mode__i"] = "trivial"
        self._dset_T.attrs["coords_mode__j"] = "trivial"

    def epoch(self, *, sinkhorn_kwargs: dict):
        """Trains the model for a single epoch.

        :param sinkhorn_kwargs: passed to the numerical solver (Sinkhorn algorithm)

        """

        # Get a sample for the cost matrix
        C_pred = torch.reshape(
            self.neural_netC(
                torch.rand(self.neural_netC.input_dim, requires_grad=True)
            ),
            (self.M, self.N),
        )

        # Get a sample for the marginals
        marginals = self.neural_netM(
            torch.rand(self.neural_netM.input_dim, requires_grad=True)
        )
        mu_pred, nu_pred = torch.reshape(marginals[0: self.M], (-1, 1)), torch.reshape(
            marginals[self.M: self.M + self.N], (1, -1)
        )

        # Get the marginals from the predicted cost matrix
        m, n = base.Sinkhorn(
            mu_pred.clone().detach(),
            nu_pred.clone().detach(),
            C_pred,
            epsilon=self.epsilon,
            **sinkhorn_kwargs,
        )
        _, _, T_pred = base.marginals_and_transport_plan(m, n, C_pred, epsilon=self.epsilon)

        # Train the cost NN to match both the observed transport plan and marginals
        lossC = (
                self.loss_function(T_pred, self.T)
                + self.loss_function(T_pred.sum(dim=1, keepdim=True), self.mu)
                + self.loss_function(T_pred.sum(dim=0, keepdim=True), self.nu)
                + self.loss_function(torch.abs(C_pred).sum(dim=1, keepdim=True), torch.ones(self.M))
        )

        lossC.backward()
        self.neural_netC.optimizer.step()
        self.neural_netC.optimizer.zero_grad()

        # Train the marginal NN to match the observed marginals and the marginals from the
        # predicted cost matrix
        lossM = (
                self.loss_function(mu_pred, self.mu)
                + self.loss_function(nu_pred, self.nu)
                + self.loss_function(mu_pred, self.T.sum(dim=1, keepdim=True))
                + self.loss_function(nu_pred, self.T.sum(dim=0, keepdim=True))
        )
        lossM.backward()
        self.neural_netM.optimizer.step()
        self.neural_netM.optimizer.zero_grad()

        # Write the data
        self.current_loss = torch.tensor(
            [
                self.loss_function(T_pred, self.T),
                self.loss_function(mu_pred, self.mu),
                self.loss_function(nu_pred, self.nu),
            ]
        )
        self.current_marginals = torch.cat(
            [mu_pred.detach().flatten(), nu_pred.detach().flatten()]
        )
        self.current_C = C_pred.detach()
        self.current_T = T_pred.detach()

        self._time += 1
        self.write_data()

    def write_data(self):
        """Write the current state (loss and parameter predictions) into the state dataset.

        In the case of HDF5 data writing that is used here, this requires to
        extend the dataset size prior to writing; this way, the newly written
        data is always in the last row of the dataset.
        """
        if self._time >= self._write_start and (self._time % self._write_every == 0):

            # Write the loss
            self._dset_loss.resize(self._dset_loss.shape[0] + 1, axis=0)
            self._dset_loss[-1, :] = self.current_loss

            # Write the marginals
            self._dset_mu.resize(self._dset_mu.shape[0] + 1, axis=0)
            self._dset_mu[-1, :] = self.current_marginals[0: self.M]
            self._dset_nu.resize(self._dset_nu.shape[0] + 1, axis=0)
            self._dset_nu[-1, :] = self.current_marginals[
                                            self.M: self.M + self.N
                                            ]

            # Write the cost matrix
            self._dset_C.resize(self._dset_C.shape[0] + 1, axis=0)
            self._dset_C[-1, :] = self.current_C

            # Write the transport plan
            self._dset_T.resize(self._dset_T.shape[0] + 1, axis=0)
            self._dset_T[-1, :] = self.current_T
