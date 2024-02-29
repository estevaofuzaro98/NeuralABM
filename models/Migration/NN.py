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

class Migration_NN:
    def __init__(
            self,
            *,
            rng: np.random.Generator,
            h5group: h5.Group,
            neural_net: base.NeuralNet,
            write_every: int = 1,
            write_start: int = 1,
            training_data: torch.Tensor,
            loss_function: dict,
            device: str,
            **__,
    ):
        """Initialize the model instance with a previously constructed RNG and
        HDF5 group to write the output data to.

        :param rng (np.random.Generator): The shared RNG
        :param h5group (h5.Group): The output file group to write data to
        :param neural_net: The neural network
        :param write_every: write every nth epoch
        :param write_start: epoch at which to start writing
        :param training_data: net migration data
        :param loss_function (dict): the loss function to use
        """
        self._h5group = h5group
        self._rng = rng

        # Neural network
        self.neural_net = neural_net
        self.neural_net.optimizer.zero_grad()

        # Get the loss function
        self.loss_function = base.LOSS_FUNCTIONS[loss_function.get("name").lower()](
            loss_function.get("args", None), **loss_function.get("kwargs", {})
        )

        # Neural net loss
        self.current_loss: torch.Tensor = torch.inf * torch.ones(1, dtype=torch.float)

        # Get the training data, which is the net migration data with nans dropped. These are reinserted later
        self.nan_indices = torch.isnan(training_data).nonzero().flatten()
        self.training_data = training_data[~training_data.isnan()]

        self.N = len(self.training_data)

        # Store the current predictions
        self.current_T: torch.Tensor = torch.zeros(self.N, self.N)

        # Epochs processed
        self._time = 0
        self._write_every = write_every
        self._write_start = write_start
        self.device = device

        # --- Set up chunked dataset to store the state data in --------------------------------------------------------
        # Write the loss after every epoch
        self._dset_loss = self._h5group.create_dataset(
            "loss",
            (0,),
            maxshape=(None,),
            chunks=True,
            compression=3,
        )
        self._dset_loss.attrs["dim_names"] = ["epoch"]
        self._dset_loss.attrs["coords_mode__epoch"] = "start_and_step"
        self._dset_loss.attrs["coords__epoch"] = [write_start, write_every]

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

        # Prediction
        self._dset_prediction = self._h5group.create_dataset(
            "prediction",
            (0, len(training_data), len(training_data)),
            maxshape=(None, len(training_data), len(training_data)),
            chunks=True,
            compression=3,
        )
        self._dset_prediction.attrs["dim_names"] = ["epoch", "i", "j"]
        self._dset_prediction.attrs["coords_mode__epoch"] = "start_and_step"
        self._dset_prediction.attrs["coords__epoch"] = [write_start, write_every]
        self._dset_prediction.attrs["coords_mode__i"] = self._h5group["net_migration"].attrs["coords_mode__i"]
        self._dset_prediction.attrs["coords__i"] = self._h5group["net_migration"].attrs["coords__i"]
        self._dset_prediction.attrs["coords_mode__j"] = self._h5group["net_migration"].attrs["coords_mode__i"]
        self._dset_prediction.attrs["coords__j"] = self._h5group["net_migration"].attrs["coords__i"]

        # Net migration
        self._dset_pred_net_migration = self._h5group.create_dataset(
            "pred_net_migration",
            (0, len(training_data)),
            maxshape=(None, len(training_data)),
            chunks=True,
            compression=3,
        )
        self._dset_pred_net_migration.attrs["dim_names"] = ["epoch", "i"]
        self._dset_pred_net_migration.attrs["coords_mode__epoch"] = "start_and_step"
        self._dset_pred_net_migration.attrs["coords__epoch"] = [write_start, write_every]
        self._dset_pred_net_migration.attrs["coords_mode__i"] = self._h5group["net_migration"].attrs["coords_mode__i"]
        self._dset_pred_net_migration.attrs["coords__i"] = self._h5group["net_migration"].attrs["coords__i"]

    def epoch(self, *, iteration: int = None):
        """Trains the model for a single epoch.

        :param iteration: (optional) iteration count (for debugging purposes)
        """

        # Make a prediction
        prediction = torch.reshape(
            self.neural_net(
                torch.rand(self.neural_net.input_dim, requires_grad=True)
            ),
            (self.N, self.N),
        )

        # Calculate the loss
        loss = (

                # Net migration agrees with data
                self.loss_function(torch.sum(prediction - torch.transpose(prediction, 0, 1), dim=0), self.training_data)

                # Trace is zero
                + torch.trace(prediction)

                # Total should sum to zero
                + torch.sum(prediction - torch.transpose(prediction, 0, 1))
        )
        loss.backward()

        self.neural_net.optimizer.step()
        self.neural_net.optimizer.zero_grad()

        # Write the data
        if loss < self.current_loss:
            self.current_loss = loss.detach()
            self.current_prediction = prediction.detach()

        self._time += 1
        # self.write_data()

    def write_data(self):
        """Write the current state (loss and parameter predictions) into the state dataset.

        In the case of HDF5 data writing that is used here, this requires to
        extend the dataset size prior to writing; this way, the newly written
        data is always in the last row of the dataset.
        """
        if self._time >= self._write_start and (self._time % self._write_every == 0):

            # Write the loss
            self._dset_loss.resize(self._dset_loss.shape[0] + 1, axis=0)
            self._dset_loss[-1] = self.current_loss

            # Write the prediction by reinserting Nan rows and columns at the missing entries
            self._dset_prediction.resize(self._dset_prediction.shape[0] + 1, axis=0)
            _pred = self.current_prediction
            for idx in self.nan_indices:

                # Add a NaN row
                _pred = torch.cat([_pred[:idx], torch.nan * torch.ones(1, _pred.size()[1]), _pred[idx:]])

                # Add a NaN column
                _pred = torch.cat([_pred.transpose(0, 1)[:idx],
                                   torch.nan * torch.ones(1, _pred.transpose(0, 1).size()[1]),
                                   _pred.transpose(0, 1)[idx:]]).transpose(0, 1)

            self._dset_prediction[-1, :] = _pred

            # Write predicted net migration
            self._dset_pred_net_migration.resize(self._dset_pred_net_migration.shape[0]+1, axis=0)
            self._dset_pred_net_migration[-1, :] = torch.nansum(
                _pred - torch.transpose(_pred, 0, 1),
                dim=0
            )
