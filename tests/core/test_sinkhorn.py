import sys
from os.path import dirname as up
import torch
import importlib
from dantro._import_tools import import_module_from_path

from utopya.yaml import load_yml

sys.path.insert(0, up(up(up(__file__))))

utils = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include.utils")

# Load the test config
CFG_FILENAME = importlib.resources.files('tests') / 'cfgs/sinkhorn.yml'
test_cfg = load_yml(CFG_FILENAME)

sys.path.insert(0, up(up(up(__file__))))

sinkhorn = import_module_from_path(
    mod_path=up(up(up(__file__))), mod_str="include.sinkhorn"
)
IOT = import_module_from_path(
    mod_path=up(up(up(__file__))), mod_str="models.IOT"
)

def test_Sinkhorn():

    for _, config in test_cfg.items():

        _max_iter = 100
        _tol = 1e-5
        eps = config['epsilon']
        data = IOT.DataGeneration.generate_synthetic_data(config, epsilon=eps)
        m, n, err = sinkhorn.Sinkhorn(data["mu"], data["nu"], data["C"], epsilon=eps,
                                      tolerance=_tol, max_iter = _max_iter, DEBUG=True)
        mu_pred, nu_pred, T_pred = sinkhorn.marginals_and_transport_plan(m, n, data["C"], epsilon=eps)

        assert mu_pred.shape == data["mu"].shape
        assert nu_pred.shape == data["nu"].shape
        assert T_pred.shape == data["T"].shape

        assert torch.all(torch.abs(data["T"] - T_pred) < 1e-5)
        assert torch.all(torch.abs(mu_pred - data["mu"]) < 1e-5)
        assert torch.all(torch.abs(nu_pred - data["nu"]) < 1e-5)

        assert torch.all(err[-1] / torch.tensor([mu_pred.shape[0], nu_pred.shape[1]]) < _tol)
        assert err.shape[0] <= _max_iter



