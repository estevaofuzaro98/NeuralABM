import torch
from typing import Sequence, Union, Tuple

def Sinkhorn(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    *,
    epsilon: Union[torch.Tensor, float] = 1,
    tolerance: float = 1e-9,
    max_iter: int = 100,
    requires_gradient: bool = True,
    DEBUG: bool = False
) -> Sequence[torch.Tensor]:

    """Basic Sinkhorn algorithm. The Sinkhorn algorithm is a fixed-point iteration that, given marginal constraints
    ``a`` and ``b`` and a cost matrix ``C`` finds the optimal transport solution ``T``. The iteration is conducted until
    the algorithm has found a steady state or reached a maximum number of iterations, whichever comes sooner.

    :param a: first target marginal, of dimension (M, 1)
    :param b: second target marginal, of dimension (1, N)
    :param C: cost matrix, of dimension (M, N)
    :param epsilon: epsilon value for the regularisation
    :param tolerance: cutoff criterion to use. The algorithm terminates if the difference between successive guesses
        falls below the tolerance
    :param max_iter: maximum number of iterations to perform. The algorithm returns once the maximum number of iterations
        has been reached, regardless of outcome.
    :param requires_gradient: whether the marginal constraints require differentiation
    :param DEBUG: flag that can be set to return the
        algorithm will return the difference between successive guesses
    :returns: tuple of marginal constraints
    """

    # Initial values: vector of ones
    _m = torch.ones_like(a)
    _m.requires_gradient = requires_gradient
    _n = torch.ones_like(b)
    _n.requires_gradient = requires_gradient

    # Exponential of cost matrix. The transport plan is given by diag(m) * T * diag(n).
    _T = torch.exp(-C / epsilon)

    # Ascertain convergence
    _has_converged: bool = False

    if DEBUG:
        _err = []
    # Iterate the Sinkhorn algorithm until the convergence criterion is reached or the algorithm is terminated.
    _iter = 0
    while not _has_converged:
        _n_prev = _n.clone().detach()
        _m_prev = _m.clone().detach()
        _n = b / torch.matmul(_T.transpose(0, 1), _m).transpose(0, 1)
        _m = a / torch.matmul(_T, _n.transpose(0, 1))
        _iter += 1

        # Calculate the difference between successive guesses
        _m_diff, _n_diff = torch.abs(_m_prev - _m), torch.abs(_n_prev - _n)

        # Check if convergence criteria have been met
        _has_converged = (torch.all(_n_diff < tolerance) and torch.all(_m_diff < tolerance)) or _iter > max_iter
        if DEBUG:
            _err.append(torch.tensor([_m_diff.sum(), _n_diff.sum()]))
    if DEBUG:
        return _m, _n, torch.stack(_err)
    else:
        return _m, _n


def marginals_and_transport_plan(mu: torch.Tensor,
                                 nu: torch.Tensor,
                                 C: torch.Tensor,
                                 *,
                                 epsilon: Union[torch.Tensor, float]
                                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Calculates the marginals and transport plan from a cost matrix and marginal constraint diagonals
    (Sinkhorn output).

    :param mu: first marginal constraint
    :param nu: second marginal constraint
    :param C: cost matrix
    :param epsilon: entropy regularisation
    :returns: tuple of marginals and transport plan
    """

    # Predicted transport plan
    T_pred = torch.matmul(
        torch.matmul(torch.diag(mu.flatten()), torch.exp(- C / epsilon)), torch.diag(nu.flatten())
    )

    # Predicted marginals
    mu_pred, nu_pred = T_pred.sum(dim=1, keepdim=True), T_pred.sum(dim=0, keepdim=True)

    return mu_pred, nu_pred, T_pred
