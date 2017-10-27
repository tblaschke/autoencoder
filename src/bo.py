# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn import gaussian_process as gp


# from joblib import Parallel, delayed

def expected_improvement(x, gaussian_process, evaluated_loss, greater_is_better=False, n_params=1):
    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)

    scaling_factor = (-1) ** (not greater_is_better)

    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0

    return -1 * expected_improvement


def sample_next_point(acquisition_func, gaussian_process, evaluated_loss, greater_is_better=False,
                      bounds=(-2, 2), n_restarts=25, n_jobs=-1):
    best_x = None
    best_acquisition_value = 1
    n_params = bounds.shape[0]
    resl = []
    #   resl = Parallel(n_jobs=n_jobs)(delayed(minimize)(fun=acquisition_func, x0=starting_point, bounds=bounds, method='L-BFGS-B', args=(gaussian_process, evaluated_loss, greater_is_better, n_params)) for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)))
    for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):
        res = minimize(fun=acquisition_func,
                       x0=starting_point.reshape(1, -1),
                       bounds=bounds,
                       method='L-BFGS-B',
                       args=(gaussian_process, evaluated_loss, greater_is_better, n_params))
        resl.append(res)
    for res in resl:
        if res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x

    return best_x


def bayesian_optimisation(n_iters, sample_loss, bounds, x0=None, y0=None, n_pre_samples=5,
                          gp_params=None, random_search=False, alpha=1e-5, epsilon=1e-7, model=None, n_jobs=-1):
    x_list = []
    y_list = []

    n_params = bounds.shape[0]

    if x0 is None:
        for params in np.random.uniform(bounds[:, 0], bounds[:, 1], (n_pre_samples, bounds.shape[0])):
            x_list.append(params)
    else:
        for params in x0:
            x_list.append(params)

    if y0 is None:
        y_list = [sample_loss(params) for params in x_list]
    #        y_list = Parallel(n_jobs=n_jobs)(delayed(sample_loss)(params) for params in x_list)
    else:
        for params in y0:
            y_list.append(params)

    xp = np.array(x_list)
    yp = np.array(y_list)

    # Create the GP
    if model is None:
        if gp_params is not None:
            model = gp.GaussianProcessRegressor(**gp_params)
        else:
            kernel = gp.kernels.Matern()
            model = gp.GaussianProcessRegressor(kernel=kernel,
                                                alpha=alpha,
                                                n_restarts_optimizer=10,
                                                normalize_y=True)

    for n in range(n_iters):

        model.fit(xp, yp)

        # Sample next point
        if random_search:
            x_random = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(random_search, n_params))
            ei = -1 * expected_improvement(x_random, model, yp, greater_is_better=True, n_params=n_params)
            next_sample = x_random[np.argmax(ei), :]
        else:
            next_sample = sample_next_point(expected_improvement, model, yp, greater_is_better=True, bounds=bounds,
                                            n_restarts=100, n_jobs=n_jobs)

        # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
        # if np.any(np.abs(next_sample - xp) <= epsilon):
        #    print("Warning. Got duplicate point to sample. Choose random point instead. (Iter: {})".format(n))
        #    print("Proposed point was", next_sample)
        #    next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])

        # Sample loss for new set of parameters
        cv_score = sample_loss(next_sample)

        # Update lists
        x_list.append(next_sample)
        y_list.append(cv_score)

        # Update xp and yp
        xp = np.array(x_list)
        yp = np.array(y_list)

    return xp, yp, model
