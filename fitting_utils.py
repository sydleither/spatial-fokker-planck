import numpy as np

from common import classify_game, game_colors
from fokker_planck import param_names


def get_regime_awms(mu, amw, sm, sigma):
    if sigma <= 0:
        print("sigma <= 0 invalid for getting regime awms")
        return
    regimes = {}
    regimes["maintenance"] = (mu * amw) / (sm * (1 + sm))
    regimes["masking"] = amw + 2 * sm
    regimes["mirroring"] = -((2 * sm) / (1 - sm)) + mu * ((2 * sm + amw) / (sm * (1 - sm)))
    regimes["mimicry"] = -(sigma / (1 + sigma)) + ((mu * (amw - sigma)) / (sigma * (1 + sigma)))
    return regimes


def get_regime_amws(mu, awm, sm, sigma):
    if sigma >= 0:
        print("sigma >= 0 invalid for getting regime amws")
        return
    regimes = {}
    regimes["maintenance"] = -((mu * awm * (1 + sm)) / sm)
    regimes["masking"] = awm - 2 * sm
    regimes["mirroring"] = -2 * sm - mu * ((2 * sm - awm * (1 - sm)) / sm)
    regimes["mimicry"] = sigma - (mu * (sigma + (1 + sigma) * awm)) / sigma
    return regimes


def calculate_sigma(mu, awm, amw, sm):
    sigma_bot = 2 * sm + awm * (sm + np.sqrt(4 * mu**2 + sm**2) - np.sign(sm) * 2 * mu)
    if sigma_bot == 0:
        print("Denominator of sigma equation equals zero")
        print(f"\tmu={mu}, awm={awm}, amw={amw}, sm={sm}")
        return -np.inf
    sigma_top = (
        2 * sm**2
        + sm * (amw - awm)
        + (np.sign(sm) * 2 * mu - np.sqrt(4 * mu**2 + sm**2)) * (amw + awm)
    )
    return sigma_top / sigma_bot


def regime_distances(mu, awm, amw, sm):
    """
    Get the distance the parameter set is from each MMMM regime
    """
    sigma = calculate_sigma(mu, awm, amw, sm)
    if sigma in (-np.inf, 0):
        return {"maintenance": -1, "masking": -1, "mirroring": -1, "mimicry": -1}
    if sigma > 0:
        awm_regimes = get_regime_awms(mu, amw, sm, sigma)
        distances = {k: abs(v - awm) for k, v in awm_regimes.items()}
    else:
        amw_regimes = get_regime_amws(mu, awm, sm, sigma)
        distances = {k: abs(v - amw) for k, v in amw_regimes.items()}
    return distances


def parameter_distances(true_ends, walker_ends):
    differences = {}
    for p, param_name in enumerate(param_names):
        walkers_param = walker_ends[:, p]
        walkers_param_distance = np.abs(walkers_param - true_ends[p])
        differences[f"Mean {param_name} Difference"] = np.mean(walkers_param_distance)
        differences[f"Variance in {param_name} Difference"] = np.var(walkers_param_distance)
    return differences


def quadrant_classification(true_params, walker_ends):
    game_matches = []
    true_game = classify_game(*true_params[2:5])
    for walker_end in walker_ends:
        walker_game = classify_game(*walker_end[2:5])
        game_matches.append(1 if walker_game == true_game else 0)
    game_matches = sum(game_matches) / len(game_matches)
    return {"Correct Game Classifications": game_matches, "Game": game_colors[true_game]}


def evaluate_performance(fp, xdata, true_ydata, walker_ends, n, mu, awm, amw, sm, c):
    true_params = np.array([n, mu, awm, amw, sm, c])
    len_data = len(xdata)
    idv_param_distances = parameter_distances(true_params, walker_ends)
    mse = []
    for walker_params in walker_ends:
        walker_ydata = fp(xdata, *walker_params)
        mse.append(np.sum((true_ydata - walker_ydata) ** 2) / len_data)
    regimes = regime_distances(mu, awm, amw, sm)
    game_classifications = quadrant_classification(true_params, walker_ends)
    data = {
        "N": n,
        "mu": mu,
        "awm": awm,
        "amw": amw,
        "sm": sm,
        "c": c,
        "Mean Curve MSE": np.mean(mse),
        "Variance in Curve MSE": np.var(mse),
        "Mean Probability Density": np.mean(true_ydata),
    }
    #TODO add game quadrant distance
    return data | regimes | idv_param_distances | game_classifications
