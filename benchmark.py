import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_diabetes,
    make_friedman1,
    make_moons,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split

import pymc_bart as pmb

# Set to None if seed is not needed
SEED = 42
CURR_PATH = Path(__file__).parent.resolve()


def fit_bart_and_evaluate(bart_model, X_train, y_train, X_test, y_test, m, draws, tune, chains):
    classification = len(np.unique(y_test)) == 2
    if classification:
        with pm.Model() as model:
            start_time = time.time()
            X_data = pm.Data("X", X_train)
            mu = bart_model("mu", X=X_data, Y=y_train, m=m)
            p = pm.Deterministic("p", pm.math.sigmoid(mu))
            pm.Bernoulli("y_obs", p=p, observed=y_train)

            idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=1)
            train_time = time.time() - start_time

        with model:
            pm.set_data({"X": X_test})
            ppc = pm.sample_posterior_predictive(idata, var_names=["p"])
            y_pred = ppc.posterior_predictive["p"]
            y_pred_point = (y_pred.mean(axis=(0, 1)) > 0.5).astype(int)

        accuracy = accuracy_score(y_test, y_pred_point)
        f1 = f1_score(y_test, y_pred_point)

        metrics = {
            "train_time": train_time,
            "Accuracy": accuracy,
            "F1-score": f1,
        }
    else:
        with pm.Model() as model:
            start_time = time.time()
            X_data = pm.Data("X_data", X_train)
            y_data = pm.Data("y_data", y_train)
            mu = bart_model("mu", X=X_data, Y=y_train, m=m)
            sigma = pm.HalfNormal("sigma", 1)
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_data, shape=y_data.shape)

            idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=1)
            train_time = time.time() - start_time

            with model:
                pm.set_data({"X_data": X_test, "y_data": np.zeros(len(X_test))})
                ppc = pm.sample_posterior_predictive(idata, var_names=["mu"])

            mu_samples = ppc.posterior_predictive["mu"]
            y_pred_point = mu_samples.mean(axis=(0, 1))

            metrics = {
                "train_time": train_time,
                "RMSE": root_mean_squared_error(y_test, y_pred_point),
                "MAE": mean_absolute_error(y_test, y_pred_point),
                "R2": r2_score(y_test, y_pred_point),
            }

    return metrics


def friedman_function_test(bart_model, m=20, draws=20, tune=20, chains=1):
    if SEED is not None:
        np.random.seed(SEED)
    X, y = make_friedman1(n_samples=1000)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)
    metrics = fit_bart_and_evaluate(
        bart_model, X_train, y_train, X_test, y_test, m, draws, tune, chains
    )
    return metrics


def moons_test(bart_model, m=20, draws=20, tune=20, chains=1):
    if SEED is not None:
        np.random.seed(SEED)
    X, y = make_moons(n_samples=200, noise=0.2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)
    metrics = fit_bart_and_evaluate(
        bart_model, X_train, y_train, X_test, y_test, m, draws, tune, chains
    )
    return metrics


def sin_test(bart_model, m=20, draws=20, tune=20, chains=1):
    rng = np.random.default_rng()
    if SEED is not None:
        rng = np.random.default_rng(42)
    X_train = rng.uniform(-np.pi, np.pi, size=100)[:, np.newaxis]
    y_train = np.sin(X_train).ravel() + rng.normal(0, 0.1, size=100)
    X_test = np.linspace(-np.pi, np.pi, 200)[:, np.newaxis]
    y_test = np.sin(X_test).ravel()

    metrics = fit_bart_and_evaluate(
        bart_model, X_train, y_train, X_test, y_test, m, draws, tune, chains
    )
    return metrics


def cancer_test(bart_model, m=20, draws=20, tune=20, chains=1):
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)
    metrics = fit_bart_and_evaluate(
        bart_model, X_train, y_train, X_test, y_test, m, draws, tune, chains
    )
    return metrics


def CSGO_test(bart_model, m=20, draws=20, tune=20, chains=1):
    CSGO_df = pd.read_csv(CURR_PATH / "datasets" / "CSGO_df.csv")
    y = CSGO_df["winnerSide"]
    X = CSGO_df.drop(columns=["winnerSide"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)
    metrics = fit_bart_and_evaluate(
        bart_model, X_train, y_train, X_test, y_test, m, draws, tune, chains
    )
    return metrics


def raisin_test(bart_model, m=20, draws=20, tune=20, chains=1):
    raisin_df = pd.read_csv(CURR_PATH / "datasets" / "Raisin_Dataset.csv")
    y = raisin_df["Class"].map(
        {raisin_df["Class"].unique()[0]: 0, raisin_df["Class"].unique()[1]: 1}
    )
    X = raisin_df.drop(columns=["Class"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)
    metrics = fit_bart_and_evaluate(
        bart_model, X_train, y_train, X_test, y_test, m, draws, tune, chains
    )
    return metrics


def california_test(bart_model, m=20, draws=20, tune=20, chains=1):
    california = fetch_california_housing()
    X = california.data
    y = california.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)
    metrics = fit_bart_and_evaluate(
        bart_model, X_train, y_train, X_test, y_test, m, draws, tune, chains
    )
    return metrics


def diabetes_test(bart_model, m=20, draws=20, tune=20, chains=1):
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)
    metrics = fit_bart_and_evaluate(
        bart_model, X_train, y_train, X_test, y_test, m, draws, tune, chains
    )
    return metrics


datasets = {
    "friedman": friedman_function_test,
    "sin": sin_test,
    "cancer": cancer_test,
    "csgo": CSGO_test,
    "california": california_test,
    "moons": moons_test,
    "diabetes": diabetes_test,
    "raisin": raisin_test,
}
models = {"BART": pmb.BART, "BARTOnTables": pmb.BARTOnTables}


def worker(queue, dataset_func, model_cls, params):
    """This function runs in a separate process."""
    try:
        # Unpack parameters
        chains = params["chains"]
        draws = params["draws"]
        tune = params["tune"]
        m = params["m"]

        # Start training
        metrics = dataset_func(model_cls, chains=chains, draws=draws, tune=tune, m=m)
        # Put result in queue
        queue.put({"status": "success", "data": metrics})
    except Exception as e:
        queue.put({"status": "error", "error": str(e)})


def evaluate(filename="logs", chains=2, draws=250, tune=25, m=50, runs=1):
    LOGS_PATH = CURR_PATH / "logs"
    if runs > 1:
        LOGS_PATH = LOGS_PATH / f"{filename}_{runs}_runs"
        LOGS_PATH.mkdir(parents=True, exist_ok=True)

    # Parameters to pass to the process
    params = {"chains": chains, "draws": draws, "tune": tune, "m": m}

    for i in range(runs):
        logs = {}
        logs["model_params"] = params

        for dataset_name, dataset_func in datasets.items():
            print("==" * 20)
            print(f"Run {i + 1}/{runs} | Dataset: {dataset_name}")
            print("==" * 20)

            logs[dataset_name] = {}

            for model_name, model_cls in models.items():
                print(f"--> Training {model_name}...")

                # Create queue to receive result
                queue = mp.Queue()

                # Create process
                # IMPORTANT: pass dataset_func and model_cls.
                # Ensure they are importable or defined in the global scope.
                p = mp.Process(target=worker, args=(queue, dataset_func, model_cls, params))

                # Start process
                p.start()

                # Wait for result (blocking call until process returns data)
                res = queue.get()

                # Wait for process to finish
                p.join()

                # Process the result
                if res["status"] == "success":
                    metrics = res["data"]
                    logs[dataset_name][model_name] = metrics
                    print(f"    Done. Metrics: {metrics}")
                else:
                    print(f"    ERROR in process: {res['error']}")
                    # Save partial logs on error
                    with open(LOGS_PATH / f"partial_{filename}.json", "w") as f:
                        json.dump(logs, f, indent=4)
                    raise RuntimeError(f"Training failed: {res['error']}")

        if runs > 1:
            with open(LOGS_PATH / f"{i}.json", "w") as f:
                json.dump(logs, f, indent=4)
        else:
            with open(LOGS_PATH / f"{filename}.json", "w") as f:
                json.dump(logs, f, indent=4)
            return logs


if __name__ == "__main__":
    # FILENAME = "benchmark_results"
    RUNS = 10
    # evaluate(filename="benchmark_results", chains=2, draws=250, tune=25, m=50, runs=RUNS)
    evaluate(filename="benchmark_results_150m", chains=2, draws=250, tune=25, m=150, runs=RUNS)
