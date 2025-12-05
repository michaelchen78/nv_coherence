from model import run_ensemble
import numpy as np
import yaml
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import time
import copy


# +++++++++++++++++++++++ Auxiliary functions +++++++++++++++++++++++ #
def make_output_dir(run_id: str, base: str = "output"):
    """
    Make output directory for a run: ./output/<run_id>/
    :param run_id: the run id from the config file
    :param base: the base output directory, should be ./output/
    :return: the output directory path as a string
    """
    base_path = Path(base)
    if not base_path.is_dir():
        raise RuntimeError(f"Base output directory '{base_path}' does not exist.")

    run_path = base_path / run_id
    try:
        run_path.mkdir()  # error if already exists
    except FileExistsError:
        raise RuntimeError(
            f"Run directory '{run_path}' already exists. "
            "Choose a new run_id or delete/rename the existing folder."
        )
    return str(run_path)


def load_config(path):
    """
    Loads the configuration from a YAML file.
    :param path: string path of the yaml file
    :return: the configuration as a dictionary
    """
    with open(path) as file:
        cfg = yaml.safe_load(file)

    # fix any non-Pythonic yaml variables
    time_space = cfg["compute_params"].pop("time_space")
    cfg["compute_params"]["time_space"] = np.linspace(time_space[0], time_space[1], time_space[2])

    # add run_id to every set of parameters
    cfg["model_params"]["run_id"] = cfg["run_id"]
    cfg["simulator_params"]["run_id"] = cfg["run_id"]
    cfg["compute_params"]["run_id"] = cfg["run_id"]
    cfg["ensemble_params"]["run_id"] = cfg["run_id"]

    return cfg


def average_ensemble(results, avg_method):
    """
    Expects a dictionary of length 3 of 2D arrays each with shape (num time_steps, ensemble_size), as is the output from
    the function model.run_ensemble. This will average over the ensemble using the method identified. For now there is
    just a simple mean and median.
    :param results: the output of model.run_ensemble
    :param avg_method: a string identifying the averaging method, straight from the yaml config file
    :return: [FILL THIS OUT!]
    """
    if avg_method == "mean":
        averaged = np.column_stack([np.mean(results[k], axis=-1) if results[k] is not None
                            else np.zeros(next(v for v in results.values() if v is not None).shape[0])  # num time_steps
                            for k in ("conv", "gen", "MC")])
    elif avg_method == "median":
        averaged = np.column_stack([np.median(results[k], axis=-1) if results[k] is not None
                            else np.zeros(next(v for v in results.values() if v is not None).shape[0])  # num time_steps
                            for k in ("conv", "gen", "MC")])
    else:
        raise Exception(f"Unknown averaging method")

    return averaged


def fit_curve_basic(results, time_space, cce_type="conv"):
    """
    Fit coherence L(t) to the stretched exponential exp(-(t/T2)^p).
    :param results: L(t)
    :param time_space: a numpy time space lining up with results
    :return: T2 (ms) and p.
    """
    # initial guess and bounds
    p0 = (0.1, 1.5)
    bounds = ([1e-6, 0.5], [1e6, 3.0])

    def stretch(t, T2, p):
        return np.exp(- (t / T2) ** p)

    if cce_type == "conv":
        idx = 0
    elif cce_type == "gen":
        idx = 1
    elif cce_type == "MC":
        idx = 2
    else:
        raise Exception(f"Unknown cce type")

    y = results.T[idx]  # CCE coherence
    T2, p = curve_fit(stretch, time_space, y, p0=p0, bounds=bounds)[0]
    return T2, p


def plot_coherence_panel(ax, time_space, result, cce_types, title=None, T2=None, p=None, runtime=None, config=None):
    """
    Draw a single coherence plot on the given axes.
    :param ax: axes to plot on
    :param time_space: numpy time space lining up with results
    :param result: numpy array with 3 columns, each being L(t) calculated with a different CCE method
    :param cce_types: the CCE methods used
    :param title: title of plot
    :param T2: T2 of the fitted curve
    :param p: p of the fitted curve
    :param runtime: how long it took to run
    :param config: config file, in case want to print more info on the plots
    :return: nothing
    """
    # plot each CCE curve
    for i, cce_type in enumerate(cce_types):
        ax.plot(time_space, result[:, i], label=cce_type)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Coherence")

    if title:
        ax.set_title(title)

    text_lines = []
    if config is not None:
        text_lines.append(f"size = {config['model_params']['size']}, p1_conc = {config['model_params']['p1_conc']}")
        text_lines.append(f"order = {config['simulator_params']['order']}, r_bath = "
                        f"{config['simulator_params']['r_bath']}, r_dipole = {config['simulator_params']['r_dipole']}")
        text_lines.append(f"ensemble = {config['ensemble_params']['ensemble_size']}, magnetic field = "
                          f"{config['compute_params']['magnetic_field']}")
    if T2 is not None:
        text_lines.append(f"Conventional " + r"$\mathbf{T2}$" + f"= {T2:.3g} ms")
    if p is not None:
        text_lines.append(f"Conventional " + r"$\mathbf{p}$" + f"= {p:.3g}")
    if runtime is not None:
        text_lines.append(r"$\mathbf{runtime}$" +  f"= {runtime:.2f} s")
    if text_lines:
        ax.text(0.99, 0.85, "\n".join(text_lines), transform=ax.transAxes, va="top", ha="right", fontsize=10)
    ax.legend()


# +++++++++++++++++++++++ Primary functions +++++++++++++++++++++++ #
def run_coherence_experiment(config, base_dir="output", grid_mode=False):
    """
    Runs a coherence experiment given some config. Fits the data, averages over the ensemble, and plots the coherence.
    :param config: see any yaml file for description.
    :param base_dir: the directory in which the new output directory associated with this experiment is put
    :param grid_mode: if running a grid search, will plot onto an ax which will be passed in as grid_mode
    :return: nothing
    """
    run_id = config.pop("run_id")
    run_dir = make_output_dir(run_id, base=base_dir)  # string path

    # create output text file
    out_dir = Path(run_dir)
    out_file = out_dir / (run_id + ".txt")
    with out_file.open("x", encoding="utf-8") as f:  # "x" = create, fail if exists
        f.write("Initialized output text file.\n")
        f.write(f"\nConfig:{config}.\n")

    # run experiment
    t_start = time.time()
    results = run_ensemble(**config)
    runtime = time.time() - t_start  # seconds

    # put results into csvs
    time_space = config["compute_params"]["time_space"]
    for cce_type, arr in results.items():  # recall, arr shape: (num time steps, ensemble_size)
        if arr is None:
            continue  # this cce_type wasn't computed
        data = np.column_stack([time_space, arr])  # first col = time
        header = "t," + ",".join(f"traj_{i}" for i in range(arr.shape[1]))
        out_path = out_dir / f"{run_id}_{cce_type}.csv"
        np.savetxt(out_path, data, delimiter=",", header=header, comments="")

    # average results over the ensemble
    avg_results = average_ensemble(results, avg_method=config["ensemble_params"]["avg_method"])
    for i, cce_type in enumerate(config["compute_params"]["cce_types"]):
        arr = avg_results[:, i]
        data = np.column_stack([time_space, arr])
        out_path = out_dir / f"{run_id}_{cce_type}_averaged.csv"
        header = "t,avg_L"
        np.savetxt(out_path, data, delimiter=",", header=header, comments="")

    # fit the curve and write to output
    T2, p = fit_curve_basic(avg_results, time_space, cce_type="conv")
    with out_file.open("a", encoding="utf-8") as f:
        f.write("\nResults:\n")
        f.write(f"\nRuntime:{runtime} s\n")
        f.write(f"\nConventional T2, p: {T2}, {p}\n")
        if "gen" in config["compute_params"]["cce_types"]:
            T2_gen, p_gen = fit_curve_basic(avg_results, time_space, cce_type="gen")
            f.write(f"\nGeneralized T2, p: {T2_gen}, {p_gen}\n")
        if "MC" in config["compute_params"]["cce_types"]:
            T2_mc, p_mc = fit_curve_basic(avg_results, time_space, cce_type="MC")
            f.write(f"\nGeneralized with MC T2, p: {T2_mc}, {p_mc}\n")

    # plot the results
    if not grid_mode:  # normal single experiment
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_coherence_panel(ax, time_space=time_space, result=avg_results, config=config,
                             cce_types=config["compute_params"]["cce_types"], title=run_id, T2=T2, p=p, runtime=runtime)
        fig.tight_layout()
        fig.savefig(out_dir / f"{run_id}-plot.png", dpi=300)
        plt.close(fig)
    elif isinstance(grid_mode, Axes):  # if running a grid search
        ax = grid_mode  # grid_mode *is* the Axes
        plot_coherence_panel(ax, time_space=time_space, result=avg_results, config=config,
                             cce_types=config["compute_params"]["cce_types"], title=run_id, T2=T2, p=p, runtime=runtime)
    else:
        raise Exception(f"grid_mode must be False or a matplotlib Axes object")


def grid_search(base_config, param1, param2, param1_values, param2_values):
    """
    Runs a grid of experiments using the method run_coherence_experiment. Also returns a figure with all results. As of
    now, does not include a text file with all results. Each experiment has its own output folder, as created by
    run_coherence_experiment.
    :param base_config: the base config, see any yaml file
    :param param1: a list of two strings, the first gives the sub-config (e.g., model_params or compute_params) and the
    second gives the specific parameter, for instance size or pulse_id.
    :param param2: same as param1
    :param param1_values: a list, each element is some value of param1
    :param param2_values: same as param1
    :return: nothing
    """
    run_id = base_config.pop("run_id")
    base_dir = make_output_dir(run_id)  # string path

    # create output text file
    out_dir = Path(base_dir)
    out_file = out_dir / (run_id + ".txt")
    with out_file.open("x", encoding="utf-8") as f:  # "x" = create, fail if exists
        f.write("Initialized output text file.\n")
        f.write(f"\nBase config:{base_config}.\n")
        f.write(f"\nGrid: param1 is {param1}, param2 is {param2}")
        f.write(f"\nparam1 range is {param1_values}, param2 range is {param2_values}\n\n")

    # initialize grid figure
    n_rows = len(param1_values)
    n_cols = len(param2_values)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True, sharey=True)
    if n_rows == 1 and n_cols == 1:  # normalize axes to a 2D array so the grid figure works
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    grid_t_start = time.time()
    for i, v1 in enumerate(param1_values):
        for j, v2 in enumerate(param2_values):
            print(f"\n\n\nstarting grid run: {param1[1]}={v1}, {param2[1]}={v2}")
            grid_run_config = copy.deepcopy(base_config)

            grid_run_config[param1[0]][param1[1]] = v1
            grid_run_config[param2[0]][param2[1]] = v2
            grid_run_config["run_id"] = f"{param1[1]}={v1}--{param2[1]}={v2}"

            ax = axes[i, j]
            run_coherence_experiment(grid_run_config, base_dir=base_dir, grid_mode=ax)
            ax.set_title(f"{param1[1]}={v1}, {param2[1]}={v2}")
            print(f"finished grid run:{param1}={v1}, param2={v2}\n\n\n")
    grid_runtime = time.time() - grid_t_start
    with out_file.open("a", encoding="utf-8") as f:  # record total grid run time
        f.write(f"\nTotal grid run time: {grid_runtime}\n")

    # finish the grid plot
    fig.suptitle(f"Coherence: grid over {param1[1]} and {param2[1]}", fontsize=16)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_dir / f"grid_{param1[1]}-{param2[1]}.png", dpi=300)
    plt.close(fig)


def main(config_path, run_type_id, grid_dict=None):
    config = load_config(config_path)
    if run_type_id == "standard":
        run_coherence_experiment(config)
    elif run_type_id == "grid":
        grid_search(config, **grid_dict)
    else:
        raise RuntimeError(f"Unknown run type '{run_type_id}'")


if __name__ == '__main__':
    yaml_path = "config/4.dec.2025_large-r-dipole.yaml"

    run_type = "standard"  # "standard" or "grid"
    main(yaml_path, run_type)

    # run_type = "grid"
    # grid_dict = {
    #     "param1": ["compute_params", "magnetic_field"],
    #     "param2": ["model_params", "p1_conc"],
    #     "param1_values": [[0,0,400], [0,0,1900], [0,0,3000]],
    #     "param2_values": [0, 9.0e-3]
    # }
    # main(yaml_path, run_type, grid_dict=grid_dict)
