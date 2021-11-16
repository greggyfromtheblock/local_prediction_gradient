from __future__ import annotations

import copy
import dataclasses
from itertools import cycle
import random
import os
import shutil
from typing import Callable, Optional, TypeVar

from cga import cga
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import neptune
import plotly.express as px
import pickle
from pysurvival.models.non_parametric import KaplanMeierModel
from pysurvival.models.simulations import SimulationModel
import pandas as pd
# import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
# import pickle
import seaborn as sns
from scipy.linalg import cholesky
from scipy.linalg import toeplitz
import seaborn as sn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tap
import tqdm.auto


def df2csv(
    df: pd.DataFrame,
    output_dir: str,
):
    """
    Writes csv given a dataframe + name
    """
    train, valid = train_test_split(df, test_size=0.3)
    train.to_csv(
        f"{output_dir}/train.csv",
        index=False,
    )
    valid.to_csv(
        f"{output_dir}/valid.csv",
        index=False,
    )


def get_coxph_coeffs(df):
    cph = CoxPHFitter()
    cph.fit(df, "time", "event")
    cph.print_summary()


def get_simpsons_paradox(
    p: float = 2,
    q: float = 1,
    n: float = 500,
    n_groups: int = 5,
):

    k = np.random.choice(5, size=n)
    scaling = np.random.normal(size=n)

    noise_x = np.random.normal(scale=0.25, size=n)
    noise_y = np.random.normal(scale=0.25, size=n)
    y = scaling * np.sin(p / q) + k + noise_y
    x = scaling * np.cos(p / q) + k + noise_x
    return x, y


T = TypeVar("T")


def ifnone(maybe: Optional[T], default: T) -> T:
    if maybe is None:
        return default
    else:
        return maybe


@cga.node
def simpson_x(
    scaling: float,
    group: float,
    noise: float,
) -> float:
    return scaling * np.cos(2 / 1) + group + noise


@cga.node
def simpson_y(
    scaling: float,
    group: float,
    noise: float,
) -> float:
    return scaling * np.sin(2 / 1) + group + noise


@cga.node
def simpson_hazzard(
    scaling: float,
    group: float,
) -> float:
    return np.where(group == 2, scaling, -scaling).item()


class SimpsonsParadoxGraph(cga.Graph):
    def __init__(self):
        # define functions
        noise = cga.node(lambda: np.random.normal(scale=0.27))
        get_group = cga.node(lambda: np.random.choice(5))
        get_scaling = cga.node(lambda: np.random.normal())

        self.noise_x = noise(name="noise_x")
        self.noise_y = noise(name="noise_y")
        self.group = get_group(name="group")

        self.scaling = get_scaling(name="scaling")

        self.x = simpson_x(self.scaling, self.group, self.noise_x, name="x")
        self.y = simpson_y(self.scaling, self.group, self.noise_y, name="y")

        self.hazzard = simpson_hazzard(self.scaling, self.group, name="hazzard")
        super().__init__([self.x, self.y, self.hazzard])

    def transform(
        self,
        dataset_row: pd.Series,
        set_values: dict = {},
        replace_nodes: dict = {},
    ) -> pd.Series:
        # print(dataset_row.keys())
        result = self.sample(
            set_values={
                self.scaling: dataset_row["predictive0"],
                self.group: np.digitize(
                    dataset_row["nonpredictive0"], [-1.5, -1, 0, 1, 1.5]
                ),
            },
            replace=replace_nodes,
        )
        return pd.Series(
            index=["x", "y", "event", "time"],
            data=[
                result[self.x],
                result[self.y],
                dataset_row["event"],
                dataset_row["time"],
            ],
        )


class Args(tap.Tap):
    dataset: str        # dataset to generate either `simpson`, `linear`.
    output_dir: str     # saves generated dataset to this directory.
    force_overwrite: bool = False   # overwrites output directory if exists.
    n_samples: int      # number of samples to generate.
    survival_distribution: str = 'weibull'
    risk_type: str = 'linear'
    censored_parameter: float = 5.0
    alpha: float = 1.
    beta: float = 5.


def run_simpson(args: Args):
    def fname(name: str) -> str:
        return os.path.join(args.output_dir, name)

    simpson_graph = SimpsonsParadoxGraph()

    gnx = simpson_graph.to_networkx()
    nx.draw_networkx(
        gnx,
        labels={n: n.name for n in gnx.nodes},
        pos=nx.layout.spring_layout(gnx, k=3),
        node_size=2_000,
    )
    plt.savefig(fname("graph.png"))

    data = []
    for _ in tqdm.auto.trange(args.n_samples):
        result = simpson_graph.sample()
        data.append(
            {
                "x": result[simpson_graph.x],
                "y": result[simpson_graph.y],
                "hazzard": float(result[simpson_graph.hazzard]),
                "group": result[simpson_graph.group],
                "scaling": result[simpson_graph.scaling],
            }
        )

    df = pd.DataFrame(data)
    df.plot.scatter("x", "y", c="hazzard", cmap="Reds", colorbar=True)
    plt.savefig(fname("hazzard.png"))
    plt.close()

    plt.hist(df.hazzard)
    plt.savefig(fname("hazzard.png"))
    plt.close()

    sim = SimulationModelWithRisk(
        survival_distribution=args.survival_distribution,
        risk_type=args.risk_type,
        censored_parameter=args.censored_parameter,
        alpha=args.alpha,
        beta=args.beta,
    )

    df = sim.generate_data(df)

    df.plot.scatter("x", "y", c="time", cmap="Reds", colorbar=True)
    plt.savefig(fname("times.png"), bbox="tight")

    df2csv(df, args.output_dir)

    df_interventions = get_interventions(simpson_graph, sim, args.n_samples)
    df_interventions.to_csv(
        os.path.join(args.output_dir, "interventions.csv")
    )


def get_interventions(
    g: SimpsonsParadoxGraph,
    sim: SimulationModelWithRisk,
    n_samples: int,
) -> pd.DataFrame:
    data = []
    for node in [g.noise_x, g.noise_y]:
        for _ in tqdm.auto.trange(n_samples,
                                  desc=f"Intervention {node.name}"):
            orig, intervention = g.sample_do(
                action=cga.Resample(node),
            )
            row = {'modified_attribute': node.name}
            row.update({
                n.name + "_orig": v
                for n, v in orig.items()
            })
            row.update({
                n.name + "_do": v
                for n, v in intervention.items()
            })
            data.append(row)
    df = pd.DataFrame(data)

    sim_df = sim.generate_data(df, hazzard_column='hazzard_orig')
    df['event_orig'] = sim_df.event
    df['time_orig'] = sim_df.time

    sim_df = sim.generate_data(df, hazzard_column='hazzard_do')
    df['event_do'] = sim_df.event
    df['time_do'] = sim_df.time
    return pd.DataFrame(data)


def run_linear_case(args: Args):
    sim = SimulationModel(
        survival_distribution=args.survival_distribution,
        risk_type=args.risk_type,
        censored_parameter=args.censored_parameter,
        alpha=args.alpha,
        beta=args.beta,
    )

    num_samples = 100_000

    """
    Feature weight controls how predictive the variables will be.
    0 variables with no effect on survival
    > 1 for variables with effect on survival time
    """

    # Set betas
    pred_weights = [
        np.log(5.5),
        np.log(1.0),
    ]  # , np.log(1.75), np.log(1.5), np.log(1.4)]
    nonpred_weights = [0] * 2
    # extra weights are created with pretty high beta, because we want the
    # extra variables to have correlation with the label
    extra_weights = [np.log(1.5)] * 3
    feature_weights = pred_weights + nonpred_weights + extra_weights

    dataset = sim.generate_data(
        num_samples=num_samples,
        num_features=len(feature_weights),
        feature_weights=feature_weights,
    )

    # Set feature names
    pred_features = [f"predictive{x}" for x in range(len(pred_weights))]
    nonpred_features = [f"nonpredictive{x}" for x in range(len(nonpred_weights))]
    if len(extra_weights) > 0:
        extra_weights = [f"n{i+1}" for i in range(len(extra_weights))]

    standard_scaler = StandardScaler()
    features = pred_features + nonpred_features + extra_weights
    features.append("time")
    features.append("event")
    dataset.columns = features

    for col in dataset.columns.to_list()[:-2]:
        dataset[col] = standard_scaler.fit_transform(dataset[[col]])

    get_coxph_coeffs(dataset)
    # (Greg): Finish exporting of the dataset


class SimulationModelWithRisk(SimulationModel):
    """
    Subclasses `SimulationModel` to generated data from an predefined
    risk factor.
    """

    def generate_data(
        self,
        dataframe: pd.DataFrame,
        hazzard_column="hazzard",
    ):
        """
        Generating a dataset of simulated survival times from a given
        distribution through the hazard function using the Cox model

        Parameters:
        -----------

        * `dataframe`: **pd.Dataframe** --
            A pandas dataframe with a risk column.

        * `hazzard_column`: **str** *(default="risk")* --
            Name of the risk column.

        Returns:
        --------
        * dataset: pandas.DataFrame
            dataset of simulated survival times, event status and features


        Example:
        --------
        from pysurvival.models.simulations import SimulationModel

        # Initializing the simulation model
        sim = SimulationModel( survival_distribution = 'gompertz',
                               risk_type = 'linear',
                               censored_parameter = 5.0,
                               alpha = 0.01,
                               beta = 5., )

        # Generating N Random samples
        N = 1000
        dataset = sim.generate_data(num_samples = N, num_features=5)

        # Showing a few data-points
        dataset.head()
        """

        def risk_function(risk: np.ndarray) -> np.ndarray:
            # Choosing the type of risk
            if self.risk_type.lower() == "linear":
                return risk.reshape(-1, 1)

            elif self.risk_type.lower() == "square":
                risk = np.square(risk * self.risk_parameter)

            elif self.risk_type.lower() == "gaussian":
                risk = np.square(risk)
                risk = np.exp(-risk * self.risk_parameter)

            return risk.reshape(-1, 1)

        num_samples = len(dataframe)

        BX = risk_function(np.array(dataframe[hazzard_column]))

        # Building the survival times
        T = self.time_function(BX)
        C = np.random.normal(loc=self.censored_parameter, scale=5, size=num_samples)
        C = np.maximum(C, 0.0)
        time = np.minimum(T, C)
        E = 1.0 * (T == time)

        # Building dataset
        self.dataset = copy.deepcopy(dataframe)
        self.dataset["time"] = time
        self.dataset["event"] = E

        # Building the time axis and time buckets
        self.times = np.linspace(0.0, max(self.dataset["time"]), self.bins)
        self.get_time_buckets()

        # Building baseline functions
        self.baseline_hazard = self.hazard_function(self.times, 0)
        self.baseline_survival = self.survival_function(self.times, 0)

        # Printing summary message
        message_to_print = "Number of data-points: {} - Number of events: {}"
        print(message_to_print.format(num_samples, sum(E)))
        return self.dataset


def main():
    args = Args().parse_args()

    if os.path.exists(args.output_dir):
        if args.force_overwrite:
            shutil.rmtree(args.output_dir)
        else:
            raise Exception(
                "Output directory is not empty and --force_overwrite not set. "
                f"Got {args.output_dir}"
            )
    os.makedirs(args.output_dir)

    if args.dataset == "simpson":
        run_simpson(args)
    elif args.dataset == "linear":
        run_linear_case(args)
    else:
        raise Exception("Unknown dataset: {args.dataset}")
