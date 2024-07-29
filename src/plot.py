#
# Plotting methods
#
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import numpy as np
from src.utils import set_plotting_format


def plot_comparison_data(simulation, datasets, xs=[0.25, 0.5, 0.75, 1], plotting_format="paper"):
    set_plotting_format(plotting_format)
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.5), sharey=True, sharex=True)

    L = simulation.parameter_values["Pipe length [m]"]
    R = simulation.parameter_values["Capsule radius [m]"]

    solution = simulation.solution

    for data, x in zip(datasets["PCM"], xs):
        time = solution["Time [s]"].entries
        T_HTF = solution["Phase-change material temperature [degC]"](t=time, x=x*L, r=0.8*R)
        axes[1, 0].plot(time, T_HTF)
        axes[1, 1].plot(data["Time [min]"] * 60, data["PCM Temperature [degC]"], '.-', label=f"{x:.2f}L")

    for data, x in zip(datasets["HTF"], xs):
        time = solution["Time [s]"].entries
        T_HTF = solution["Heat transfer fluid temperature [degC]"](t=time, x=x*L)
        axes[0, 0].plot(time, T_HTF, label=f"{x:.2f}L")
        axes[0, 1].plot(data["Time [min]"] * 60, data["HTF Temperature [degC]"], '.-', label=f"{x:.2f}L")

    for ax in axes[1, :]:
        ax.set_xlabel("Time [s]")

    for (ax, letter) in zip(axes.flatten(), ["a", "b", "c", "d"]):
        ax.text(0, 70, f"({letter})", horizontalalignment='left', verticalalignment='top',)

    axes[1, 0].set_ylabel("PCM temperature [°C]")
    axes[0, 0].set_ylabel("HTF temperature [°C]")
    axes[0, 0].set_title("Model")
    axes[0, 1].set_title("Experimental data")
    axes[0, 0].legend()

    return fig, axes

def compare_0D_variables(simulations, output_variables=None, variable_names=None, plotting_format="paper"):
    if output_variables is None:
        output_variables = [
            "Outlet temperature [degC]",
            "X-averaged state of charge",
            "Stored energy per unit area [J.m-2]",
            "Relative error in energy conservation [%]",
        ]
        if variable_names is None:
            variable_names = [
                "Outlet temperature [°C]",
                "State of charge",
                "Stored energy per\n unit area [J m${}^{-2}$]",
                "Relative error in\n energy conservation [%]",
            ]
    if variable_names is None:
        variable_names = output_variables

    if not isinstance(simulations, list):
        simulations = [simulations]

    set_plotting_format(plotting_format)
    N_rows = math.ceil(len(output_variables)/ 2)
    fig, axes = plt.subplots(N_rows, 2, figsize=(5.5, 0.5 + 1.5 * N_rows), sharey=False, sharex=True)

    if len(simulations) > 1:
        for i, var_name in enumerate(output_variables):
            ax = axes.flat[i]
            time = simulations[1].solution["Time [s]"].data
            var = simulations[1].solution[var_name].data
            ax.plot(time, var, "k--", label=simulations[1].model.name)

    for i, var_name in enumerate(output_variables):
        ax = axes.flat[i]
        time = simulations[0].solution["Time [s]"].data
        var = simulations[0].solution[var_name].data
        ax.plot(time, var, label=simulations[0].model.name)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(variable_names[i])
    
    axes[0, 0].legend()
    fig.tight_layout()
    return fig, axes

def compare_1D_variables(simulations, output_variables=None, variable_names=None, times=5, plotting_format="paper"):
    if output_variables is None:
        output_variables = [
            "Heat transfer fluid temperature [degC]",
            "Phase-change material surface temperature [degC]",
        ]
        if variable_names is None:
            variable_names = [
                "HTF temperature [°C]",
                "PCM surface\n temperature [°C]",
            ]
    if variable_names is None:
        variable_names = output_variables

    if not isinstance(simulations, list):
        simulations = [simulations]

    if not isinstance(times, list):
        if isinstance(times, int):
            end_time = simulations[0].solution["Time [s]"].data[-1]
            times = [end_time * i / (times - 1) for i in range(times)]
        else:
            raise ValueError("times must be an integer or a list")


    set_plotting_format(plotting_format)
    N_rows = math.ceil(len(output_variables)/ 2)
    fig, axes = plt.subplots(N_rows, 2, figsize=(5.5, 0.5 + 1.5 * N_rows), sharey=False, sharex=True)
    
    viridis = cm.get_cmap('viridis')
    colours = viridis(np.linspace(0, 0.9, len(times)))

    if len(simulations) > 1:
        for i, var_name in enumerate(output_variables):
            label = simulations[1].model.name
            for t in times:
                ax = axes.flat[i]
                time = simulations[1].solution["x [m]"](t=t)
                var = simulations[1].solution[var_name](t=t)
                ax.plot(time, var, "k--", label=label)
                label = None

    for i, var_name in enumerate(output_variables):
        label = simulations[0].model.name
        for t, c in zip(times, colours):
            ax = axes.flat[i]
            time = simulations[0].solution["x [m]"](t=t)
            var = simulations[0].solution[var_name](t=t)
            ax.plot(time, var, color=c, label=label)
            label = None

            ax.set_xlabel("x [m]")
            ax.set_ylabel(variable_names[i])
    
    axes.flat[0].legend()
    fig.tight_layout()
    return fig, axes

def compare_2D_variables(simulations, output_variable=None, variable_name=None, times=4, xs=5, plotting_format="paper"):
    if output_variable is None:
        output_variable = "Phase-change material enthalpy [J.m-3]"
        if variable_name is None:
            variable_name = "Phase-change material\n enthalpy [J m${}^{-3}$]"
    if variable_name is None:
        variable_name = output_variable

    if not isinstance(times, list):
        if isinstance(times, int):
            end_time = simulations[0].solution["Time [s]"].data[-1]
            times = [end_time * i / (times - 1) for i in range(times)]
        else:
            raise ValueError("times must be an integer or a list")

    if not isinstance(xs, list):
        if isinstance(xs, int):
            Z = simulations[0].parameter_values["Pipe length [m]"]
            xs = [Z * i / (xs - 1) for i in range(xs)]
        else:
            raise ValueError("xs must be an integer or a list")

    set_plotting_format(plotting_format)
    N_rows = math.ceil(len(times)/ 2)
    fig, axes = plt.subplots(N_rows, 2, figsize=(5.5, 0.5 + 1.5 * N_rows), sharey=True, sharex=True)
    
    for t, ax in zip(times, axes.flat):
        if len(simulations) > 1:
            label = simulations[1].model.name
            for x in xs:
                r = simulations[1].solution["r [mm]"](t=t, x=Z/2)
                var = simulations[1].solution[output_variable](t=t, x=x)
                ax.plot(r, var, "lightgray", label=label)
                label = None

        r = simulations[0].solution["r [mm]"](t=t, x=Z/2)
        var = simulations[0].solution[output_variable](t=t, x=Z/2)
        ax.plot(r, var, label=simulations[0].model.name)
        ax.set_title(f"t = {t:.0f} s")

        ax.set_xlabel("r [mm]")
        ax.set_ylabel(variable_name)
    
    axes.flat[0].legend()
    fig.tight_layout()
    return fig, axes