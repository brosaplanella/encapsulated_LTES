import pybamm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import src
import math


src.set_plotting_format("paper")

models = [
    src.ReducedModel(),
    src.FullModel(),
]

min_mesh = 0
max_mesh = 4

level_range = range(min_mesh, max_mesh + 1)



param = src.get_parameter_values("Nallusamy2007")
param["Heat transfer coefficient [W.m-2.K-1]"] = 1000

simulations = [[], []]

plt.figure(figsize=(2.75, 2))

for l in range(min_mesh, max_mesh + 1):
    for model, simulation in zip(models, simulations):
        r = model.variables["r [m]"]
        x = model.variables["x [m]"]
        var_pts = {
            r: math.floor(10 * 2 ** l),
            x: math.floor(20 * 2 ** l)
        }
        sim = pybamm.Simulation(model, parameter_values=param, var_pts=var_pts)
        solution = sim.solve(np.linspace(0, 10000, 5000))
        
        simulation.append(sim)
        print(f"{model.name}: {solution.solve_time}")

# Plot convergence of energy conservation
print("Generating energy conservation plot")
errors = [[], []]
for simulation, error in zip(simulations, errors):
    for level, sim in zip(level_range, simulation):
            err = sim.solution["Relative error in energy conservation [%]"].data.mean()
            error.append(err)


for model, error in zip(models, errors):
    plt.loglog([2 ** i for i in level_range], error, ".-", label=model.name)

plt.legend()
plt.xlabel("Mesh refinement factor")
plt.ylabel("Relative error in\n energy conservation [%]")
plt.tight_layout()

plt.savefig("convergence_conservation_error.png", dpi=300)

print("Energy conservation plot saved")

## Plot convergence of variables
x = np.linspace(0, param["Pipe length [m]"], 100)
r = np.linspace(0, param["Capsule radius [m]"], 100)
t = np.linspace(0, 10000, 500)
data = {}
# process HTF temperature
print("Generating HTF temperature plot")
errors = [[], []]
for simulation, error in zip(simulations, errors):
    benchmark = simulation[-1].solution["Heat transfer fluid temperature [K]"](t=t, x=x)
    for sim in simulation[:-1]:
        solution = sim.solution["Heat transfer fluid temperature [K]"](t=t, x=x)
        err = np.sqrt(((solution - benchmark) ** 2).mean()) / np.sqrt((benchmark ** 2).mean())
        error.append(err)

data["HTF temperature [K]"] = errors

# process PCM temperature
print("Generating PCM temperature plot")
errors = [[], []]
for simulation, error in zip(simulations, errors):
    benchmark = simulation[-1].solution["Phase-change material temperature [K]"](t=t, x=x, r=r)
    for sim in simulation[:-1]:
        solution = sim.solution["Phase-change material temperature [K]"](t=t, x=x, r=r)
        err = np.sqrt(((solution - benchmark) ** 2).mean()) / np.sqrt((benchmark ** 2).mean())
        error.append(err)

data["PCM temperature [K]"] = errors

fig, axes = plt.subplots(1, 2, figsize=(5.5, 2))
for (var, errors), ax in zip(data.items(), axes):
    for model, error in zip(models, errors):
        ax.loglog([2 ** i for i in level_range[:-1]], error, ".-", label=model.name)
        ax.set_xlabel("Mesh refinement factor")
        ax.set_ylabel(f"Relative error")
        ax.set_title(var)

axes[0].legend()
fig.tight_layout()

fig.savefig("convergence_variables.png", dpi=300)

print("Plots saved")