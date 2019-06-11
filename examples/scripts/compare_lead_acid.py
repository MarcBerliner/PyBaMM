import numpy as np
import pybamm

pybamm.set_logging_level("DEBUG")

# load models
models = [
    # pybamm.lead_acid.LOQS(),
    # pybamm.lead_acid.Composite(),
    pybamm.lead_acid.NewmanTiedemann()
]

# create geometry
geometry = models[-1].default_geometry

# load parameter values and process models and geometry
param = models[0].default_parameter_values
param.update(
    {
        "Typical current [A]": 17,
        "Typical electrolyte concentration [mol.m-3]": 5600,
        "Negative electrode reference exchange-current density [A.m-2]": 0.08,
        "Positive electrode reference exchange-current density [A.m-2]": 0.006,
        # "Maximum porosity of negative electrode": 0.92,
        # "Maximum porosity of positive electrode": 0.92,
    }
)
for model in models:
    param.process_model(model)
param.process_geometry(geometry)

# set mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 90, var.x_s: 90, var.x_p: 90}
mesh = pybamm.Mesh(geometry, models[-1].default_submesh_types, var_pts)

# discretise models
for model in models:
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)

# solve model
solutions = [None] * len(models)
t_eval = np.linspace(0, 1, 100)
solver = pybamm.ScikitsDaeSolver(root_tol=1e-8)
for i, model in enumerate(models):
    solution = solver.solve(model, t_eval)
    solutions[i] = solution

# plot
output_variables = [
    "Interfacial current density [A.m-2]",
    "Electrolyte concentration [mol.m-3]",
    "Volume-averaged velocity",
    "Electrolyte current density [A.m-2]",
    "Electrolyte potential [V]",
    "Terminal voltage [V]",
]
plot = pybamm.QuickPlot(models, mesh, solutions, output_variables)
plot.dynamic_plot()
