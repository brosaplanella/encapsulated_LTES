#
# Enthalpy method
#
import pybamm


class Enthalpy(pybamm.models.base_model.BaseModel):
    def __init__(self, coord_sys="cartesian", name="Stefan problem (enthalpy)"):
        super().__init__(name=name)

        ######################
        # Parameters
        ######################
        k_s = pybamm.Parameter("Solid phase conductivity [W.m-1.K-1]")
        k_l = pybamm.Parameter("Liquid phase conductivity [W.m-1.K-1]")
        rho_s = pybamm.Parameter("Solid phase density [kg.m-3]")
        rho_l = pybamm.Parameter("Liquid phase density [kg.m-3]")
        c_p_s = pybamm.Parameter("Solid phase specific heat capacity [J.kg-1.K-1]")
        c_p_l = pybamm.Parameter("Liquid phase specific heat capacity [J.kg-1.K-1]")
        L = pybamm.Parameter("Latent heat [J.kg-1]")
        T_m = pybamm.Parameter("Melting temperature [K]")
        T_b = pybamm.Parameter("Boundary temperature [K]")
        R = pybamm.Parameter("Radius [m]")
        H0 = pybamm.Parameter("Initial enthalpy [J.m-3]")

        ######################
        # Variables
        ######################
        H = pybamm.Variable("Enthalpy [J.m-3]", domain="PCM", scale=H0)
        T = pybamm.Variable(
            "Temperature [K]", domain="PCM", scale=abs(T_m - T_b), reference=T_m
        )
        r = pybamm.SpatialVariable("r", domain="PCM", coord_sys=coord_sys)
        self.r = r

        ######################
        # Governing equations
        ######################
        def T_fun(H):
            solid = H / (rho_s * c_p_s)
            liquid = T_m + (H - rho_s * (c_p_s * T_m + L)) / (rho_l * c_p_l)
            return pybamm.minimum(solid, T_m) + pybamm.maximum(liquid, T_m) - T_m

        def k(H):
            H_s = rho_s * c_p_s * T_m
            H_l = rho_s * (c_p_s * T_m + L)
            k_m = (k_l - k_s) / (rho_s * L) * (H - H_s) + k_s
            return k_s * (H <= H_s) + k_m * (H > H_s) * (H < H_l) + k_l * (H >= H_l)

        dHdt = pybamm.div(k(H) * pybamm.grad(T))
        self.rhs = {H: dHdt}
        self.algebraic = {T: T - T_fun(H)}

        self.boundary_conditions = {
            T: {"left": (pybamm.Scalar(0), "Neumann"), "right": (T_b, "Dirichlet")}
        }

        self.initial_conditions = {H: H0, T: T_fun(H0)}

        ######################
        # (Some) variables
        ######################
        self.variables = {
            "Temperature [K]": T,
            "Enthalpy [J.m-3]": H,
            "Time [s]": pybamm.t,
            "Time [min]": pybamm.t / 60,
            "x [m]": r,
        }

    @property
    def default_geometry(self):
        R = pybamm.Parameter("Radius [m]")
        return pybamm.Geometry({"PCM": {self.r: {"min": pybamm.Scalar(0), "max": R}}})

    @property
    def default_submesh_types(self):
        return {"PCM": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh)}

    @property
    def default_var_pts(self):
        return {self.r: 50}

    @property
    def default_spatial_methods(self):
        return {"PCM": pybamm.FiniteVolume()}

    @property
    def default_solver(self):
        return pybamm.IDAKLUSolver()
        # return pybamm.CasadiSolver("fast")
