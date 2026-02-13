import numpy as np
from pathlib import Path

# define length and number of points
s0 = 0.0
s1 = 7.0
s2 = 10.0
n_points = 200

# randomize positions
np.random.seed(0)
r_OP0 = 2 * (np.random.rand(3) - 0.5)
r_OP1 = 2 * (np.random.rand(3) - 0.5)
r_OP2 = 2 * (np.random.rand(3) - 0.5)

t_0 = 2 * (np.random.rand(3) - 0.5)
t_1 = 2 * (np.random.rand(3) - 0.5)
t_2 = 2 * (np.random.rand(3) - 0.5)

print(f"{r_OP0 = }")
print(f"{r_OP1 = }")
print(f"{r_OP2 = }")
print(f"{  t_0 = }")
print(f"{  t_1 = }")
print(f"{  t_2 = }")

# cubic hermite polynomials in [0, 1]
h_basic = lambda s, d: np.array([
    2 * s**3 - 3 * s**2 + 1,
    (s**3 - 2 * s**2 + s) * d,
    -2 * s**3 + 3 * s**2,
    (s**3 - s**2)*d
])
h_basic_s = lambda s, d: np.array([
    6 * s**2 - 6 * s,
    (3 * s**2 - 4 * s + 1) * d,
    -6 * s**2 + 6 * s**1,
    (3 * s**2 - 2 * s)*d
])

# map polynomials to first domain: s in [0, s1]
ds0 = s1 - s0
h0 = lambda s: h_basic((s - s0) / ds0, ds0)
h0_s = lambda s: h_basic_s((s - s0) / ds0, ds0) / ds0

# map polynomials to second domain: s in [s1, s2]
ds1 = s2 - s1
h1 = lambda s: h_basic((s - s1) / ds1, ds1)
h1_s = lambda s: h_basic_s((s - s1) / ds1, ds1) / ds1


# interpolate
s_values0 = np.linspace(s0, s1, int(n_points * (s1 - s0) / s2))
r_OP_0 = (
    np.outer(r_OP0, h0(s_values0)[0])
    + np.outer(t_0, h0(s_values0)[1])
    + np.outer(r_OP1, h0(s_values0)[2])
    + np.outer(t_1, h0(s_values0)[3])
)
t__0 = (
    np.outer(r_OP0, h0_s(s_values0)[0])
    + np.outer(t_0, h0_s(s_values0)[1])
    + np.outer(r_OP1, h0_s(s_values0)[2])
    + np.outer(t_1, h0_s(s_values0)[3])
)
s_values1 = np.linspace(s1, s2, int(n_points * (s2 - s1) / s2))
r_OP_1 = (
    np.outer(r_OP1, h1(s_values1)[0])
    + np.outer(t_1, h1(s_values1)[1])
    + np.outer(r_OP2, h1(s_values1)[2])
    + np.outer(t_2, h1(s_values1)[3])
)
t__1 = (
    np.outer(r_OP1, h1_s(s_values1)[0])
    + np.outer(t_1, h1_s(s_values1)[1])
    + np.outer(r_OP2, h1_s(s_values1)[2])
    + np.outer(t_2, h1_s(s_values1)[3])
)

# create header and stack data together
header = "s, r_OP_x, r_OP_y, r_OP_z, t_x, t_y, t_z"
h2 = ", ".join([f"q{i}" for i in range(1, 7)])
s_values = np.concatenate([s_values0, s_values1[1:]])
r_OP = np.hstack([r_OP_0, r_OP_1[:, 1:]])
t_ = np.hstack([t__0, t__1[:, 1:]])
data = np.vstack([s_values, *r_OP, *t_]).T

# create path and save
path = Path(Path(__file__).parent, "results.csv")
np.savetxt(
    path,
    data,
    delimiter=", ",
    header=header,
    comments="",
)
