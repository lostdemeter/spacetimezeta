import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpmath import zeta, mp

mp.dps = 20  # Higher precision for zeta computations

# Schwarzschild embedding function
def schwarzschild_embedding(r, M=1):
    return 2 * np.sqrt(2 * M * (r - 2 * M))

# Freefall geodesic parametric function
def freefall_geodesic(t, r0, M=1):
    return (r0**(3/2) - np.sqrt(2*M)*t)**(2/3)

# Generate data
r_vals = np.linspace(2.01, 10, 200)
z_vals = schwarzschild_embedding(r_vals)
t_vals = np.linspace(0, 150, 300)  # Extended t with reasonable resolution for speed
zeta_mags = [float(abs(zeta(mp.mpc(0.5, t)))) for t in t_vals]
inv_t = 1 / (t_vals + 1e-6)  # Freefall remap

# First 50 non-trivial zeros (imaginary parts, accurate to ~10 decimals from known values)
zeros = [14.1347251417, 21.0220396388, 25.0108575801, 30.4248761259, 32.9350615877, 
         37.5861781588, 40.9187190121, 43.3270732809, 48.0051508812, 49.7738324777, 
         52.9703214777, 56.4462476971, 59.3470440026, 60.8317785246, 65.1125440481, 
         67.0798105295, 69.5464017112, 72.0671576745, 75.7046906991, 77.1448400689, 
         79.3373750202, 82.9103808541, 84.7354929805, 87.4252746131, 88.8091112076, 
         92.4918992706, 94.6513440405, 95.8706342282, 98.8311942182, 101.3178510057, 
         103.7255380405, 105.4466230523, 107.1686111843, 111.0295355432, 111.8746591770, 
         114.3202209155, 116.2266803209, 118.7907828660, 121.3701250024, 122.9468292936, 
         124.2568185543, 127.5166838796, 129.5787042000, 131.0876885309, 133.4977372030, 
         134.7565097534, 138.1160420545, 139.7362089521, 141.1237074040, 143.1118458076]

# Verify zeros numerically (should be near zero on critical line)
verifications = [float(abs(zeta(mp.mpc(0.5, mp.mpf(zero))))) for zero in zeros]

# Compute spacings for patterns
spacings = np.diff(zeros)

# Mapped zeros and spacings
zeros_mapped = 1 / np.array(zeros)
mapped_spacings = np.diff(zeros_mapped)

# Geodesics
t_geod = np.linspace(0, 5, 100)
r_geod10 = freefall_geodesic(t_geod, r0=10)
z_geod10 = schwarzschild_embedding(r_geod10)
r_geod7 = freefall_geodesic(t_geod, r0=7)
z_geod7 = schwarzschild_embedding(r_geod7)
r_geod5 = freefall_geodesic(t_geod, r0=5)
z_geod5 = schwarzschild_embedding(r_geod5)

# Plots
fig = plt.figure(figsize=(12, 10))

# 3D Schwarzschild with geodesics
ax_sch = fig.add_subplot(2, 2, 1, projection='3d')
theta = np.linspace(0, 2*np.pi, 50)
r_mesh, theta_mesh = np.meshgrid(r_vals, theta)
x_mesh = r_mesh * np.cos(theta_mesh)
y_mesh = r_mesh * np.sin(theta_mesh)
z_mesh = schwarzschild_embedding(r_mesh)
ax_sch.plot_surface(x_mesh, y_mesh, z_mesh, cmap='plasma')
ax_sch.plot(r_geod10 * np.cos(0), r_geod10 * np.sin(0), z_geod10, 'r--', label='r0=10')
ax_sch.plot(r_geod7 * np.cos(np.pi/3), r_geod7 * np.sin(np.pi/3), z_geod7, 'g--', label='r0=7')
ax_sch.plot(r_geod5 * np.cos(2*np.pi/3), r_geod5 * np.sin(2*np.pi/3), z_geod5, 'b--', label='r0=5')
ax_sch.set_title('Schwarzschild Flamm Paraboloid with Freefall Geodesics')
ax_sch.set_xlabel('x')
ax_sch.set_ylabel('y')
ax_sch.set_zlabel('z')
ax_sch.legend()

# Zeta magnitude on critical line
axs = fig.add_subplot(2, 2, 2)
axs.plot(t_vals, zeta_mags, label='|ζ(0.5 + it)|')
axs.set_title('Zeta Magnitude on Critical Line (Extended to t=150)')
axs.set_xlabel('t (Im(s))')
axs.set_ylabel('|ζ|')
axs.grid(True)

# Freefall-style remap with zeros marked
axs2 = fig.add_subplot(2, 2, 3)
axs2.plot(inv_t, zeta_mags, label='|ζ| vs 1/t')
axs2.scatter(1 / np.array(zeros), [0]*len(zeros), c='red', label='Non-trivial Zeros')
axs2.set_title('Freefall-Style Remap of Zeta with 50 Zeros')
axs2.set_xlabel('1/t (Radial-like)')
axs2.set_ylabel('|ζ|')
axs2.legend()
axs2.grid(True)

# 3D Zeta landscape (limited to t<100 for computation speed)
sigma_vals = np.linspace(0.1, 1.5, 30)  # Reduced resolution
t_limited = t_vals[t_vals < 100]
t_mesh, sigma_mesh = np.meshgrid(t_limited, sigma_vals)
zeta_3d = np.array([[float(abs(zeta(mp.mpc(sigma, t)))) for t in t_limited] for sigma in sigma_vals])
ax3d = fig.add_subplot(2, 2, 4, projection='3d')
ax3d.plot_surface(sigma_mesh, t_mesh, zeta_3d, cmap='viridis')
ax3d.set_title('3D Zeta Landscape (|ζ(s)|, t<100)')
ax3d.set_xlabel('Re(s)')
ax3d.set_ylabel('Im(s)')
ax3d.set_zlabel('|ζ|')

plt.tight_layout()
plt.savefig('extended_freefall_plots.png', dpi=300)
