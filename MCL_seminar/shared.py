import numpy as np
import matplotlib.pyplot as plt
import tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

def init_plot(fig=None, lim_from=-1, lim_to=1, window_title="IRS 01"):
    if fig is None:
        fig = plt.figure(figsize=(8, 8))
    try:
        fig.canvas.manager.set_window_title(window_title)
    except:
        0
        
    ax = fig.add_subplot(projection="3d")
    ax.view_init(azim=30, elev=20)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim3d([lim_from, lim_to])
    ax.set_ylim3d([lim_from, lim_to])
    ax.set_zlim3d([lim_from, lim_to])
    ax.set_xlabel("$x_j$")
    ax.set_xlabel("$y_j$")
    ax.set_xlabel("$z_j$")
    
    return fig, ax

def init_tkinter():
    tk = tkinter.Tk()
    tk.wm_title("IRS: Direktna kinematika robotske ruke")

    plt.ion()
    fig = Figure(figsize=(8, 8))
    img = tkinter.Frame(tk)

    canvas = FigureCanvasTkAgg(fig, master=img)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

    img.pack(side="left")

    fig, ax = init_plot(fig)
    

    toolbar = NavigationToolbar2Tk(canvas, tk)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
        
    return tk, fig, ax

def init_plot_2D(fig=None, lim_from=-2, lim_to=2):
    if fig is None:
        fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot()
    ax.set_box_aspect(1)
    ax.set_xlim([lim_from, lim_to])
    ax.set_ylim([lim_from, lim_to])
    ax.set_xlabel("$x_I$")
    ax.set_ylabel("$y_I$")

    ax.set_axisbelow(True)
    ax.grid()

    return fig, ax

def frame_factory():
    return np.array([
        [0, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [0, 0, 1, 1]
    ], dtype=float)

def frame_to_quiver(frame, diff=1):
    O_x, O_y, O_z = frame[0, :3]
    X = np.repeat(O_x, 3)
    Y = np.repeat(O_y, 3)
    Z = np.repeat(O_z, 3)
    U = frame[1:, 0] - O_x * diff
    V = frame[1:, 1] - O_y * diff
    W = frame[1:, 2] - O_z * diff
    return X, Y, Z, U, V, W

def update_quiver(frame, quiver):
    segs = np.array(frame_to_quiver(frame, diff=0)).reshape(6,-1)
    new_segs = [[[X,Y,Z],[U,V,W]] for X,Y,Z,U,V,W in zip(*segs.tolist())]
    quiver.set_segments(new_segs)
    
def Rot_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ], dtype=float)    

def T_rot_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta), np.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=float)   

# slajdovi p. 43
def omega_to_R(omega):
    theta = np.linalg.norm(omega)
    if theta == 0:
        return np.eye(3, dtype=float)
    omega_hat = omega / theta
    
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    v_theta = 1 - np.cos(theta)
    omega_x, omega_y, omega_z = omega_hat
    
    return np.array([
        [
            omega_x ** 2 * v_theta + c_theta, 
            omega_x * omega_y * v_theta - omega_z * s_theta, 
            omega_x * omega_z * v_theta + omega_y * s_theta
        ],
        [
            omega_x * omega_y * v_theta + omega_z * s_theta, 
            omega_y ** 2 * v_theta + c_theta, 
            omega_y * omega_z * v_theta - omega_x * s_theta
        ],
        [
            omega_x * omega_z * v_theta - omega_y * s_theta, 
            omega_y * omega_z * v_theta + omega_x * s_theta, 
            omega_z ** 2 * v_theta + c_theta
        ]
    ], dtype=float)

def A(DH_params):
    a, alpha, d, theta = DH_params
    c = np.cos
    s = np.sin

    return np.array([
        [c(theta), -s(theta) * c(alpha), s(theta) * s(alpha), a * c(theta)],
        [s(theta), c(theta) * c(alpha), -c(theta) * s(alpha), a * s(theta)],
        [0, s(alpha), c(alpha), d],
        [0, 0, 0, 1]
    ], dtype=float)


# inverz homogene transformacijske matrice
def T_inv(T):
    T_ = np.eye(4, dtype=float)
    T_[:3, :3] = T[:3, :3].T
    T_[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return T_


# operator koji preslikava kutnu brzinu \omega
# u antimsimetricnu matricu \lfloor \omega \rfloor
def skew(omega):
    omega_x, omega_y, omega_z = omega
    return np.array([
        [0, -omega_z, omega_y],
        [omega_z, 0, -omega_x],
        [-omega_y, omega_x, 0]
    ], dtype=float)

# lecture 4, slide 18
def R(theta):
    return np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ], dtype=float) 

# without last mask
wlm = np.array([1, 1, 0], dtype=float)

def update_wedge(wedge_patch, I_xi, alpha=10):
    wedge_patch.center = I_xi[:2]
    wedge_patch.theta1 = np.rad2deg(I_xi[2]) + alpha
    wedge_patch.theta2 = np.rad2deg(I_xi[2]) - alpha
    wedge_patch._recompute_path()

def normalize_angle(theta):
    return np.arctan2(np.sin(theta), np.cos(theta))

def R_array(thetas):
    n = thetas.shape[0]
    R_ = np.zeros((n, 3, 3), dtype=float)
    R_[:, 0, 0] = np.cos(thetas)
    R_[:, 0, 1] = np.sin(thetas)
    R_[:, 1, 0] = -np.sin(thetas)
    R_[:, 1, 1] = np.cos(thetas)
    R_[:, 2, 2] = 1

    return R_