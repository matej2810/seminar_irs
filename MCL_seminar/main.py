import numpy as np
from shared import init_plot_2D, update_wedge, normalize_angle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Wedge
import colors
from robot import Robot
from motion import sample_motion_model_velocity

alpha_slow = 0.02
alpha_fast = 0.3
w_slow = 0.0
w_fast = 0.0

sigma_r = 0.5
sigma_phi = 0.3
intro_hold = 12
kidnap_step = 170

sign = -1
def path_planning(i):
    global sign
    v = 0.2
    omega = np.pi / 8
    if i % 60 == 0:
        sign *= -1
    return np.array([v, 0, sign * omega], dtype=float)


# model mjerenja 
def measure(pose, landmarks):
    dx = landmarks[:, 0] - pose[0]
    dy = landmarks[:, 1] - pose[1]
    r = np.sqrt(dx ** 2 + dy ** 2)
    phi = normalize_angle(np.arctan2(dy, dx) - pose[2])
    return r, phi

# funkcije za izracun tezina i resampling
def compute_weights(z, particles, landmarks):
    r_z, phi_z = z
    dx  = landmarks[:, 0] - particles[:, 0:1]
    dy  = landmarks[:, 1] - particles[:, 1:2]
    r   = np.sqrt(dx ** 2 + dy ** 2)
    phi = normalize_angle(np.arctan2(dy, dx) - particles[:, 2:3])

    e_r   = r - r_z
    e_phi = normalize_angle(phi - phi_z)

    log_w = -0.5 * ((e_r / sigma_r) ** 2 + (e_phi / sigma_phi) ** 2)
    return np.exp(log_w.sum(axis=1))


def low_variance_resample(particles, weights, n):
    positions  = (np.arange(n) + np.random.uniform()) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    idx = np.searchsorted(cumulative, positions)
    return particles[idx]


def animate(i, robot, estimated_robot, particles, shapes, dt, state):
    global w_slow, w_fast

    # uvodna faza: samo prikaz stvarne poze i raspodjele cestica bez gibanja
    if i < intro_hold:
        shapes[1].set_offsets(particles[:, :2])
        mean = np.mean(particles, axis=0)
        mean[2] = np.arctan2(np.sin(particles[:, 2]).mean(),
                             np.cos(particles[:, 2]).mean())
        estimated_robot.I_xi = mean
        update_wedge(shapes[0], robot.I_xi)
        update_wedge(shapes[2], estimated_robot.I_xi)
        return

    R_xi_dot = path_planning(i)
    robot.update_state_R(R_xi_dot, dt)

    # otmica robota na nasumicnu pozu
    if i == kidnap_step and not state['kidnapped']:
        robot.I_xi[:] = np.array([
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(-np.pi, np.pi)])
        state['kidnapped'] = True
    update_wedge(shapes[0], robot.I_xi)

    # mjerenje iz stvarne poze
    z = measure(robot.I_xi, landmarks)

    # model gibanja
    particles[:] = sample_motion_model_velocity(R_xi_dot, particles, dt)
    particles[:, 2] = normalize_angle(particles[:, 2])

    # tezine (nenormalizirane)
    w = compute_weights(z, particles, landmarks)

    # w_avg, w_slow, w_fast prije normalizacije
    w_avg   = w.mean()
    w_slow += alpha_slow * (w_avg - w_slow)
    w_fast += alpha_fast * (w_avg - w_fast)

    # udio slucajnih cestica 
    p_rand_raw      = max(0.0, 1.0 - w_fast / (w_slow + 1e-300))
    state['p_rand'] = max(p_rand_raw, state['p_rand'] * 0.92)
    p_rand          = state['p_rand']

    w_sum = w.sum()
    w = np.full(M, 1.0 / M) if w_sum < 1e-300 else w / w_sum

    # augmented resampling: n_rand slucajnih + n_norm low-variance
    n_rand = int(M * p_rand)
    n_norm = M - n_rand

    new_p = low_variance_resample(particles, w, n_norm)
    if n_rand > 0:
        rp       = np.random.uniform(-2.5, 2.5, (n_rand, 3))
        rp[:, 2] = np.random.uniform(-np.pi, np.pi, n_rand)
        new_p    = np.vstack([new_p, rp])
    particles[:] = new_p

    # roughening 
    particles[:, :2] += np.random.randn(M, 2) * 0.03
    particles[:, 2]  += np.random.randn(M) * 0.05
    particles[:, 2]   = normalize_angle(particles[:, 2])

    shapes[1].set_offsets(particles[:, :2])

    mean    = np.mean(particles, axis=0)
    mean[2] = np.arctan2(np.sin(particles[:, 2]).mean(),
                         np.cos(particles[:, 2]).mean())
    estimated_robot.I_xi = mean
    update_wedge(shapes[2], estimated_robot.I_xi)


if __name__ == "__main__":
    fig, ax = init_plot_2D(lim_from=-3.0, lim_to=3.0)

    num_frames = 350
    fps = 30
    dt = 1 / fps

    shapes = []

    L = 7
    landmarks = np.random.uniform(-2.5, 2.5, (L, 2))
    ax.scatter(landmarks[:, 0], landmarks[:, 1], color=colors.orange)

    robot = Robot(np.array([
        np.random.uniform(-2.0, 2.0),
        np.random.uniform(-2.0, 2.0),
        np.random.uniform(-np.pi, np.pi)], dtype=float))

    robot_patch = ax.add_patch(
        Wedge(robot.I_xi[:2], 0.2,
              np.rad2deg(robot.I_xi[2]) + 10, np.rad2deg(robot.I_xi[2]) - 10,
              facecolor=colors.darkgray, alpha=0.7, zorder=2))
    shapes.append(robot_patch)

    M = 500
    particles = np.empty((M, 3))
    particles[:, 0] = np.random.uniform(-2.5, 2.5, M)
    particles[:, 1] = np.random.uniform(-2.5, 2.5, M)
    particles[:, 2] = np.random.uniform(-np.pi, np.pi, M)
    particles_scatter = ax.scatter(
        particles[:, 0], particles[:, 1], s=2, color=colors.green)
    shapes.append(particles_scatter)

    estimated_robot = Robot(np.mean(particles, axis=0))
    estimated_robot_patch = ax.add_patch(
        Wedge(estimated_robot.I_xi[:2], 0.2,
              np.rad2deg(estimated_robot.I_xi[2]) + 10,
              np.rad2deg(estimated_robot.I_xi[2]) - 10,
              facecolor=colors.yellow, alpha=0.7, zorder=3))
    shapes.append(estimated_robot_patch)

    state = {'kidnapped': False, 'p_rand': 0.0}

    ani = FuncAnimation(
        fig, animate,
        fargs=(robot, estimated_robot, particles, shapes, dt, state),
        frames=num_frames, interval=dt * 1000,
        repeat=False, blit=False, init_func=lambda: None)

    plt.show()