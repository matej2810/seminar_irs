import numpy as np

alpha_1 = alpha_2 = alpha_3 = alpha_4 = alpha_5 = alpha_6 = 0.2


def sample_triangle_distribution(b_squared, M):
    a = np.sqrt(6 * b_squared) / 2
    return np.random.uniform(-a, a, M) + np.random.uniform(-a, a, M)

# model kretanja
def sample_motion_model_velocity(u_current, x_prev, dt):
    v = u_current[0]
    omega = u_current[2]
    M = x_prev.shape[0]

    v_hat = v + sample_triangle_distribution(
        alpha_1 * v ** 2 + alpha_2 * omega ** 2, M
    )
    omega_hat = omega + sample_triangle_distribution(
        alpha_3 * v ** 2 + alpha_4 * omega ** 2, M
    )
    gamma_hat = sample_triangle_distribution(
        alpha_5 * v ** 2 + alpha_6 * omega ** 2, M
    )

    omega_hat = np.where(np.abs(omega_hat) < 1e-9, 1e-9, omega_hat)

    x, y, theta = x_prev[:, 0], x_prev[:, 1], x_prev[:, 2]
    frac = v_hat / omega_hat

    x_dash = x - frac * np.sin(theta) + frac * np.sin(theta + omega_hat * dt)
    y_dash = y + frac * np.cos(theta) - frac * np.cos(theta + omega_hat * dt)
    theta_dash = theta + (omega_hat + gamma_hat) * dt

    return np.vstack((x_dash, y_dash, theta_dash)).T