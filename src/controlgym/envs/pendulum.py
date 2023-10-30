from typing import Optional

import control as ct
import control.matlab as ctm
import numpy as np
import matplotlib.pyplot as plt

from gym.envs.classic_control.pendulum import PendulumEnv


from controlgym.plotting import Multiple


class UpPendulumEnv(PendulumEnv):
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        bounds = np.array([np.pi / 6, 0.1])
        self.state = self.np_random.uniform(low=-bounds, high=bounds, size=(2,))
        self.last_u = None
        if not return_info:
            return self._get_obs()
        else:
            return self._get_obs(), {}


class DownPendulumEnv(PendulumEnv):
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(2,))
        self.state[0] += np.pi
        self.last_u = None
        if not return_info:
            return self._get_obs()
        else:
            return self._get_obs(), {}


class RandomPendulumEnv(PendulumEnv):
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        bounds = np.array([np.pi, self.max_speed])
        self.state = self.np_random.uniform(low=-bounds, high=bounds)
        self.last_u = None
        if not return_info:
            return self._get_obs()
        else:
            return self._get_obs(), {}


def thetas_from_obs(observations):
    xs = observations[:, 0]
    ys = observations[:, 1]
    thetadots = observations[:, 2]
    thetas = np.arctan2(ys, xs)
    return thetas, thetadots


def plot_pendulum_history(time, command, thetas, thetadots, max_torque=np.inf):
    _, axes = plt.subplots(5, 1, sharex=True, figsize=(6.4, 9.8))
    axes: list[plt.Axes] = axes
    axes[0].plot(time, thetas, label="Theta")
    axes[0].set_title(r"$\theta$")
    den = 1 + int(2 * np.pi / (np.max(thetas) - np.min(thetas)))
    pi4multiple = Multiple(den, np.pi)
    axes[0].yaxis.set_major_formatter(pi4multiple.formatter())
    axes[0].yaxis.set_major_locator(pi4multiple.locator())
    # axes[0].set_ylim((-np.pi, np.pi))
    axes[0].grid()
    axes[1].plot(time, thetadots, label="ThetaDot", color="g")
    axes[1].set_title(r"$\dot\theta$")
    # axes[1].set_ylim((-1.2*env.max_speed, 1.2*env.max_speed))
    axes[1].grid()
    axes[2].plot(time, command, label="Command", color="r", linestyle=":")
    axes[2].plot(
        time, np.clip(command, -max_torque, max_torque), label="Command", color="r"
    )
    # axes[2].set_ylim((-1.2*env.max_torque, 1.2*env.max_torque))
    axes[2].set_title("u")
    axes[2].grid()

    axes[3].plot(time, np.cos(thetas), label="x")
    axes[3].set_title(r"x")
    axes[3].set_ylim((-1, 1))
    axes[3].grid()

    axes[4].plot(time, np.sin(thetas), label="y")
    axes[4].set_title(r"y")
    axes[4].set_ylim((-1, 1))
    axes[4].grid()

    plt.xlabel("Time")
    plt.tight_layout()
    plt.show()


def get_pendulum_linear_model(env: PendulumEnv, costheta: float = 1):
    A = np.array(
        [
            [0, 1],
            [costheta * env.g / env.l, 0],
        ]
    )

    B = np.array([[0], [-3 / (env.m * env.l**2)]])

    C = np.eye(2)

    D = np.zeros((2, 1))

    continuous_model = ct.ss(A, B, C, D)
    return ctm.c2d(continuous_model, Ts=env.dt)


def get_pendulum_linear_lqr_gain(model: ct.StateSpace):
    Qx1 = np.diag([1, 0.1])
    Qu1a = np.diag([0.001])
    K, _, _ = ct.lqr(model, Qx1, Qu1a)
    return np.array(K)


def pendulum_rhs(t, x, u, params: dict):
    # Parameter setup
    gravity = params.get("g", 9.81)
    length = params.get("l", 1.0)
    mass = params.get("m", 1.0)
    dt = params.get("dt", 0.05)
    max_speed = params.get("max_speed", 5.0)

    # Map the states into local variable names
    th = x[0]
    thdot = x[1]

    # Compute the control action
    u_0 = u[0]

    # Compute the discrete updates
    dthdot = 3 * gravity / (2 * length) * np.sin(th) + 3.0 / (mass * length**2) * u_0
    dth = thdot + dthdot * dt
    dth = np.clip(dth, -max_speed, max_speed)

    return np.array([dth, dthdot])


def pendulum_obs_to_state(observation) -> np.ndarray:
    x, y, theta_dot = observation
    theta = np.arctan2(y, x)
    return np.array([theta, theta_dot])
