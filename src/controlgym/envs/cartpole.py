from typing import Optional

from matplotlib import pyplot as plt
import numpy as np
import control as ct
import control.matlab as ctm
import pygame
from pygame import gfxdraw

from gym.envs.classic_control.cartpole import CartPoleEnv


class FixedCartPoleEnv(CartPoleEnv):
    # We fix rendering to be able to change x_threshold
    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        polewidth = scale * 0.1
        polelen = scale * (2 * self.length)
        cartwidth = scale * 0.5
        cartheight = scale * 0.3

        if self.state is None:
            return None

        x = self.state

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.state[2] = self.np_random.uniform(low=-np.pi / 6, high=np.pi / 6)
        self.steps_beyond_done = None
        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}


class CartPoleDownEnv(FixedCartPoleEnv):
    def __init__(self):
        super().__init__()
        self.theta_threshold_radians = np.inf

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.state[2] = np.pi + self.np_random.uniform(low=-np.pi / 6, high=np.pi / 6)
        self.steps_beyond_done = None
        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}


def get_cartpole_linear_model(env: CartPoleEnv) -> ct.StateSpace:
    M = env.masscart
    m = env.masspole
    L = env.length
    g = env.gravity
    dt = env.tau  # Step time
    F = env.force_mag

    A = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, -m * g / M, 0],
            [0, 0, 0, 1],
            [0, 0, -(m + M) * g / (M * L), 0],
        ]
    )

    B = np.array([[0], [F / M], [0], [F / (M * L)]])

    C = np.eye(4)

    D = np.zeros((4, 1))

    continuous_model = ct.ss(A, B, C, D)
    return ctm.c2d(continuous_model, Ts=dt)


def get_cartpole_linear_lqr_gain(model: ct.StateSpace):
    Qx1 = np.diag([10, 1, 100, 1000])
    Qu1a = np.diag([0.1])
    K, _, _ = ct.lqr(model, Qx1, Qu1a)
    return np.array(K)


def cartpole_rhs(t, x, u, params: dict):
    # Parameter setup
    gravity = params.get("gravity")
    length = params.get("length")
    masspole = params.get("masspole")
    total_mass = params.get("total_mass")
    polemass_length = params.get("polemass_length")
    force_mag = params.get("force_mag")
    dt = params.get("dt")

    # Map the states into local variable names

    x, x_dot, theta, theta_dot = x[0], x[1], x[2], x[3]
    force = force_mag * np.clip(u[0], -1, 1)
    costheta = np.cos(theta)
    sintheta = np.sin(theta)

    # For the interested reader:
    # https://coneural.org/florian/papers/05_cart_pole.pdf
    temp = (force + polemass_length * theta_dot**2 * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / (
        length * (4.0 / 3.0 - masspole * costheta**2 / total_mass)
    )
    xacc = temp - polemass_length * thetaacc * costheta / total_mass

    x = x + x_dot * dt
    x_dot = x_dot + xacc * dt
    theta = theta + theta_dot * dt
    theta_dot = theta_dot + thetaacc * dt

    return np.array([x, x_dot, theta, theta_dot])

def plot_cartpole_history(time, xs, thetas, pole_lenght:float, x_threshold:float):
    ax = plt.figure().add_subplot(projection="3d")

    x_cart = xs
    y_cart = np.zeros_like(xs)
    ax.plot(time, x_cart, y_cart, label="Cart")

    x_pole = x_cart + pole_lenght * np.sin(thetas)
    y_pole = y_cart + pole_lenght * np.cos(thetas)
    ax.plot(time, x_pole, y_pole, label="Pole")

    ax.legend()
    ax.set_ylim((-x_threshold, x_threshold))
    ax.set_zlim((-1, 1))

    plt.tight_layout()
    plt.show()
