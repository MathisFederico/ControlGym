from typing import Any, List
import pygame
import numpy as np
from gym.utils.play import play
from hebg import HEBGraph, FeatureCondition, Behavior

from controlgym.envs.cartpole import (
    CartPoleDownEnv,
    get_cartpole_linear_model,
    get_cartpole_linear_lqr_gain,
)
from controlgym.agents.control import LinearControlAgent


class DiscreteLinearControlAgent(LinearControlAgent):
    def act(self, observation):
        continuous_action = LinearControlAgent.act(self, observation)
        return int(continuous_action[0] > 0)


env = CartPoleDownEnv()
call_history = []


class KeepSteadyCart(Behavior):
    def __init__(self) -> None:
        super().__init__(name="Keep steady cart")
        self.last_action = 0

    def __call__(self, observation, *args, **kwargs):
        call_history.append(self.name)
        self.last_action = 1 - self.last_action
        return self.last_action


class IsCartTooFast(FeatureCondition):
    def __init__(self) -> None:
        super().__init__(name="Is the cart too fast ?")

    def __call__(self, observation):
        x, x_dot, theta, theta_dot = observation
        return int(np.abs(x_dot) > 10 * env.force_mag * env.tau)


class SlowDownCart(Behavior):
    def __init__(self) -> None:
        super().__init__(name="Slow down the cart")

    def __call__(self, observation, *args, **kwargs):
        call_history.append(self.name)
        x, x_dot, theta, theta_dot = observation
        return np.array(x_dot < 0, dtype=np.int32)


class CenterCart(Behavior):
    def __init__(self) -> None:
        super().__init__(name="Center the cart")

    def __call__(self, observation, *args, **kwargs):
        call_history.append(self.name)
        x, x_dot, theta, theta_dot = observation
        return np.array(x < 0, dtype=np.int32)


class IsCartCentered(FeatureCondition):
    def __init__(self) -> None:
        super().__init__(name="Is the cart centered ?")

    def __call__(self, observation):
        x, x_dot, theta, theta_dot = observation
        return int(np.abs(x) < 0.4 * env.x_threshold)


class KeepSteadyCenteredCart(Behavior):
    def __init__(self) -> None:
        super().__init__(name="Keep steady cart near the center")

    def build_graph(self) -> HEBGraph:
        graph = HEBGraph(self)

        cart_too_fast = IsCartTooFast()
        graph.add_edge(cart_too_fast, SlowDownCart(), index=1)

        cart_centered = IsCartCentered()
        graph.add_edge(cart_too_fast, cart_centered, index=0)
        graph.add_edge(cart_centered, CenterCart(), index=0)
        graph.add_edge(cart_centered, KeepSteadyCart(), index=1)

        return graph


class StabilizeAroundUpEquilibrium(Behavior):
    def __init__(self) -> None:
        super().__init__(name="Stabilize around up equilibrium")
        model = get_cartpole_linear_model(env)
        K = get_cartpole_linear_lqr_gain(model)
        self.control_agent = DiscreteLinearControlAgent(K)

    def __call__(self, observation, *args, **kwargs):
        call_history.append(self.name)
        return self.control_agent.act(observation)


class IsPoleSlow(FeatureCondition):
    def __init__(self) -> None:
        super().__init__(name="Is the pole slow ?")

    def __call__(self, observation):
        x, x_dot, theta, theta_dot = observation
        return np.abs(theta_dot) < 0.5


class IsPoleVertical(FeatureCondition):
    def __init__(self) -> None:
        super().__init__(name="Is the pole vertical ?")

    def __call__(self, observation):
        x, x_dot, theta, theta_dot = observation
        return np.abs(np.cos(theta)) > np.sqrt(2) / 2


class SpeedDownPole(Behavior):
    def __init__(self) -> None:
        super().__init__(name="Speed down pole")

    def __call__(self, observation, *args, **kwargs):
        call_history.append(self.name)
        x, x_dot, theta, theta_dot = observation
        return np.array(theta_dot * np.cos(theta) >= 0, dtype=np.int32)


class SpeedUpPole(Behavior):
    def __init__(self) -> None:
        super().__init__(name="Speed up pole")

    def __call__(self, observation, *args, **kwargs):
        call_history.append(self.name)
        x, x_dot, theta, theta_dot = observation
        return np.array(theta_dot * np.cos(theta) < 0, dtype=np.int32)


class SteadySpeedDownPole(Behavior):
    def __init__(self) -> None:
        super().__init__(name="Speed down pole with a steady cart")

    def build_graph(self) -> HEBGraph:
        graph = HEBGraph(self)
        is_pole_slow = IsPoleSlow()
        steady = KeepSteadyCenteredCart()
        graph.add_edge(is_pole_slow, steady, index=1)

        cart_too_fast = IsCartTooFast()
        graph.add_edge(is_pole_slow, cart_too_fast, index=0)
        graph.add_edge(cart_too_fast, SlowDownCart(), index=1)

        pole_vertical = IsPoleVertical()
        graph.add_edge(cart_too_fast, pole_vertical, index=0)
        graph.add_edge(pole_vertical, SpeedDownPole(), index=1)
        graph.add_edge(pole_vertical, steady, index=0)

        return graph


class SteadySpeedUpPole(Behavior):
    def __init__(self) -> None:
        super().__init__(name="Speed up pole with a steady cart")

    def build_graph(self) -> HEBGraph:
        graph = HEBGraph(self)

        is_pole_slow = IsPoleSlow()
        steady = KeepSteadyCenteredCart()
        graph.add_edge(is_pole_slow, steady, index=1)

        cart_too_fast = IsCartTooFast()
        graph.add_edge(is_pole_slow, cart_too_fast, index=0)
        graph.add_edge(cart_too_fast, SlowDownCart(), index=1)

        pole_vertical = IsPoleVertical()
        graph.add_edge(cart_too_fast, pole_vertical, index=0)
        graph.add_edge(pole_vertical, SpeedUpPole(), index=1)
        graph.add_edge(pole_vertical, steady, index=0)
        return graph


class PoleMomentumLevel(FeatureCondition):
    def __init__(self) -> None:
        super().__init__(name="Pole momentum level ?")
        self.g = env.gravity
        self.l = env.length
        self.m = env.masspole
        self.M = env.masscart

    def __call__(self, observation):
        x, x_dot, theta, theta_dot = observation
        normed_theta = (theta + np.pi) % (2 * np.pi) - np.pi
        z = np.cos(normed_theta) + 1
        energy = (
            (self.M + self.m) * np.square(x_dot) / 2
            + self.m * np.square(self.l * theta_dot) / 2
            + self.m * self.l * z * x_dot * theta_dot
            + self.m * self.g * self.l * z
        )
        required_potential_inertia = 2 * self.m * self.g * self.l
        print(
            energy / required_potential_inertia,
            theta_dot * normed_theta < 0,
        )
        if energy < 0.95 * required_potential_inertia:
            return 0  # Too slow
        if energy > 1.1 * required_potential_inertia:
            return 2  # Too fast
        return 1


class SwingToUpEquilibrium(Behavior):
    def __init__(self) -> None:
        super().__init__(name="Swing to up equilibrium")

    def build_graph(self) -> HEBGraph:
        graph = HEBGraph(self)
        inertia_level = PoleMomentumLevel()
        graph.add_edge(inertia_level, SteadySpeedUpPole(), index=0)
        graph.add_edge(inertia_level, SteadySpeedDownPole(), index=2)
        graph.add_edge(inertia_level, KeepSteadyCenteredCart(), index=1)
        return graph


class IsCloseToUpEquilibrium(FeatureCondition):
    def __init__(self) -> None:
        super().__init__(name="Is state close to up equilibrium ?")

    def __call__(self, observation):
        x, x_dot, theta, theta_dot = observation
        normed_theta = (theta + np.pi) % (2 * np.pi) - np.pi
        close_in_theta = np.abs(normed_theta) < np.pi / 5
        close_in_x = np.abs(x) < 2
        not_too_fast = np.abs(theta_dot) < 3 or (
            theta_dot * normed_theta < 0 and np.abs(theta_dot) < 6
        )
        return close_in_theta and not_too_fast


class SwingUpAndStabilize(Behavior):
    def __init__(self) -> None:
        super().__init__(name="Swing to up equilibrium and stabilize")

    def __call__(self, observation, *args, **kwargs):
        print(call_history[-5:])
        return super().__call__(observation, *args, **kwargs)

    def build_graph(self) -> HEBGraph:
        graph = HEBGraph(self)
        is_close_to_up_eq = IsCloseToUpEquilibrium()
        graph.add_edge(is_close_to_up_eq, SwingToUpEquilibrium(), index=int(False))
        graph.add_edge(
            is_close_to_up_eq, StabilizeAroundUpEquilibrium(), index=int(True)
        )
        return graph


full_auto = SwingUpAndStabilize()


def act_func_with_behavior(behavior: Behavior):
    return lambda: behavior(env.state)


stab_behavior = StabilizeAroundUpEquilibrium()
steady_behavior = KeepSteadyCenteredCart()


def steady_stab() -> int:
    x, x_dot, theta, theta_dot = env.state
    m, M = env.masspole, env.masscart
    g = env.gravity
    l = env.length
    normed_theta = (theta + np.pi) % (2 * np.pi) - np.pi
    z = np.cos(normed_theta) + 1
    energy = (
        (M + m) * np.square(x_dot) / 2
        + m * np.square(l * theta_dot) / 2
        + m * l * z * x_dot * theta_dot
        + m * g * l * z
    )
    required_potential_inertia = 2 * m * g * l
    # print(
    #     energy / required_potential_inertia,
    #     theta_dot * normed_theta < 0,
    # )
    if (
        np.abs(normed_theta) < np.pi / 5
        and (theta_dot * normed_theta < 0 or np.abs(theta_dot) < 2)
        and (np.abs(x_dot) < 3 or x * x_dot < 0)
    ):
        return stab_behavior(env.state)
    return steady_behavior(env.state)


class CallDict(dict):
    def get(self, __key, __default):
        val = super().get(__key, __default)
        if callable(val):
            val = val()
        return val


speed_down_behavior = SteadySpeedDownPole()
speed_up_behavior = SteadySpeedUpPole()
mapping = CallDict()
mapping[(pygame.K_LEFT,)] = 0
mapping[(pygame.K_RIGHT,)] = 1
mapping[(pygame.K_UP,)] = act_func_with_behavior(speed_up_behavior)
mapping[(pygame.K_DOWN,)] = act_func_with_behavior(speed_down_behavior)
mapping[(pygame.K_SPACE,)] = act_func_with_behavior(full_auto)
mapping[tuple()] = steady_stab
play(env, keys_to_action=mapping)
