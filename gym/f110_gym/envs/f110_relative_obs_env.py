import math
import numpy as np
import gymnasium as gym

from f110_gym.envs.f110_env import F110Env
from f110_gym.envs.utils import AngleOp
from f110_gym import ThrottledPrinter

class F110RelativeObsEnv(F110Env):
    """
    F110 environment with relative observations.
    The agent receives observations relative to the track, not absolute.
    """

    metadata = {'render_modes': ['human', 'human_fast'], 'render_fps': 300}

    def __init__(self,
                 horizon_mode='',
                 steps_horizon=0,
                 delta_s_between_horizon_steps=0.0,
                 **kwargs):

        super().__init__(**kwargs)

        if horizon_mode == '':
            raise ValueError("horizon_mode must be specified for F110RelativeObsEnv.")
        self.horizon_mode = horizon_mode
        if horizon_mode not in ['descriptive', 'stats_aggregated']:
            raise ValueError(f"Unknown horizon_mode: {horizon_mode}. Supported modes are 'descriptive' and 'stats_aggregated'.")

        if steps_horizon <= 0:
            raise ValueError("steps_horizon must be greater than 0 for F110RelativeObsEnv.")
        self.steps_horizon = steps_horizon

        if delta_s_between_horizon_steps <= 0:
            raise ValueError("delta_s_between_horizon_steps must be greater than 0 for F110RelativeObsEnv.")
        self.delta_s_between_horizon_steps = delta_s_between_horizon_steps

        state_single_agent_low = [-np.inf, -np.inf, -np.inf, -np.inf, -np.pi, -np.inf, 0.0]
        state_single_agent_high = [+np.inf, +np.inf, +np.inf, +np.inf, np.pi, +np.inf, +np.inf]

        if self.horizon_mode == 'descriptive':
            additional_n_states = 5 * steps_horizon
            add_single_agent_low = [-np.inf, -np.inf, -np.inf, -np.pi, 0.0] * steps_horizon
            add_single_agent_high = [+np.inf, +np.inf, +np.inf, np.pi, +np.inf] * steps_horizon
        elif self.horizon_mode == 'stats_aggregated':
            additional_n_states = 2 * 5
            add_single_agent_low = [-np.inf, 0.0] * 5
            add_single_agent_high = [+np.inf, +np.inf] * 5
        else:
            raise ValueError("Unknown horizon_mode: {self.horizon_mode}.")

        single_agent_low = np.array(state_single_agent_low + add_single_agent_low, dtype=np.float32)
        single_agent_high = np.array(state_single_agent_high + add_single_agent_high, dtype=np.float32)

        obs_low = np.tile(single_agent_low, (self.num_agents, 1))
        obs_high = np.tile(single_agent_high, (self.num_agents, 1))

        self.observation_space = gym.spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(self.num_agents, 7 + additional_n_states),
            dtype=np.float32
        )

        print('F110RelativeObsEnv initialized')

    def get_relative_obs(self, obs):
        """
        Some choices on the frenet coordinate system:
        - `d` is positive to the right of the track
        - `body_sleep_angle` follows the counter-clockwise convention, i.e., vy is rotated counter-clockwise by
          90 degrees wrt the car's heading (where vx is in the same direction as the car's heading)
        - `yaw` is positive when the car is turning right wrt the tangent of the raceline at the current position
          (i.e., angle starting from the raceline to the car's heading is positive when clockwise, and of course the
          absolute difference is less than pi)
        """

        s = self.ego_s
        d = self.ego_d
        k = self.raceline.curvature_spline(s)
        vx = obs[self.ego_idx, 3]
        body_slip_angle = obs[self.ego_idx, 6]
        vy = vx * math.tan(body_slip_angle)
        yaw_absolute = obs[self.ego_idx, 4]
        yaw_raceline = self.raceline.heading_spline(s)
        yaw = AngleOp.angle_diff(yaw_absolute, yaw_raceline)
        yaw_rate = obs[self.ego_idx, 5]
        x, y = self.raceline.to_cartesian(s, d)
        x_next, y_next = self.raceline.to_cartesian(s + self.delta_s_between_horizon_steps, 0.0)
        l2_dist_next_point = math.sqrt((x_next - x) ** 2 + (y_next - y) ** 2)

        state = [d, k, vx, vy, yaw, yaw_rate, l2_dist_next_point]
        population = {'d_right': [], 'd_left': [], 'k_future': [], 'delta_yaw': [], 'v_speed_profile': []}

        # for N points in the future (horizon), we can get the relative state:
        # delta_s (is implicit), d_right, d_left, k_future, delta_yaw, v_speed_profile
        for i in range(self.steps_horizon):
            # We don't need to worry about `s` wrapping, its handling is already implemented in the `raceline` object.
            future_s = s + (i + 1) * self.delta_s_between_horizon_steps

            d_right = self.raceline.width_right_spline(future_s)
            d_left = self.raceline.width_left_spline(future_s)
            k_future = self.raceline.curvature_spline(future_s)
            # At every future point, we compute the relative angle between the nearest index raceline heading and
            # the one at the future point. Same reference system as above for `yaw`.
            delta_yaw = AngleOp.angle_diff(
                self.raceline.heading_spline(future_s),
                yaw_raceline
            )
            v_speed_profile = self.raceline.speed_spline(future_s)

            population['d_right'].append(d_right)
            population['d_left'].append(d_left)
            population['k_future'].append(k_future)
            population['delta_yaw'].append(delta_yaw)
            population['v_speed_profile'].append(v_speed_profile)

        if self.horizon_mode == 'descriptive':
            # Descriptive horizon: we concatenate the states for each point in the horizon
            future_state = lambda j: [population['d_right'][j], population['d_left'][j], population['k_future'][j],
                                      population['delta_yaw'][j], population['v_speed_profile'][j]]

            for i in range(self.steps_horizon):
                state += future_state(i)
        elif self.horizon_mode == 'stats_aggregated':
            # Stats aggregated horizon: we compute the mean and std for each of the states in the horizon
            state += [
                np.mean(population['d_right']),
                np.std(population['d_right']),
                np.mean(population['d_left']),
                np.std(population['d_left']),
                np.mean(population['k_future']),
                np.std(population['k_future']),
                np.mean(population['delta_yaw']),
                np.std(population['delta_yaw']),
                np.mean(population['v_speed_profile']),
                np.std(population['v_speed_profile'])
            ]

        return np.array([state], dtype=np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        obs = self.get_relative_obs(obs)

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        obs = self.get_relative_obs(obs)

        return obs, info
