# gym imports
import gymnasium as gym

# base classes
from f110_gym.envs.base_classes import Simulator, Integrator
from f110_gym.envs.raceline import Raceline
from f110_gym.envs.utils import AngleOp
from f110_gym import ThrottledPrinter

# others
import numpy as np
import os
import math
import time
import yaml
from PIL import Image

# gl
import pyglet

pyglet.options['debug_gl'] = False

# constants

# rendering
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

class F110Env(gym.Env):
    """
    OpenAI gym environment for F1TENTH
    
    Env should be initialized by calling gym.make('f110_gym:f110-v0', **kwargs)

    Args:
        kwargs:
            seed (int, default=12345): seed for random state and reproducibility
            
            map (str, default='vegas'): name of the map used for the environment. Currently, available environments include: 'berlin', 'vegas', 'skirk'. You could use a string of the absolute path to the yaml file of your custom map.
        
            map_ext (str, default='png'): image extension of the map image file. For example 'png', 'pgm'
        
            params (dict, default={'mu': 1.0489, 'C_Sf':, 'C_Sr':, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch':7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}): dictionary of vehicle parameters.
            mu: surface friction coefficient
            C_Sf: Cornering stiffness coefficient, front
            C_Sr: Cornering stiffness coefficient, rear
            lf: Distance from center of gravity to front axle
            lr: Distance from center of gravity to rear axle
            h: Height of center of gravity
            m: Total mass of the vehicle
            I: Moment of inertial of the entire vehicle about the z axis
            s_min: Minimum steering angle constraint
            s_max: Maximum steering angle constraint
            sv_min: Minimum steering velocity constraint
            sv_max: Maximum steering velocity constraint
            v_switch: Switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
            a_max: Maximum longitudinal acceleration
            v_min: Minimum longitudinal velocity
            v_max: Maximum longitudinal velocity
            width: width of the vehicle in meters
            length: length of the vehicle in meters

            num_agents (int, default=2): number of agents in the environment

            timestep (float, default=0.01): physics timestep

            ego_idx (int, default=0): ego's index in list of agents
    """
    metadata = {'render_modes': ['human', 'human_fast'], 'render_fps': 300}

    # rendering
    renderer = None
    current_obs = None
    render_callbacks = []

    def __init__(self, **kwargs):
        self.throttled_printer = ThrottledPrinter(min_interval=0.5)
        self.unthrottled_printer = ThrottledPrinter(min_interval=1e-9)

        try:
            self.render_mode = kwargs['render_mode']
            print('Render mode:', self.render_mode)
        except:
            self.render_mode = None
        try:
            self.disable_scan_simulator = kwargs['disable_scan_simulator']
        except:
            self.disable_scan_simulator = False

        # kwargs extraction
        try:
            self.seed = kwargs['seed']
        except:
            self.seed = 12345
        try:
            self.map_name = kwargs['map']
            self.map_path = self.map_name + '.yaml'
        except:
            raise ValueError('Map name not provided. Please provide a map name.')

        try:
            self.map_ext = kwargs['map_ext']
        except:
            self.map_ext = '.png'

        if not os.path.exists(self.map_name + '.yaml') or not os.path.exists(self.map_name + self.map_ext):
            raise FileNotFoundError(f"Map file {self.map_name + '.yaml'} or image file {self.map_name + self.map_ext} "
                                    f"not found.")
        with open(self.map_name + '.yaml', 'r') as file:
            try:
                map_yaml = yaml.safe_load(file)
                self.resolution = map_yaml['resolution']
                self.origin_x = map_yaml['origin'][0]
                self.origin_y = map_yaml['origin'][1]
            except yaml.YAMLError as exc:
                print(exc)
                raise ValueError(f"Error loading map file {self.map_name + '.yaml'}")

        self.map_img = np.array(Image.open(self.map_name + self.map_ext).transpose(Image.FLIP_TOP_BOTTOM))
        # grayscale -> binary
        self.map_img[self.map_img <= 128.] = 0.
        self.map_img[self.map_img > 128.] = 255.

        # show_map(self.map_name + self.map_ext)

        try:
            self.raceline_path = kwargs['raceline_path']
        except:
            raise ValueError("Raceline path not provided. Please provide a raceline path.")

        try:
            self.params = kwargs['params']
        except:
            self.params = {'mu': 1.0489,
                           'C_Sf': 4.718,
                           'C_Sr': 5.4562,
                           'lf': 0.15875,
                           'lr': 0.17145,
                           'h': 0.074,
                           'm': 3.74,
                           'I': 0.04712,
                           's_min': -0.46,
                           's_max': 0.46,
                           'sv_min': -3.2,
                           'sv_max': 3.2,
                           'v_switch': 7.319,
                           'a_max': 9.51,
                           'v_min':-5.0,
                           'v_max': 20.0,
                           'width': 0.31,
                           'length': 0.58}

        # simulation parameters
        try:
            self.num_agents = kwargs['num_agents']
            if self.num_agents <= 0:
                raise ValueError('Number of agents must be positive')
        except:
            self.num_agents = 1

        try:
            self.timestep = kwargs['timestep']
        except:
            self.timestep = 0.01

        # default ego index
        # try:
        #     self.ego_idx = kwargs['ego_idx']
        # except:
        self.ego_idx = 0

        # default integrator
        try:
            self.integrator = kwargs['integrator']
        except:
            self.integrator = Integrator.RK4

        try:
            self.points_in_foreground = kwargs['points_in_foreground']
        except:
            self.points_in_foreground = False

        try:
            self.reward_function = kwargs['reward_function']
        except:
            self.reward_function = 's'
        try:
            self.w_s = kwargs['w_s']
            self.w_d = kwargs['w_d']
            if self.w_s < 0.0 or self.w_d < 0.0:
                raise ValueError("Reward weights must be non-negative.")
        except:
            raise ValueError("Reward weights not provided. Please provide w_s and w_d.")
        try:
            self.lower_bound_penalty_yaw_collision = kwargs['lower_bound_penalty_yaw_collision']
            self.upper_bound_penalty_yaw_collision = kwargs['upper_bound_penalty_yaw_collision']
            if self.lower_bound_penalty_yaw_collision < 0.0 or self.upper_bound_penalty_yaw_collision < 0.0:
                raise ValueError("Yaw collision penalty bounds must be non-negative.")
            if self.lower_bound_penalty_yaw_collision > self.upper_bound_penalty_yaw_collision:
                raise ValueError("Lower bound penalty yaw collision must be less than upper bound penalty yaw collision.")
        except:
            raise ValueError("Yaw collision penalty bounds not provided. Please provide "
                             "lower_bound_penalty_yaw_collision and upper_bound_penalty_yaw_collision.")

        self.m_yaw_penalty = (self.upper_bound_penalty_yaw_collision - self.lower_bound_penalty_yaw_collision) / np.pi
        self.q_yaw_penalty = self.lower_bound_penalty_yaw_collision

        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
        # Create a single row vector for one agent
        single_agent_low = np.array(
            [-np.inf, -np.inf, self.params['s_min'], self.params['v_min'], -np.pi, self.params['sv_min'], -np.inf], dtype=np.float32)
        single_agent_high = np.array(
            [+np.inf, +np.inf, self.params['s_max'], self.params['v_max'], np.pi, self.params['sv_max'], +np.inf], dtype=np.float32)

        # Duplicate for all agents to match the shape (num_agents, 7)
        obs_low = np.tile(single_agent_low, (self.num_agents, 1))
        obs_high = np.tile(single_agent_high, (self.num_agents, 1))

        # Now create the observation space with the proper dimensions
        self.observation_space = gym.spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(self.num_agents, 7),
            dtype=np.float32
        )

        single_agent_low = np.array([self.params['s_min'], 0.0], dtype=np.float32)
        single_agent_high = np.array([self.params['s_max'], np.inf], dtype=np.float32)

        action_low = np.tile(single_agent_low, (self.num_agents, 1))
        action_high = np.tile(single_agent_high, (self.num_agents, 1))

        self.action_space = gym.spaces.Box(
            low=action_low,
            high=action_high,
            shape=(self.num_agents, 2),
            dtype=np.float32
        )

        self.poses_x = []
        self.poses_y = []
        self.poses_theta = []
        self.polygon_collisions = np.zeros((self.num_agents, ))
        self.lidar_collisions = np.zeros((self.num_agents, ))
        self.off_track = np.zeros((self.num_agents, ))

        # race info
        self.lap_times = np.zeros((self.num_agents, ))
        self.lap_counts = np.zeros((self.num_agents, ))

        # initiate stuff
        self.sim = Simulator(self.params, self.num_agents, self.seed, time_step=self.timestep,
                             integrator=self.integrator, disable_scan_simulator=self.disable_scan_simulator)
        self.sim.set_map(self.map_path, self.map_ext)

        self.total_steps = 0
        self.lap_steps = 0
        self.last_lap_time = 0.0

        self.raceline = Raceline(self.raceline_path)
        self.current_observation = None
        self.previous_s = None
        self.ego_s = None
        self.ego_d = None
        self.ego_lap_count = 0
        self.raceline_zones_rendered = False

        # stateful observations for rendering
        self.render_obs = None

    def _get_obs(self):
        """
        Get the current observation of the environment

        Args:
            None

        Returns:
            obs (np.ndarray): current observation of the environment
        """
        obs = np.zeros((self.num_agents, 7), dtype=np.float32)
        for i in range(self.num_agents):
            obs[i, :] = self.sim.agents[i].state

        return obs

    def _get_info(self):
        """
        Get the current information of the environment

        Args:
            None

        Returns:
            info (dict): current information of the environment
        """
        info = {
            'polygon_collisions': self.polygon_collisions,
            'lidar_collisions': self.lidar_collisions
        }
        return info

    def close(self):
        """
        Finalizer, does cleanup
        """
        pass

    def is_inside_track(self, x, y):
        """
        Checks if a given (x, y) coordinate is inside the track (free space).

        Args:
            x (float): x-coordinate in world space
            y (float): y-coordinate in world space

        Returns:
            bool: True if the point is inside the track (free space), False otherwise
        """
        # Convert to pixel coordinates
        x_pixel = int((x - self.origin_x) / self.resolution)
        y_pixel = int((y - self.origin_y) / self.resolution)

        # Check if the pixel is within the bounds of the map image
        if 0 <= x_pixel < self.map_img.shape[1] and 0 <= y_pixel < self.map_img.shape[0]:
            return self.map_img[y_pixel, x_pixel] != 0
        else:
            return False

    def _check_done(self, s):
        """
        Check if the current rollout is done
        
        Args:
            obs (np.array): observation of the current step
        """

        # This check is being done only for ego
        lap_info = self.raceline.is_lap_completed(s)
        lap_completed = lap_info['lap_completed']
        lap_orientation = lap_info['lap_orientation']

        if lap_completed:
            self.last_lap_time = self.lap_steps * self.timestep
            # Overwriting real cpu time with the simulation time
            lap_info['lap_time'] = self.last_lap_time
            self.lap_steps = 0

        if lap_completed and lap_orientation == 'forward':
            self.ego_lap_count += 1

        for i in range(self.num_agents):
            self.off_track[i] = 0.
            if not self.is_inside_track(self.poses_x[i], self.poses_y[i]):
                self.off_track[i] = 1.
                self.throttled_printer.print(f"Agent {i} is off track: ({self.poses_x[i]}, {self.poses_y[i]})",
                                             'yellow')

            if F110Env.renderer is not None:
                F110Env.renderer.draw_point(self.poses_x[i], self.poses_y[i], size=4, track=True)

        done_poly_collisions = bool(self.polygon_collisions[self.ego_idx])
        done_laps_ego = bool(self.ego_lap_count >= 2)
        done_off_track_ego = bool(self.off_track[self.ego_idx])
        done_shapely_collisions = bool(self.raceline.is_colliding_with_track(self.poses_x[self.ego_idx],
                                                                             self.poses_y[self.ego_idx],
                                                                             self.poses_theta[self.ego_idx],
                                                                             self.params['width'],
                                                                             self.params['length']))

        done = done_off_track_ego or done_poly_collisions or done_shapely_collisions

        return done, done_poly_collisions, done_laps_ego, done_off_track_ego, done_shapely_collisions, lap_info

    def _update_state(self, obs_dict):
        """
        Update the env's states according to observations
        
        Args:
            obs_dict (dict): dictionary of observation

        Returns:
            None
        """
        self.poses_x = obs_dict['poses_x']
        self.poses_y = obs_dict['poses_y']
        self.poses_theta = obs_dict['poses_theta']
        self.polygon_collisions = obs_dict['polygon_collisions']
        self.lidar_collisions = obs_dict['lidar_collisions']

    def get_frenet_state(self):
        """
        Get the Frenet state of the ego vehicle

        Args:
            None

        Returns:
            s (float): distance along the raceline
            d (float): lateral distance from the raceline
        """
        return self.ego_s, self.ego_d

    def step(self, action):
        """
        Step function for the gym env

        Args:
            action (np.ndarray(num_agents, 2))

        Returns:
            obs (np.array): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            terminated (bool): if the simulation is terminated
            truncated (bool): if the simulation is truncated
            info (dict): auxiliary information dictionary
        """

        self.total_steps += 1
        self.lap_steps += 1

        old_x, old_y = self.poses_x[self.ego_idx], self.poses_y[self.ego_idx]

        # call simulation step
        obs = self.sim.step(action)

        # update data member
        self._update_state(obs)

        self.current_observation = self._get_obs()
        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
        x, y = self.current_observation[self.ego_idx, 0], self.current_observation[self.ego_idx, 1]

        nearest_index, s, d, status = self.raceline.get_nearest_index(x, y, self.previous_s)

        # check done
        done, done_poly_collisions, done_laps_ego, done_off_track_ego, done_shapely_collisions, lap_info = \
            self._check_done(s)

        if done_shapely_collisions:
            self.unthrottled_printer.print(f'done_shapely_collisions: {done_shapely_collisions}', 'red')

        if self.previous_s is not None:
            delta_s = self.raceline.get_delta_s(s, self.previous_s)
        else:
            # Not projecting on s-spline, so we just use the Euclidean distance between the two points
            delta_s = math.sqrt((x - old_x) ** 2 + (y - old_y) ** 2)

        if abs(delta_s) > 2.0:
            # TODO: test if it ever happens
            self.unthrottled_printer.print(f'delta_s is too large: {delta_s} - id: {self.total_steps}', 'red')

        if self.reward_function == 's':
            reward = self.w_s * delta_s
        elif self.reward_function == 's+d':
            reward = self.w_s * delta_s - self.w_d * abs(d)
        else:
            raise ValueError(f"Unknown reward function: {self.reward_function}")

        terminated = done  # if not recoverable off track or collision with another agent
        truncated = done_laps_ego  # if laps completed

        if done_off_track_ego or done_shapely_collisions:
            # Here we are just assuming negative reward for going off track, we are not considering collisions with
            # other agents
            angle_diff = AngleOp.angle_diff(self.raceline.heading_spline(s), self.poses_theta[self.ego_idx])
            reward -= self.m_yaw_penalty * abs(angle_diff) + self.q_yaw_penalty

        if done_poly_collisions:
            reward -= 1.0

        if done_laps_ego:
            # reward += 10
            # We make the RL model confused by giving it a reward for completing the lap. In the observation space,
            # there is no proxy for lap completion, so that reward would be completely unexpected. Then, let's
            # just truncate the episode.
            # self.throttled_printer.print(f'Ego completed {self.ego_lap_count} laps', 'green')
            pass

        info = self._get_info()
        info.update({'legacy_obs': obs})
        info.update(lap_info)

        self.previous_s = s
        self.ego_s = s
        self.ego_d = d

        self.lap_times[self.ego_idx] = self.last_lap_time
        self.lap_counts[self.ego_idx] = self.ego_lap_count
        obs['lap_times'] = self.lap_times
        obs['lap_counts'] = self.lap_counts

        self.render_obs = {
            'ego_idx': obs['ego_idx'],
            'poses_x': obs['poses_x'],
            'poses_y': obs['poses_y'],
            'poses_theta': obs['poses_theta'],
            'lap_times': obs['lap_times'],
            'lap_counts': obs['lap_counts']
        }

        F110Env.current_obs = obs

        return self.current_observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the gym environment by given poses

        Args:
            poses (np.ndarray (num_agents, 3)): poses to reset agents to

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """

        super().reset(seed=seed)

        poses = np.zeros((self.num_agents, 3))
        if options is not None:
            if 'pose' in options:
                poses = options['pose']
            elif 'poses' in options:
                poses = options['poses']
        else:
            if self.num_agents != 1:
                raise ValueError("Must specify either poses or poses for multiple agents.")

            # We have to sample poses from the observation space
            # Sample s between 0 and raceline total s
            s_start = np.random.uniform(0, self.raceline.total_s)
            left_d = -self.raceline.width_left_spline(s_start)
            right_d = self.raceline.width_right_spline(s_start)

            left_d = left_d + 0.2 if left_d + 0.2 <= 0 else left_d
            right_d = right_d - 0.2 if right_d - 0.2 >= 0 else right_d
            d_start = np.random.uniform(left_d, right_d)

            noise = np.random.normal(0, 0.2)
            yaw_start = AngleOp.normalize_angle(self.raceline.heading_spline(s_start) + noise)
            x_init, y_init = self.raceline.to_cartesian(s_start, d_start)
            poses = np.array([[x_init, y_init, yaw_start]], dtype=np.float32)

        # check that poses are valid and not off track
        for i in range(self.num_agents):
            if not self.is_inside_track(float(poses[i, 0]), float(poses[i, 1])):
                raise gym.error.Error(f"Agent {i} pose is off track: ({poses[i, 0]}, {poses[i, 1]})")

        # reset counters and data members
        self.poses_x = []
        self.poses_y = []
        self.poses_theta = []
        for i in range(self.num_agents):
            self.poses_x.append(poses[i, 0])
            self.poses_y.append(poses[i, 1])
            self.poses_theta.append(poses[i, 2])
        self.polygon_collisions = np.zeros((self.num_agents,))
        self.lidar_collisions = np.zeros((self.num_agents,))
        self.off_track = np.zeros((self.num_agents,))

        self.lap_times = np.zeros((self.num_agents,))
        self.lap_counts = np.zeros((self.num_agents,))

        self.render_obs = None

        # call reset to simulator
        self.sim.reset(poses)

        obs = np.zeros((self.num_agents, 7), dtype=np.float32)
        for i in range(self.num_agents):
            obs[i, 0] = self.poses_x[i]
            obs[i, 1] = self.poses_y[i]
            obs[i, 4] = self.poses_theta[i]

        self.total_steps = 0
        self.lap_steps = 0
        self.last_lap_time = 0.0

        nearest_index, s, d, _ = self.raceline.get_nearest_index(obs[self.ego_idx, 0], obs[self.ego_idx, 1], previous_s=None)
        self.raceline.reset(s)
        self.current_observation = self._get_obs()
        self.previous_s = None
        self.ego_s = s
        self.ego_d = d
        self.ego_lap_count = 0
        self.raceline_zones_rendered = False

        if F110Env.renderer is not None:
            F110Env.renderer.remove_tracked_points()

        return obs, {}

    def update_map(self, map_path, map_ext):
        """
        Updates the map used by simulation

        Args:
            map_path (str): absolute path to the map yaml file
            map_ext (str): extension of the map image file

        Returns:
            None
        """
        self.sim.set_map(map_path, map_ext)

    def update_params(self, params, index=-1):
        """
        Updates the parameters used by simulation for vehicles
        
        Args:
            params (dict): dictionary of parameters
            index (int, default=-1): if >= 0 then only update a specific agent's params

        Returns:
            None
        """
        self.sim.update_params(params, agent_idx=index)

    def add_render_callback(self, callback_func):
        """
        Add extra drawing function to call during rendering.

        Args:
            callback_func (function (EnvRenderer) -> None): custom function to called during render()
        """

        F110Env.render_callbacks.append(callback_func)

    def render(self):
        """
        Renders the environment with pyglet. Use mouse scroll in the window to zoom in/out, use mouse click drag to pan. Shows the agents, the map, current fps (bottom left corner), and the race information near as text.

        Args:
            mode (str, default='human'): rendering mode, currently supports:
                'human': slowed down rendering such that the env is rendered in a way that sim time elapsed is close to real time elapsed
                'human_fast': render as fast as possible

        Returns:
            None
        """

        if self.render_mode is None:
            return
        
        if F110Env.renderer is None:
            # first call, initialize everything
            from f110_gym.envs.rendering import EnvRenderer
            F110Env.renderer = EnvRenderer(WINDOW_W, WINDOW_H, self.points_in_foreground)
            F110Env.renderer.update_map(self.map_name, self.map_ext)
            F110Env.renderer.update_raceline(self.raceline)
            
        F110Env.renderer.update_obs(self.render_obs)

        # To draw left and right raceline boundaries
        # x_left, y_left = self.raceline.to_cartesian(self.ego_s, -self.raceline.width_left_spline(self.ego_s))
        # F110Env.renderer.draw_point(x_left, y_left, size=10, color=(255, 0, 0), track=True)
        # x_right, y_right = self.raceline.to_cartesian(self.ego_s, self.raceline.width_right_spline(self.ego_s))
        # F110Env.renderer.draw_point(x_right, y_right, size=10, color=(0, 200, 255))

        # Render zones
        # colors = {0: (255, 128, 0), 1: (0, 0, 153), 2: (127, 0, 255)}
        # if not self.raceline_zones_rendered:
        #     for zone_id, (s_start, s_end, _) in self.raceline.zones.items():
        #         x_start, y_start = self.raceline.to_cartesian(s_start, 0.0)
        #         x_end, y_end = self.raceline.to_cartesian(s_end, 0.0)
        #         F110Env.renderer.draw_point(x_start, y_start, size=10, color=colors[zone_id], track=True)
        #         F110Env.renderer.draw_point(x_end, y_end, size=10, color=colors[zone_id], track=True)
        #     self.raceline_zones_rendered = True

        for render_callback in F110Env.render_callbacks:
            render_callback(F110Env.renderer)
        
        F110Env.renderer.dispatch_events()
        F110Env.renderer.on_draw()
        F110Env.renderer.flip()
        if self.render_mode == 'human':
            time.sleep(1.0 / self.metadata['render_fps'])
        elif self.render_mode == 'human_fast':
            pass


def show_map(image_path):
    """
    Show the map image using matplotlib
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    # Load the image
    image = Image.open(image_path)

    # Flip the image
    flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)

    # Plot both images side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Flipped image
    axes[1].imshow(flipped_image)
    axes[1].set_title("Flipped Image")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()