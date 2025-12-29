import gymnasium as gym
import mujoco
import numpy as np
import time
from typing import Optional
from physics import generate_xml


'''
TODO:

add comments for each function articulating usage

implement render

figure out file structure
'''
class DoublePendulum(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self,
                 mass1 : float,
                 mass2 : float,
                 length1 : float,
                 length2 : float,
                 timestep : float = 0.001,
                 render_mode : str | None=None,
                 render_width : int = 640,
                 render_height : int = 480,
                 camera_id : int | None=None,
                 ):
        #dont generate the model here
        
        self.mass1 = mass1
        self.mass2 = mass2
        self.length1 = length1
        self.length2 = length2

        self.dt = timestep

        self.state = np.array([-1, -1, -1, -1], dtype=np.float64) #in the form [theta1, theta2, omega1, omega2]
        self.torques = np.array([-1, -1], dtype=np.float64)

        self.max_tau = 10.0

        self.observation_space = gym.spaces.Box(low=np.array([0.0, 0.0, -40*np.pi, -40*np.pi]), high=np.array([2*np.pi, 2*np.pi, 40*np.pi, 40*np.pi]), shape=(4,), dtype=np.float64) #angl and omega values
        self.action_space = gym.spaces.Box(low=-self.max_tau, high=self.max_tau, shape=(2,), dtype=np.float64) #joint torques
        
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Invalid render_mode={render_mode}. Must be one of {self.metadata['render_modes']} or None.")
        self.render_mode = render_mode

        # render config
        self._render_width = int(render_width)
        self._render_height = int(render_height)
        self._camera_id = camera_id

        # lazy-created render objects
        self._viewer = None          # for "human"
        self._renderer = None        # for "rgb_array"
        self._last_render_time = None

        #update metadata based on timestep
        if self.dt > 0:
            self.metadata["render_fps"] = int(round(1.0 / self.dt))

        #model + data created in reset()
        self.model = None
        self.data = None

    def _get_obs(self):
        return np.array([self.state[0]% (2*np.pi), self.state[1] % (2*np.pi), self.state[2], self.state[3]], dtype=np.float64)

    def _get_info(self) -> dict:
        'this is the state vector we input into the model'
        #i think we want to be collecting mujuco trajectories here
        #so collect robot state, [[thetas, omegas], taus, contacts] where taus is the torque on each joint,
        #and contacts is in the form [robot contact point, world contact point, contact normal, contact distance ]
        return {}

    def _sync_state(self):
        self.state[:] = np.array([self.data.qpos[0], self.data.qpos[1], self.data.qvel[0], self.data.qvel[1]], dtype=np.float64)

    def reset(self, seed : Optional[int] = None, options : Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: specific ground, theta, omega configurations for testing. 
                     dict is in the form {"ground" : (depth, quat), "thetas" : (theta1, theta2), "omegas" : (omega1, omega2)}

        Returns:
            tuple: (observation, info) for the initial state
        """
        super().reset(seed=seed)

        #generate thetas, omegas, taus

        thetas = self.np_random.uniform(0.0, 2*np.pi, size=2)
        omegas = self.np_random.uniform(-40*np.pi, 40*np.pi, size=2)
        self.state[:2] = thetas
        self.state[2:] = omegas

        self.torques = self.np_random.uniform(-self.max_tau, self.max_tau, size=2)
        tau1, tau2 = self.torques

        #generate random ground (depth and quaternion) in progress
        ground_depth = -2
        ground_quat = "0 0 0 1"
        ground = (ground_depth, ground_quat)

        xml = generate_xml(self.mass1, self.mass2, self.length1, self.length2, ground, self.dt)
        self.model = mujoco.MjModel.from_xml_string(xml) 
        self.data = mujoco.MjData(self.model)

        #adjust thetas omegas and taus
        theta1, theta2, omega1, omega2 = self.state
        self.data.qpos[:] = np.array([theta1, theta2])
        self.data.qvel[:] = np.array([omega1, omega2])
        self.data.ctrl[:] = np.array([tau1, tau2])

        mujoco.mj_forward(self.model, self.data)
        self._sync_state()

        self._destroy_rendering()
        
        observation = self._get_obs()
        info = self._get_info()
        

        return observation, info
        


    def step(self, action : tuple[float, float]):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        tau1, tau2 = action
        tau1 = np.clip(tau1, -self.max_tau, self.max_tau)
        tau2 = np.clip(tau2, -self.max_tau, self.max_tau)

        self.data.ctrl[:] = (tau1, tau2)
        mujoco.mj_step(self.model, self.data)
        
        self._sync_state()

        terminated = (not np.isfinite(self.state).all())
        observation = self._get_obs()
        info = self._get_info()
        reward = 0.0
        truncated = False #truncation will be done through a timestep wrapper or rollout function/object
        
        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info


    def render(self):
        if self.render_mode is None:
            return None
        if self.model is None or self.data is None:
            raise RuntimeError("Call reset() before render().")

        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer as mj_viewer
                self._viewer = mj_viewer.launch_passive(self.model, self.data)

                # for dt-based pacing
                self._last_render_time = time.perf_counter()

            self._viewer.sync()

           #real time pacing
            if self.dt > 0:
                now = time.perf_counter()
                if self._last_render_time is not None:
                    target = self._last_render_time + self.dt
                    sleep_for = target - now
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                self._last_render_time = time.perf_counter()

            return None

        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=self._render_height, width=self._render_width)

            # update scene from current physics state
            if self._camera_id is None:
                self._renderer.update_scene(self.data)
            else:
                self._renderer.update_scene(self.data, camera=self._camera_id)

            img = self._renderer.render()  # (H, W, 3) uint8
            return img.copy()

        raise ValueError(f"Unknown render_mode={self.render_mode}")

    def _destroy_rendering(self):
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None

        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None

        self._last_render_time = None

    def close(self):
        self._destroy_rendering()