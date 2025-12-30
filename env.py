import gymnasium as gym
import mujoco
import numpy as np
import time
from typing import Optional
from physics import generate_xml


'''
TODO:

add comments for each function articulating usage/improve typehinting

implement _get_info to collect contacts and torques as well as changing state to robtot centric representation

figure out file structure

figure out randomized ground depth/angles

smaller: 

set a fixed fps (so that we dont run at 10000fps for small timesteps)
try to see if its possible to keep the viewer open and just update states between episodes
frameskips
'''
class DoublePendulum(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self,
                 mass1 : float,
                 mass2 : float,
                 length1 : float,
                 length2 : float,
                 timestep : float = 0.002,
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

        self.max_tau = 4.0

        self.observation_space = gym.spaces.Box(low=np.array([0.0, 0.0, -10*np.pi, -10*np.pi]), high=np.array([2*np.pi, 2*np.pi, 10*np.pi, 10*np.pi]), shape=(4,), dtype=np.float64) #angl and omega values
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
        self.body_ids = None

        #CONSTANTS
        self.substeps = 4 #substeps for mujoco per env.step
        self.xi = 0.004 #contact distance threshold as described in the NeRD paper
        self.max_contacts = 16


    def _get_obs(self) -> np.ndarray:
        return np.array([self.state[0]% (2*np.pi), self.state[1] % (2*np.pi), self.state[2], self.state[3]], dtype=np.float64)

    def _get_info(self) -> dict:
        ''' this is where we calculate the inputs needed for the NeRD function. 
        We need the robot-centric state representation (we don't include 6d velocity/spatial twist because the base is fixed), 
        contacts, and gravity, as well as joint torques. These are returned as a flattened array under "state".

        For calculation of our target, delta_s_t, we also need to return the rotation matrix and world frame pendulum position, 
        which are returned under "R" and "world_pos" respectively.
        '''
        body = self.body_ids["link0"]

        #STATE CALCULATION AND TRANSFORMATION
        worldf_pos = self.data.xpos[body].copy()
        worldf_R = self.data.xmat[body].reshape(3,3).copy()
        thetas = self.state[:2]
        omegas = self.state[2:4]
        
        basef_pos = np.zeros(shape=(3,), dtype=np.float64)
        basef_R = np.identity(3)

        s = np.concatenate([basef_pos, basef_R.reshape(-1), thetas, omegas])

        #TORQUES
        tau = self.torques

        #CONTACT COLLECTION
        K = self.max_contacts
        contacts = []

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            distance = contact.dist
            if distance > self.xi:
                continue
            
            normal = contact.frame[:3]

            if contact.geom[0] == self.model.geom("floor").id:
                p0 = contact.pos - 0.5*distance*normal
                p1 = contact.pos + 0.5*distance*normal
            else:
                p0 = contact.pos + 0.5*distance*normal
                p1 = contact.pos - 0.5*distance*normal

            basef_p0 = worldf_R.T @ (p0 - worldf_pos)
            basef_p1 = worldf_R.T @ (p1 - worldf_pos)
            basef_normal = worldf_R.T @ normal
            
            contacts.append((distance, basef_p0, basef_p1, basef_normal))

        contacts.sort(key=lambda x:x[0])

        basef_contacts = np.zeros((K, 10), dtype=np.float64)
        for j, (dist, p0, p1, n) in enumerate(contacts[:K]):
            basef_contacts[j, 0:3] = p0
            basef_contacts[j, 3:6] = p1
            basef_contacts[j, 6:9] = n
            basef_contacts[j, 9] = dist

        #ROBOT-CENTRIC GRAVITY
        worldf_gravity = self.model.opt.gravity
        basef_g = worldf_R.T @ worldf_gravity

        robot_state = np.concatenate([s, tau, basef_contacts.reshape(-1), basef_g])
        
        return {
            "state" : robot_state,
            "R" : worldf_R,
            "world_pos" : worldf_pos
        }


    def _sync_state(self) -> None:
        self.state[:2] = self.data.qpos[:]
        self.state[2:4] = self.data.qvel[:]

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
        omegas = self.np_random.uniform(-10*np.pi, 10*np.pi, size=2)
        self.state[:2] = thetas
        self.state[2:] = omegas

        self.torques = self.np_random.uniform(-self.max_tau, self.max_tau, size=2)
        tau1, tau2 = self.torques

        #generate random ground (depth and quaternion) in progress
        ground_depth = -2
        ground_quat = "0 0 0 1"
        ground = (ground_depth, ground_quat)

        model_timestep = self.dt / self.substeps
        xml = generate_xml(self.mass1, self.mass2, self.length1, self.length2, ground, model_timestep)
        self.model = mujoco.MjModel.from_xml_string(xml) 
        self.data = mujoco.MjData(self.model)

        self.body_ids = {"link0" : self.model.body("link0").id,
                       "link1" : self.model.body("link1").id}

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
        


    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        tau1, tau2 = action
        tau1 = np.clip(tau1, -self.max_tau, self.max_tau)
        tau2 = np.clip(tau2, -self.max_tau, self.max_tau)

        self.torques[:] = (tau1, tau2)
        self.data.ctrl[:] = (tau1, tau2)

        for _ in range(self.substeps):
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

    def _destroy_rendering(self) -> None:
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

    def close(self) -> None:
        self._destroy_rendering()