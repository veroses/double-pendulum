import gymnasium as gym
import mujoco
import numpy as np
import time
from typing import Optional
from physics import generate_xml


# TODO: figure out randomized ground depth/angles
# TODO: try to keep the viewer open and just update states between episodes (avoid re-creating viewer on reset)
# TODO: frameskips


class DoublePendulum(gym.Env):
    """Gymnasium environment for a 2-link planar double pendulum simulated in MuJoCo.

    The pendulum hangs from a fixed base at the world origin. Both links are cylindrical
    and actuated at their hinge joints (torque-controlled). The goal of this environment
    is to generate rollout trajectories for training neural networks (e.g. NeRD).

    Coordinate convention:
        - The hinge axes are aligned with the world Y axis.
        - Link geometry runs from the joint origin downward along the -Z axis.
        - Angles (qpos) are measured from the downward-hanging rest position.

    Observation space (shape 4):
        [θ₀ (rad, wrapped to [0, 2π]),
         θ₁ (rad, wrapped to [0, 2π]),
         ω₀ (rad/s),
         ω₁ (rad/s)]

    Action space (shape 2):
        [τ₀ (N·m), τ₁ (N·m)] — joint torques, clipped to ±max_tau.

    The richer per-step state needed for NeRD training is returned in the `info` dict
    from both reset() and step() — see _get_info() for its structure.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self,
                 mass0: float,
                 mass1: float,
                 length0: float,
                 length1: float,
                 timestep: float = 0.002,
                 slide: bool = False,
                 render_mode: str | None = None,
                 render_width: int = 640,
                 render_height: int = 480,
                 camera_id: int | None = None,
                 ):
        """
        Args:
            mass0: Mass of link 0 (the upper link attached to the fixed base), in kg.
            mass1: Mass of link 1 (the lower/free link), in kg.
            length0: Length of link 0, in metres.
            length1: Length of link 1, in metres.
            timestep: Duration of one environment step in seconds. The underlying MuJoCo
                      physics runs at timestep/substeps to maintain accuracy. Defaults to 0.002s.
            render_mode: "human" opens an interactive viewer window; "rgb_array" returns
                         frames from render(). None disables rendering.
            render_width: Width in pixels for rgb_array rendering.
            render_height: Height in pixels for rgb_array rendering.
            camera_id: MuJoCo camera ID to use for rgb_array rendering. None uses the
                       default free camera.
            slide: If True, the ground is placed within contact range of the links
                   (randomised depth and tilt each episode) so the pendulum can
                   collide and slide along it. If False, the ground is parked well
                   below the pendulum and plays no role in the dynamics. Defaults
                   to False.
        """
        self.mass0 = mass0
        self.mass1 = mass1
        self.length0 = length0
        self.length1 = length1

        self.dt = timestep

        # Internal physics state — written by _sync_state() after each mj_step.
        # Initialised to sentinel values until reset() is called.
        self.state = np.array([-1, -1, -1, -1], dtype=np.float64)   # [θ₀, θ₁, ω₀, ω₁]
        self.torques = np.array([-1, -1], dtype=np.float64)          # [τ₀, τ₁]

        self.slide = slide
        self.max_tau = 4.0  # N·m — symmetric torque limit applied to both joints

        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, -10*np.pi, -10*np.pi]),
            high=np.array([2*np.pi, 2*np.pi, 10*np.pi, 10*np.pi]),
            shape=(4,),
            dtype=np.float64,
        )
        self.action_space = gym.spaces.Box(
            low=-self.max_tau,
            high=self.max_tau,
            shape=(2,),
            dtype=np.float64,
        )

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Invalid render_mode={render_mode}. Must be one of {self.metadata['render_modes']} or None.")
        self.render_mode = render_mode

        self._render_width = int(render_width)
        self._render_height = int(render_height)
        self._camera_id = camera_id

        # Render objects are created lazily on first render() call and destroyed on reset().
        self._viewer = None           # mujoco.viewer handle — used for "human" mode
        self._renderer = None         # mujoco.Renderer — used for "rgb_array" mode
        self._last_render_time = None # wall-clock time of the last render, used for real-time pacing

        # Derive render_fps from the env timestep so wrappers see a consistent value.
        if self.dt > 0:
            self.metadata["render_fps"] = int(round(1.0 / self.dt))

        # MuJoCo model and data are built in reset() so that ground geometry can be
        # randomised per episode without re-instantiating the environment.
        self.model = None
        self.data = None
        self.body_ids = None  # dict mapping body name → MuJoCo body id, populated in reset()

        # --- Constants ---
        # Each env.step() calls mj_step this many times at a finer timestep (dt/substeps).
        # More substeps → better numerical accuracy at the cost of compute.
        self.substeps = 4

        # Contact distance threshold (metres) from the NeRD paper.
        # Only contacts with dist ≤ xi are included in the info state vector.
        self.xi = 0.004

        # Number of bins each link is divided into for contact localisation.
        # Higher K → finer spatial resolution of where contact occurs along the link.
        self.segments_per_link = 8


    def _get_obs(self) -> np.ndarray:
        """Return the 4-dimensional observation for the current state.

        Angles are wrapped to [0, 2π] so the observation space bounds are always
        respected. Angular velocities are returned unwrapped.

        Returns:
            np.ndarray of shape (4,): [θ₀, θ₁, ω₀, ω₁]
        """
        return np.array(
            [self.state[0] % (2*np.pi),
             self.state[1] % (2*np.pi),
             self.state[2],
             self.state[3]],
            dtype=np.float64,
        )

    def _get_info(self) -> dict:
        """Compute the richer robot-centric state vector used for NeRD training.

        This bundles everything the NeRD network needs as inputs or targets into a
        single dict. The heavy lifting is:

          1. Robot-centric state  s  (position, orientation, joint angles/vels)
          2. Applied torques  τ
          3. Binned contact descriptors along both links
          4. Gravity vector expressed in the link0 body frame

        Contact binning:
            Each link is divided into K = segments_per_link segments along its local
            -Z axis. For every active MuJoCo contact (dist ≤ xi) involving the floor
            and one of the two links, the contact point is projected into that link's
            body frame, mapped to a bin index, and stored. If multiple contacts land
            in the same bin, the closest one (smallest dist) is kept.

            Each bin stores 10 floats:
                [p0 (3,), p1 (3,), normal (3,), dist (1,)]
            where p0 is the point on the link surface, p1 the point on the floor, and
            normal points from floor toward the link — all in link0's body frame.

            The filled mask (2*K booleans cast to float32) indicates which bins have
            a valid contact.

        Returns:
            dict with keys:
                "state"     : np.ndarray (float64) — flattened vector:
                                  [basef_pos (3,), basef_R (9,), thetas (2,),
                                   omegas (2,), tau (2,), contacts (2*K*10,),
                                   filled (2*K,), basef_g (3,)]
                                  Total length = 3 + 9 + 2 + 2 + 2 + 2*K*10 + 2*K + 3
                                               = 181 for K=8.
                "R"         : np.ndarray (3,3) — world-frame rotation matrix of link0.
                              Used externally to compute the NeRD target Δs_t.
                "world_pos" : np.ndarray (3,) — world-frame position of link0's origin.
                              Also used to compute Δs_t.
        """
        body = self.body_ids["link0"]
        link1 = self.body_ids["link1"]

        # World-frame pose of both links (rotation matrices from xmat, reshaped to 3×3)
        worldf_pos  = self.data.xpos[body].copy()
        worldf_R    = self.data.xmat[body].reshape(3, 3).copy()
        worldf_pos1 = self.data.xpos[link1].copy()
        worldf_R1   = self.data.xmat[link1].reshape(3, 3).copy()

        thetas = self.state[:2]
        omegas = self.state[2:4]

        # Because the base is fixed to the world, the base-frame position is the origin
        # and the base-frame rotation is identity. We include them explicitly so the
        # state vector has a consistent structure if the base ever becomes free.
        basef_pos = np.zeros(3, dtype=np.float64)
        basef_R   = np.eye(3, dtype=np.float64)

        s = np.concatenate([basef_pos, basef_R.reshape(-1), thetas, omegas])

        tau = np.asarray(self.torques, dtype=np.float64).reshape(2,)

        # --- Contact collection ---
        K = self.segments_per_link
        contacts = np.zeros((2*K, 10), dtype=np.float64)
        filled   = np.zeros((2*K,), dtype=np.bool_)

        floor = self.model.geom("floor").id

        for i in range(self.data.ncon):
            contact  = self.data.contact[i]
            distance = contact.dist
            if distance > self.xi:
                continue

            # MuJoCo contact normal points from geom1 toward geom0.
            normal = contact.frame[:3]
            g0, g1 = contact.geom
            b0 = self.model.geom_bodyid[g0]
            b1 = self.model.geom_bodyid[g1]

            # Reconstruct the surface contact points on each geometry.
            # p0 is the point on the non-floor geom (the link), p1 on the floor.
            if g0 == floor:
                # floor is geom0 → normal points from floor toward link
                p0 = contact.pos - 0.5 * distance * normal  # point on floor surface
                p1 = contact.pos + 0.5 * distance * normal  # point on link surface

                if b1 == body:    # contact is with link0
                    p_body  = worldf_R.T @ (p1 - worldf_pos)
                    R_contact, pos_contact = worldf_R, worldf_pos
                    z = -p_body[2]
                    bin_idx = int(np.clip((z * K) / self.length0, 0.0, K - 1))
                elif b1 == link1: # contact is with link1
                    p_body  = worldf_R1.T @ (p1 - worldf_pos1)
                    R_contact, pos_contact = worldf_R1, worldf_pos1
                    z = -p_body[2]
                    bin_idx = int(np.clip((z * K) / self.length1, 0.0, K - 1) + K)
                else:
                    continue

            elif g1 == floor:
                # floor is geom1 → normal points from link toward floor, so flip
                p0 = contact.pos + 0.5 * distance * normal  # point on link surface
                p1 = contact.pos - 0.5 * distance * normal  # point on floor surface

                if b0 == body:    # contact is with link0
                    p_body  = worldf_R.T @ (p0 - worldf_pos)
                    R_contact, pos_contact = worldf_R, worldf_pos
                    z = -p_body[2]
                    bin_idx = int(np.clip((z * K) / self.length0, 0.0, K - 1))
                elif b0 == link1: # contact is with link1
                    p_body  = worldf_R1.T @ (p0 - worldf_pos1)
                    R_contact, pos_contact = worldf_R1, worldf_pos1
                    z = -p_body[2]
                    bin_idx = int(np.clip((z * K) / self.length1, 0.0, K - 1) + K)
                else:
                    continue

            else:
                continue  # neither geom is the floor — skip

            # Express the contact geometry in the relevant link's body frame.
            basef_p0     = R_contact.T @ (p0 - pos_contact)
            basef_p1     = R_contact.T @ (p1 - pos_contact)
            basef_normal = R_contact.T @ normal

            # Keep the closest contact per bin (smallest penetration depth).
            bin = contacts[bin_idx]
            if (not filled[bin_idx]) or distance < bin[9]:
                bin[0:3] = basef_p0
                bin[3:6] = basef_p1
                bin[6:9] = basef_normal
                bin[9]   = distance
                filled[bin_idx] = True

        # Gravity in link0's body frame — used as a network input so the model can
        # implicitly reason about the direction of gravitational forces.
        worldf_gravity = self.model.opt.gravity
        basef_g = worldf_R.T @ worldf_gravity

        contacts_flat = np.concatenate([contacts.reshape(-1), filled.astype(np.float32)])
        robot_state   = np.concatenate([s, tau, contacts_flat, basef_g])

        return {
            "state":     robot_state,
            "R":         worldf_R,
            "world_pos": worldf_pos,
        }


    def _sync_state(self) -> None:
        """Copy the current MuJoCo qpos/qvel into self.state.

        Called after every mj_step / mj_forward to keep self.state in sync with
        the MuJoCo data buffer. self.state is the authoritative Python-side cache
        used by _get_obs() and _get_info().
        """
        self.state[:2] = self.data.qpos[:]
        self.state[2:4] = self.data.qvel[:]


    def _sample_ground(self, slide=True) -> tuple[float, np.ndarray]:
        """Sample a random ground plane depth and orientation for one episode.

        Depth range:
            The ground z-position is sampled uniformly between:
              - deepest : 1 m below the tip of link1  →  z = -(length0 + length1 + 1)
              - shallowest : halfway up link0          →  z = -length0 / 2

        Tilt range:
            The ground normal can tilt up to 30° away from vertical (+Z) in any
            horizontal direction. This corresponds to surface slopes of up to 30°,
            or equivalently surface normals that span 60°–120° from horizontal
            (90° being a flat floor).

            Sampling:
              - tilt magnitude θ ~ Uniform(0°, 30°)
              - azimuth φ ~ Uniform(0°, 360°)

            The tilt is realised as a rotation of the floor's default orientation
            (normal = +Z) by angle θ around the horizontal axis at azimuth φ:
              rotation axis = (-sin φ, cos φ, 0)
              quaternion    = (cos θ/2,  sin θ/2 · axis)   in MuJoCo w-x-y-z order

        Returns:
            (ground_depth, ground_quat) where ground_depth is a float (metres) and
            ground_quat is a np.ndarray of shape (4,) in (w, x, y, z) order.
        """
        eps = 1e-3

        if slide:
            z_min = -(self.length0 + self.length1 - eps)
            z_max = -self.length0 / 2.0
            ground_depth = float(self.np_random.uniform(z_min, z_max))

            tilt = self.np_random.uniform(0.0, np.radians(30.0))
            phi  = self.np_random.uniform(0.0, 2 * np.pi)
            ax, ay = -np.sin(phi), np.cos(phi)
            w = np.cos(tilt / 2)
            s = np.sin(tilt / 2)
            ground_quat = np.array([w, s*ax, s*ay, 0.0])

        else:
            ground_depth = -(self.length0 + self.length1 + 0.3)
            ground_quat = np.array([1, 0, 0, 0])

        return ground_depth, ground_quat


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode with randomised initial conditions.

        On the first call the MuJoCo model is built from XML. On every subsequent
        call the same model/data objects are reused — only the ground geom's position
        and quaternion are patched in-place, and data is reset with mj_resetData.
        This means the viewer window (if open) stays alive across episodes without
        needing to be closed or reloaded.

        Args:
            seed: RNG seed for reproducible episodes. Passed to the Gymnasium base
                  class which initialises self.np_random.
            options: Reserved for overriding specific initial conditions in testing.
                     Intended format (not yet implemented):
                     {"ground": (depth, quat), "thetas": (θ₀, θ₁), "omegas": (ω₀, ω₁)}

        Returns:
            observation (np.ndarray): Initial observation of shape (4,).
            info (dict): Initial NeRD state dict from _get_info().
        """
        super().reset(seed=seed)

        thetas = self.np_random.uniform(0.0, 2*np.pi, size=2)
        omegas = self.np_random.uniform(-10*np.pi, 10*np.pi, size=2)
        self.state[:2] = thetas
        self.state[2:]  = omegas

        self.torques = self.np_random.uniform(-self.max_tau, self.max_tau, size=2)
        tau1, tau2 = self.torques

        ground_depth, ground_quat = self._sample_ground(slide=self.slide)

        if self.model is None:
            # First reset: build model from XML and cache body IDs.
            ground_quat_str = " ".join(str(v) for v in ground_quat)
            model_timestep  = self.dt / self.substeps
            xml = generate_xml(self.mass0, self.mass1, self.length0, self.length1,
                               (ground_depth, ground_quat_str), model_timestep)
            self.model = mujoco.MjModel.from_xml_string(xml)
            self.data  = mujoco.MjData(self.model)
            self.body_ids = {
                "link0": self.model.body("link0").id,
                "link1": self.model.body("link1").id,
            }
        else:
            # Subsequent resets: patch ground geometry in-place so the viewer
            # window doesn't need to close/reopen.
            floor_id = self.model.geom("floor").id
            self.model.geom_pos[floor_id]  = [0.0, 0.0, ground_depth]
            self.model.geom_quat[floor_id] = ground_quat
            # Reset all of data back to defaults (zeros qpos/qvel, clears contacts etc.)
            mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:] = thetas
        self.data.qvel[:] = omegas
        self.data.ctrl[:] = [tau1, tau2]

        # mj_forward propagates the initial state through the kinematics/dynamics
        # without advancing time, so xpos/xmat/contacts are all valid after this call.
        mujoco.mj_forward(self.model, self.data)
        self._sync_state()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info


    def step(self, action):
        """Advance the simulation by one environment step.

        Clips the action to the valid torque range, applies it to both joints, runs
        `substeps` MuJoCo integration steps, then returns the new state.

        Args:
            action: Array-like of shape (2,) containing [τ₀, τ₁] in N·m.

        Returns:
            observation (np.ndarray): Shape (4,) — see _get_obs().
            reward (float): Always 0.0 — reward shaping is left to wrappers.
            terminated (bool): True if any state value becomes non-finite (simulation
                               blow-up). The episode should be reset immediately.
            truncated (bool): Always False — episode length is managed externally
                              via a TimeLimit wrapper or the rollout function.
            info (dict): NeRD state dict from _get_info().
        """
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        tau1 = np.clip(action[0], -self.max_tau, self.max_tau)
        tau2 = np.clip(action[1], -self.max_tau, self.max_tau)

        self.torques[:] = (tau1, tau2)
        self.data.ctrl[:] = (tau1, tau2)

        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

        self._sync_state()

        terminated  = not np.isfinite(self.state).all()
        observation = self._get_obs()
        info        = self._get_info()
        reward      = 0.0
        truncated   = False

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info


    def render(self):
        """Render the current simulation state.

        Behaviour depends on render_mode set at construction:
            "human"     — syncs the passive viewer window. Real-time pacing is
                          applied so the simulation plays back at wall-clock speed.
                          Returns None.
            "rgb_array" — renders an off-screen frame and returns it as a
                          (H, W, 3) uint8 numpy array.
            None        — no-op, returns None.

        Raises:
            RuntimeError: If called before reset().
            ValueError: If render_mode is unrecognised (should not happen if __init__
                        validation is working correctly).
        """
        if self.render_mode is None:
            return None
        if self.model is None or self.data is None:
            raise RuntimeError("Call reset() before render().")

        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer as mj_viewer
                self._viewer = mj_viewer.launch_passive(self.model, self.data)
                self._last_render_time = time.perf_counter()

            self._viewer.sync()

            # Throttle to real time by sleeping for the remainder of the step interval.
            if self.dt > 0:
                now = time.perf_counter()
                if self._last_render_time is not None:
                    sleep_for = (self._last_render_time + self.dt) - now
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                self._last_render_time = time.perf_counter()

            return None

        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=self._render_height, width=self._render_width)

            if self._camera_id is None:
                self._renderer.update_scene(self.data)
            else:
                self._renderer.update_scene(self.data, camera=self._camera_id)

            return self._renderer.render().copy()  # copy so the caller owns the buffer

        raise ValueError(f"Unknown render_mode={self.render_mode}")


    def _destroy_rendering(self) -> None:
        """Close and release all rendering objects entirely. Called by close()."""
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
        """Clean up all MuJoCo rendering resources. Call when done with the environment."""
        self._destroy_rendering()
