import mujoco
import numpy as np

def generate_xml(
                mass1 : float,
                mass2 : float,
                length1 : float,
                length2 : float,
                ground : tuple[float, str], #this needs to be figured out, need to calculate ground in env.reset seperately
                timestep : float,
                ) -> str:

    r = 0.02
    volume1 = length1 * np.pi * r * r
    density1 = mass1 / volume1
    
    volume2 = length2 * np.pi * r * r
    density2 = mass2 / volume2

    ground_depth, ground_quat = ground #ground_pos = "x y z", ground_quat = "a b c d"

    # Place the two fixed cameras at the pendulum's mid-height (it hangs from z=0 down
    # to z=-(length1+length2)) and back them off by a distance that frames the full
    # reach. cam0 looks along +Y (front), cam1 looks along +X (side).
    total_len = length1 + length2
    cam_z = -total_len / 2.0
    cam_d = 2.5 * total_len + 0.5
    xml = f"""
<mujoco>
    <option timestep="{timestep}" integrator="RK4">
        <flag energy="enable" contact="enable"/>
    </option>

    <default>
        <joint type="ball" damping="0.005"/>
        <geom type="cylinder" size="{r}" friction="0.1 0.005 0.0001"/>
    </default>

    <worldbody>
        <light pos="0 -.4 1"/>
        <camera name="cam0" pos="0 {-cam_d} {cam_z}" xyaxes="1 0 0 0 0 1"/>
        <camera name="cam1" pos="{-cam_d} 0 {cam_z}" xyaxes="0 -1 0 0 0 1"/>

            <geom name="floor" type="plane" pos="0 0 {ground_depth}" size="1.0 1.0 0.1" quat="{ground_quat}" friction="0.1 0.005 0.0001" rgba="0.9 0.9 0.9 1"/>

        <body name="link0" pos="0 0 0">
            <joint name="joint0"/>
            <geom fromto="0 0 0 0 0 {-length1}" density="{density1}" rgba="1 1 0 1"/>
            <body name="link1" pos="0 0 {-length1}">
                <joint name="joint1"/>
                <geom fromto="0 0 0 0 0 {-length2}" density="{density2}" rgba="1 0 0 1"/>
            </body>
        </body>
    </worldbody>
    <actuator>
        <motor name="motor0_x" joint="joint0" gear="1 0 0"/>
        <motor name="motor0_y" joint="joint0" gear="0 1 0"/>
        <motor name="motor0_z" joint="joint0" gear="0 0 1"/>
        <motor name="motor1_x" joint="joint1" gear="1 0 0"/>
        <motor name="motor1_y" joint="joint1" gear="0 1 0"/>
        <motor name="motor1_z" joint="joint1" gear="0 0 1"/>
    </actuator>


</mujoco>
    """

    return xml

