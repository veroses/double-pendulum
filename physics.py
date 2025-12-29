import mujoco
import numpy as np

def generate_xml(
                mass1 : float,
                mass2 : float,
                length1 : float,
                length2 : float,
                ground : tuple[float, str], #this needs to be figured out, need to calculate ground in env.reset seperately
                timestep : float,
                ):

    r = 0.02
    volume1 = length1 * np.pi * r * r
    density1 = mass1 / volume1
    
    volume2 = length2 * np.pi * r * r
    density2 = mass2 / volume2

    ground_depth, ground_quat = ground #ground_pos = "x y z", ground_quat = "a b c d"
    xml = f"""
<mujoco>
    <option timestep="{timestep}" integrator="RK4">
        <flag energy="enable" contact="enable"/>
    </option>

    <default>
        <joint type="hinge" axis="0 1 0" damping="0.05"/>
        <geom type="cylinder" size="{r}" friction="1 0.005 0.0001"/>
    </default>

    <worldbody>
        <light pos="0 -.4 1"/>
        <camera name="fixed" pos="0 -2 1" xyaxes="1 0 0 0 0 1"/>

            <geom name="floor" type="plane" pos="0 0 {ground_depth}" size="5 5 0.1" quat="{ground_quat}" friction="1 0.005 0.0001" rgba="0.9 0.9 0.9 1"/>

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
        <motor name="motor0" joint="joint0" gear="1"/>
    </actuator>
    <actuator>
        <motor name="motor1" joint="joint1" gear="1"/>
    </actuator>


</mujoco>
    """

    return xml

