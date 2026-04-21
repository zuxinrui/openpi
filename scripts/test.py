import sys
import time

import numpy as np

import panda_py
from panda_py import controllers

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise RuntimeError(f'Usage: python {sys.argv[0]} <robot-hostname>')

    panda = panda_py.Panda(sys.argv[1])
    panda.move_to_start()
    
    x0 = panda.get_position()
    q0 = panda.get_orientation()
    joint_positions = panda.q.copy()
    joint_positions_deg = np.degrees(joint_positions)
    print(f"Initial position: {x0}, Initial orientation: {q0}, joint positions: {joint_positions_deg}")
    runtime = np.pi * 4.0

    # ctrl = controllers.CartesianImpedance(filter_coeff=1.0)
    ctrl = controllers.JointPosition(stiffness=np.array([200, 200, 200, 200, 100, 50, 20]), filter_coeff=1.0)
    panda.start_controller(ctrl)

#   with panda.create_context(frequency=1e3, max_runtime=runtime) as ctx:
#     while ctx.ok():
#       x_d = np.zeros(7)
#       x_d[1] += 0.1 * np.sin(ctrl.get_time())  # y oscillation
#       ctrl.set_control(x_d, q0)
    #   x_d = np.zeros(7)
    #   x_d[1] += 0.05 * np.sin(ctrl.get_time())  # joint-1 oscillation
    #   ctrl.set_control(position=x_d)

    time.sleep(1.0)
    x_d = joint_positions.copy()
    x_d[1] += 0.05  # move joint-2 by +0.05 rad (relative)
    ctrl.set_control(position=x_d)
    time.sleep(1.0)
    ctrl.set_control(position=joint_positions)
    time.sleep(1.0)

    # time.sleep(1.0)
    # x_d = x0.copy()
    # x_d[1] += 0.05
    # ctrl.set_control(position=x_d, orientation=q0)
    # time.sleep(1.0)
    # ctrl.set_control(position=x0, orientation=q0)
    # time.sleep(1.0)


