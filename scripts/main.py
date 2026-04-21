# ruff: noqa

import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import sys
import time
import cv2
# from moviepy.editor import ImageSequenceClip
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import pandas as pd
from PIL import Image
# from droid.robot_env import RobotEnv
import tqdm
import tyro
import panda_py
from panda_py import controllers

import pyrealsense2 as rs

faulthandler.enable()

# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15


@dataclasses.dataclass
class Args:
    # Hardware parameters
    # left_camera_id: str = "<your_camera_id>"  # e.g., "24259877"
    # right_camera_id: str = "<your_camera_id>"  # e.g., "24514023"
    # wrist_camera_id: str = "<your_camera_id>"  # e.g., "13062452"

    # Policy parameters
    # external_camera: str | None = (
    #     None  # which external camera should be fed to the policy, choose from ["left", "right"]
    # )

    # Rollout parameters
    max_timesteps: int = 600
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 8

    # Remote server parameters
    remote_host: str = "0.0.0.0"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = (
        8000  # point this to the port of the policy server, default server port for openpi servers is 8000
    )


# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def main(args: Args):
    # Make sure external camera is specified by user -- we only use one external camera for the policy
    # assert (
    #     args.external_camera is not None and args.external_camera in ["left", "right"]
    # ), f"Please specify an external camera to use for the policy, choose from ['left', 'right'], but got {args.external_camera}"

    # Initialize the Panda environment. Using joint velocity action space and gripper position action space is very important.
    # env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    panda = panda_py.Panda("172.16.0.2")
  # print(panda)
    panda.move_to_start()

    zero_joint_positions = panda.q.copy()

    ctrl = controllers.JointPosition(
        stiffness=np.array([200, 200, 200, 200, 100, 50, 20]),
        filter_coeff=1.0,
        )
    panda.start_controller(ctrl)

    print("Created the franka-simple env!")

    # Connect to the policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)

    df = pd.DataFrame(columns=["success", "duration", "video_filename"])

    # ---- 0) Discover devices and pick two ----
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) < 2:
        print(f"Need at least 2 RealSense devices; found {len(devices)}.")
        sys.exit(1)

    serials = [d.get_info(rs.camera_info.serial_number) for d in devices][:2]
    print("Using devices:", serials)

    # ---- 1) Create pipelines & configs, pin to serials ----
    pipelines = []
    aligners = []
    for sn in serials:
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(sn)
        # Choose modest resolution/FPS to avoid USB bandwidth issues
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipe.start(cfg)
        pipelines.append(pipe)
        aligners.append(rs.align(rs.stream.color))  # align depth to color

    while True:
        instruction = input("Enter instruction: ")

        # Rollout parameters
        actions_from_chunk_completed = 0
        pred_action_chunk = None

        # Prepare to save video of rollout
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        video = []
        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")
        for t_step in bar:
            start_time = time.time()
            try:
                joint_positions = panda.q.copy()
                # # Get the current observation
                # curr_obs = _extract_observation(
                #     args,
                #     env.get_observation(),
                #     # Save the first observation to disk
                #     save_to_disk=t_step == 0,
                # )

                # video.append(curr_obs[f"{args.external_camera}_image"])

                # Send websocket request to policy server if it's time to predict a new chunk
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                    actions_from_chunk_completed = 0

                    # We resize images on the robot laptop to minimize the amount of data sent to the policy server
                    # and improve latency.
                    # img1 = np.zeros((224, 224, 3), dtype=np.uint8)
                    # img2 = np.zeros((224, 224, 3), dtype=np.uint8)
                    frames_list = []
                    for pipe, align in zip(pipelines, aligners):
                        frames = pipe.wait_for_frames()
                        frames = align.process(frames)  # depth aligned to color
                        color = frames.get_color_frame()
                        depth = frames.get_depth_frame()
                        if not color or not depth:
                            continue

                        color_img = np.asanyarray(color.get_data())
                        depth_img = np.asanyarray(depth.get_data())

                        depth_colormap = cv2.applyColorMap(
                            cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET
                        )
                        # Stack color + depth for this camera
                        stacked = np.hstack((color_img, depth_colormap))
                        frames_list.append(stacked)

                    if not frames_list:
                        continue

                    # ---- 3) Concatenate both cameras horizontally ----
                    # Resize if dimensions differ (they shouldn't if you used same config)
                    h_min = min(img.shape[0] for img in frames_list)
                    frames_list = [cv2.resize(img, (img.shape[1]*h_min//img.shape[0], h_min)) for img in frames_list]
                    combined = np.hstack(frames_list)

                    # print(f"Displaying combined frame shape:", combined.shape)

                    # print(f"pixel value: {combined[0,0,:]}")

                    # cv2.imshow("RealSense x2 (left: cam0, right: cam1)", combined)
                    # if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                    #     break

                    img1 = combined[:, :640, :]
                    img1 = image_tools.resize_with_pad(
                            img1, 224, 224
                        )
                    img2 = combined[:, 1280:1920, :]
                    img2 = image_tools.resize_with_pad(
                            img2, 224, 224
                        )

                    cv2.imshow("RealSense x2 (left: cam0, right: cam1)", np.hstack((img1, img2)))
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                        break

                    joint_pos = joint_positions.astype(np.float32)
                    request_data = {
                        "observation/exterior_image_1_left": img1,
                        "observation/wrist_image_left": img2,
                        "observation/joint_position": joint_pos,
                        "observation/gripper_position": np.array([0.0], dtype=np.float32),
                        "prompt": instruction,
                    }

                    # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                    # Ctrl+C will be handled after the server call is complete
                    with prevent_keyboard_interrupt():
                        # this returns action chunk [10, 8] of 10 joint velocity actions (7) + gripper position (1)
                        pred_action_chunk = policy_client.infer(request_data)["actions"]
                    assert pred_action_chunk.shape == (10, 8)

                # Select current action to execute from chunk
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                # Binarize gripper action
                if action[-1].item() > 0.5:
                    # action[-1] = 1.0
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    # action[-1] = 0.0
                    action = np.concatenate([action[:-1], np.zeros((1,))])

                # clip all dimensions of action to [-1, 1]
                action = np.clip(action, -3, 3)
                # action = np.clip(action, -0.2, 0.2)
                action = action[:-1]
                print(f"Executing actions: {action}")

                # env.step(action)
                # panda.move_to_joint_position(action, speed_factor=0.05)
                # action_definite = zero_joint_positions + action

                # print(f"Moving to joint positions: {action}")
                ctrl.set_control(position=action)

                # Sleep to match DROID data collection frequency
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
            except KeyboardInterrupt:
                break

        # video = np.stack(video)
        # save_filename = "video_" + timestamp
        # ImageSequenceClip(list(video), fps=10).write_videofile(save_filename + ".mp4", codec="libx264")

        success: str | float | None = None
        while not isinstance(success, float):
            success = input(
                "Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100 based on the evaluation spec"
            )
            if success == "y":
                success = 1.0
            elif success == "n":
                success = 0.0

            success = float(success) / 100
            if not (0 <= success <= 1):
                print(f"Success must be a number in [0, 100] but got: {success * 100}")

        # df = df.append(
        #     {
        #         "success": success,
        #         "duration": t_step,
        #         "video_filename": save_filename,
        #     },
        #     ignore_index=True,
        # )

        if input("Do one more eval? (enter y or n) ").lower() != "y":
            break
        # env.reset()

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    csv_filename = os.path.join("results", f"eval_{timestamp}.csv")
    df.to_csv(csv_filename)
    print(f"Results saved to {csv_filename}")


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None
    for key in image_observations:
        # Note the "left" below refers to the left camera in the stereo pair.
        # The model is only trained on left stereo cams, so we only feed those.
        if args.left_camera_id in key and "left" in key:
            left_image = image_observations[key]
        elif args.right_camera_id in key and "left" in key:
            right_image = image_observations[key]
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key]

    # Drop the alpha dimension
    left_image = left_image[..., :3]
    right_image = right_image[..., :3]
    wrist_image = wrist_image[..., :3]

    # Convert to RGB
    left_image = left_image[..., ::-1]
    right_image = right_image[..., ::-1]
    wrist_image = wrist_image[..., ::-1]

    # In addition to image observations, also capture the proprioceptive state
    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    # Save the images to disk so that they can be viewed live while the robot is running
    # Create one combined image to make live viewing easy
    if save_to_disk:
        combined_image = np.concatenate([left_image, wrist_image, right_image], axis=1)
        combined_image = Image.fromarray(combined_image)
        combined_image.save("robot_camera_views.png")

    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
