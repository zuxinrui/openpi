import pyrealsense2 as rs
import numpy as np
import cv2
import sys

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
    cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipe.start(cfg)
    pipelines.append(pipe)
    aligners.append(rs.align(rs.stream.color))  # align depth to color

# ---- 2) Main loop ----
try:
    while True:
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

        print("Displaying combined frame shape:", combined.shape)

        cv2.imshow("RealSense x2 (left: cam0, right: cam1)", combined)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

finally:
    for p in pipelines:
        try:
            p.stop()
        except Exception:
            pass
    cv2.destroyAllWindows()
