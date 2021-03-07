#!/usr/bin/env python3
from collections import namedtuple
import util as cm
import cv2
import time
import pyrealsense2 as rs
import math
import socket
import numpy as np
from time import sleep
from attrdict import AttrDict
from skeletontracker import skeletontracker

sleep_size = 0.1

bbport = 11000
bbhost = '127.0.0.1'
myname = 'name;controller'
targetname = 'sota;'

CAMERA_RESOLUTION = (1280, 720)
DEPTH_RESOLUTION = (1024, 768)

tm = [[-1 ,0 ,0 ],[0 ,0 ,1],[0, -1, 0]]

obj = AttrDict({
    "pipeline": rs.pipeline(),
    "pointcloud": rs.pointcloud(),
    "config": rs.config()
})
obj.config.enable_stream(
    rs.stream.depth,
    CAMERA_RESOLUTION[0],
    CAMERA_RESOLUTION[1],
    rs.format.z16,
    60
)
obj.config.enable_stream(
    rs.stream.color,
    CAMERA_RESOLUTION[0],
    CAMERA_RESOLUTION[1],
    rs.format.bgr8,
    60
)
obj.config.enable_device("F0233277")

def render_ids_3d(
    render_image, skeletons_2d, depth_map, depth_intrinsic, joint_confidence
):
    thickness = 1
    text_color = (255, 255, 255)
    rows, cols, channel = render_image.shape[:3]
    distance_kernel_size = 5

    depth_pixel_id = -1
    depth_pixel_close = []

    # calculate 3D keypoints and display them
    for skeleton_index in range(len(skeletons_2d)):
        skeleton_2D = skeletons_2d[skeleton_index]
        #print(skeleton_2D.id)
        joints_2D = skeleton_2D.joints
        did_once = False
        for joint_index in range(len(joints_2D)):
            if did_once == False:
                cv2.putText(
                    render_image,
                    "id: " + str(skeleton_2D.id),
                    (int(joints_2D[joint_index].x), int(joints_2D[joint_index].y - 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    text_color,
                    thickness,
                )
                did_once = True
            # check if the joint was detected and has valid coordinate
            if skeleton_2D.confidences[joint_index] > joint_confidence:
                distance_in_kernel = []
                low_bound_x = max(
                    0,
                    int(
                        joints_2D[joint_index].x - math.floor(distance_kernel_size / 2)
                    ),
                )
                upper_bound_x = min(
                    cols - 1,
                    int(joints_2D[joint_index].x + math.ceil(distance_kernel_size / 2)),
                )
                low_bound_y = max(
                    0,
                    int(
                        joints_2D[joint_index].y - math.floor(distance_kernel_size / 2)
                    ),
                )
                upper_bound_y = min(
                    rows - 1,
                    int(joints_2D[joint_index].y + math.ceil(distance_kernel_size / 2)),
                )
                for x in range(low_bound_x, upper_bound_x):
                    for y in range(low_bound_y, upper_bound_y):
                        distance_in_kernel.append(depth_map.get_distance(x, y))
                median_distance = np.percentile(np.array(distance_in_kernel), 50)
                depth_pixel = [
                    int(joints_2D[joint_index].x),
                    int(joints_2D[joint_index].y),
                ]

                if depth_pixel_id == -1:
                    depth_pixel_id = skeleton_2D.id
                    depth_pixel_close = depth_pixel
                elif depth_pixel_id > skeleton_2D.id:
                    depth_pixel_id = skeleton_2D.id
                    depth_pixel_close = depth_pixel

                if median_distance >= 0.3:
                    point_3d = rs.rs2_deproject_pixel_to_point(
                        depth_intrinsic, depth_pixel, median_distance
                    )
                    point_3d = np.round([float(i) for i in point_3d], 3)
                    point_str = [str(x) for x in point_3d]
                    cv2.putText(
                        render_image,
                        str(point_3d),
                        (int(joints_2D[joint_index].x), int(joints_2D[joint_index].y)),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.4,
                        text_color,
                        thickness,
                    )

    if not depth_pixel_id == -1:
        return depth_pixel_close
    else:
        return -1


# Main content begins
try:
    # Configure depth and color streams of the intel realsense
    config = rs.config()

    # Start the realsense pipeline
    pipeline = rs.pipeline()
    pipeline.start()

    # Create align object to align depth frames to color frames
    align = rs.align(rs.stream.color)
    # Get the intrinsics information for calculation of 3D point
    unaligned_frames = pipeline.wait_for_frames()
    frames = align.process(unaligned_frames)
    depth = frames.get_depth_frame()
    depth_intrinsic = depth.profile.as_video_stream_profile().intrinsics

    # Initialize the cubemos api with a valid license key in default_license_dir()
    skeletrack = skeletontracker(cloud_tracking_api_key="")
    joint_confidence = 0.2

    # Create window for initialisation
    window_name = "cubemos skeleton tracking with realsense D400 series"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL + cv2.WINDOW_KEEPRATIO)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((bbhost, bbport))
    except ConnectionRefusedError:
        exit()
    sock.send(myname.encode('utf-8'))
    sleep(sleep_size)

    while True:
        # Create a pipeline object. This object configures the streaming camera and owns it's handle
        unaligned_frames = pipeline.wait_for_frames()
        frames = align.process(unaligned_frames)
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth.get_data())
        color_image = np.asanyarray(color.get_data())

        # perform inference and update the tracking id
        skeletons = skeletrack.track_skeletons(color_image)

        # render the skeletons on top of the acquired image and display it
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        cm.render_result(skeletons, color_image, joint_confidence)

        depth_pixel = render_ids_3d(
            color_image, skeletons, depth, depth_intrinsic, joint_confidence
        )
        points = obj.pointcloud.calculate(depth)
        point_array = np.asanyarray(points.get_vertices())
        point_image = np.reshape(
            point_array,
            CAMERA_RESOLUTION[-1::-1]
        )

        if not depth_pixel == -1:
            face_ccs = np.asanyarray(
                    list(point_image[depth_pixel[1], depth_pixel[0]])
                    )
            face_ccs = (face_ccs * 100).astype(np.int32)

            try:
                a = np.array([face_ccs[0] , face_ccs[1] , face_ccs[2]])
                np.array(tm)
                b = np.dot(tm, a)

                #sotaに送る
                sendmsg = targetname + "posture faceTo {0} {1} {2}".format(str(b[0]) , str(b[1]) , str(b[2]))
                sock.send(sendmsg.encode("utf-8"))

            except RuntimeError:
                continue

        cv2.imshow(window_name, color_image)
        if cv2.waitKey(1) == 27:
            break

    pipeline.stop()
    cv2.destroyAllWindows()

except Exception as ex:
    print('Exception occured: "{}"'.format(ex))
