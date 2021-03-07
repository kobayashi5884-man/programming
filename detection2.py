import pyrealsense2 as rs
import numpy as np
import cv2
import socket
import json
import math as mt
from time import sleep
from attrdict import AttrDict

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

sleep_size = 0.1

bbport = 11000
bbhost = '127.0.0.1'
myname = 'name;controller'
targetname = 'sota;'

CAMERA_RESOLUTION = (1280, 720)
DEPTH_RESOLUTION = (1024, 768)

# ストリーム(Color/Depth)の設定
config = rs.config()
align = rs.align(rs.stream.color)

#config.enable_stream(rs.stream.color, CAMERA_RESOLUTION[0], CAMERA_RESOLUTION[1], rs.format.bgr8, 30)
#config.enable_stream(rs.stream.depth, DEPTH_RESOLUTION[0], DEPTH_RESOLUTION[1], rs.format.z16, 30)

# ストリーミング開始
pipeline = rs.pipeline()
profile = pipeline.start(config)

tm = [[-1 ,0 ,0 ],[0 ,0 ,1],[0, -1, 0]]

distance = []

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

try:
    w_pose, h_pose = 276, 276
    resize_out_ratio = 4.0
    FACE_POINT_IDS = [0, 14, 15, 16, 17]

    # set OpenPose
    # create instance for PoseEstimation
    e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(w_pose, h_pose))

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((bbhost, bbport))
    except ConnectionRefusedError:
        exit()
    sock.send(myname.encode('utf-8'))
    sleep(sleep_size)


    while True:

        # フレーム待ち
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        #RGB
        RGB_frame = aligned_frames.get_color_frame()
        RGB_image = np.asanyarray(RGB_frame.get_data())

        #depyh
        depth_frame = aligned_frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)


        #h_cam, w_cam = RGB_image.shape[:2]
        #RGB_image = RGB_image[int((h_cam - ROI[1])/2):int((h_cam - ROI[1])/2 + ROI[1]), int((w_cam - ROI[0])/2):int((w_cam - ROI[0])/2 + ROI[0])]

        # get human pose
        humans = e.inference(RGB_image, upsample_size = resize_out_ratio)

        # extract particular positions
        for hi, hm in enumerate(humans):
            faces_parts = [(max(min(int(bp.x*1280), 1280), 0), max(min(int(bp.y*720),720), 0))
                   for id_, bp in hm.body_parts.items() if id_ in FACE_POINT_IDS]

            if len(faces_parts) == 0:
                continue

            face_posi = np.average(faces_parts, axis=0).astype(np.int32)
            cv2.circle(RGB_image, (int(face_posi[0]), int(face_posi[1])), 8, (255, 255, 255), -1)

            points = obj.pointcloud.calculate(depth_frame)
            point_array = np.asanyarray(points.get_vertices())
            point_image = np.reshape(
                point_array,
                CAMERA_RESOLUTION[-1::-1]
            )

            face_ccs = np.asanyarray(
                    list(point_image[face_posi[1], face_posi[0]])
                )
            face_ccs = (face_ccs * 100).astype(np.int32)

            distance.append(mt.sqrt(face_ccs[0]**2+face_ccs[1]**2+face_ccs[2]**2))
        i = distance.index(min(distance))
        distance.clear()
        hm = humans[i]
        faces_parts = [(max(min(int(bp.x*1280), 1280), 0), max(min(int(bp.y*720),720), 0))
               for id_, bp in hm.body_parts.items() if id_ in FACE_POINT_IDS]

        points = obj.pointcloud.calculate(depth_frame)
        point_array = np.asanyarray(points.get_vertices())
        point_image = np.reshape(
            point_array,
            CAMERA_RESOLUTION[-1::-1]
        )

        face_ccs = np.asanyarray(
                list(point_image[face_posi[1], face_posi[0]])
            )
        face_ccs = (face_ccs * 100).astype(np.int32)
        #3次元
        try:
            a = np.array([face_ccs[0] , face_ccs[1] , face_ccs[2]])
            np.array(tm)
            b = np.dot(tm, a)

        except RuntimeError:
            continue

        #sotaに送る
        sendmsg = targetname + "posture faceTo {0} {1} {2}".format(str(b[0]) , str(b[1]) , str(b[2]))
        sock.send(sendmsg.encode("utf-8"))






        # 表示
        #images = np.hstack((RGB_image, depth_colormap))
        #cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)face_posi = np.average(faces_parts, axis=0).astype(np.int32)
        # draw human poses
        RGB_image = TfPoseEstimator.draw_humans(RGB_image, humans, imgcopy=False)
        cv2.imshow('RGB', RGB_image)
        cv2.imshow('depth', depth_colormap)
        if cv2.waitKey(1) & 0xff == 27:#ES
            cv2.destroyAllWindows()
            break


finally:
    # ストリーミング停止
    pipeline.stop()
