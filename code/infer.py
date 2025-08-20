import logging
import os
import datetime
from pascal_voc_writer import Writer
import random
from pathlib import Path
import sys
from PIL import Image
import numpy as np
from scipy.io import savemat
import torch
from torch import autograd
import yaml
import carla
import pygame
import time
import cv2
import ADModel
import RLModel
import AutonomousDrivingEnvironment
import ADTask
from contextlib import contextmanager
import string
from torch import Tensor, ByteTensor
import torch.nn.functional as F
from torch.autograd import Variable
import json
import concurrent.futures
import threading
import math
import torchvision.transforms as trans
import queue
from skimage.measure import label, regionprops


multi_gt_labels = None
data_dict = {}
names_dict = {}
result_container = {}
enh_height = 540
enh_width = 960
other_actor = None
class_color = {"vehicle": (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
               "truck": (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
               "traffic_light": (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
               "traffic_signs": (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
               "bus": (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
               "person": (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
               "motorcycle": (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
               "bicycle": (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
               "rider": (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))}

export_dataset_path = ""
carla_config_path = ""

weather_mapping = {
    "ClearNoon": carla.WeatherParameters.ClearNoon,
    "ClearSunset": carla.WeatherParameters.ClearSunset,
    "CloudyNoon": carla.WeatherParameters.CloudyNoon,
    "CloudySunset": carla.WeatherParameters.CloudySunset,
    "WetNoon": carla.WeatherParameters.WetNoon,
    "WetSunset": carla.WeatherParameters.WetSunset,
    "SoftRainNoon": carla.WeatherParameters.SoftRainNoon,
    "SoftRainSunset": carla.WeatherParameters.SoftRainSunset,
    "HardRainNoon": carla.WeatherParameters.HardRainNoon,
    "HardRainSunset": carla.WeatherParameters.HardRainSunset,
    "WetCloudyNoon": carla.WeatherParameters.WetCloudyNoon,
    "WetCloudySunset": carla.WeatherParameters.WetCloudySunset

}



    # Render object to keep and pass the PyGame surface

    # -------------------------------------------------------------------------------------------------------------------------------------------------------

def convert_image_to_array(self, image):
    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    img = img[:, :, :3]
    img = img[:, :, ::-1]
    return img

def make_gbuffer_matrix(self):
    global enh_height
    global enh_width
    try:
        stacked_image = np.concatenate(
            [data_dict['SceneColor'], data_dict['SceneDepth'][:, :, 0][:, :, np.newaxis], data_dict['GBufferA'],
                data_dict['GBufferB'], data_dict['GBufferC'], data_dict['GBufferD'],
                data_dict['GBufferSSAO'][:, :, 0][:, :, np.newaxis],
                data_dict['CustomStencil'][:, :, 0][:, :, np.newaxis][:, :, 0][:, :, np.newaxis]], axis=-1)
    except:
        return np.zeros((enh_height, enh_width, 18))
    return stacked_image

def split_gt_label(self, gt_labels):
    r = (multi_gt_labels == gt_labels[:, :, 0][:, :, np.newaxis].astype(np.float32))


    class_sky = r[:, :, 0][:, :, np.newaxis]
    class_road = np.any(r[:, :, [1, 2, 3, 4, 5]], axis=2)[:, :, np.newaxis]
    class_vehicle = np.any(r[:, :, [6, 7, 8, 9, 10, 11]], axis=2)[:, :, np.newaxis]
    class_terrain = r[:, :, 12][:, :, np.newaxis]
    class_vegetation = r[:, :, 13][:, :, np.newaxis]
    class_person = np.any(r[:, :, [14, 15]], axis=2)[:, :, np.newaxis]
    class_infa = r[:, :, 16][:, :, np.newaxis]
    class_traffic_light = r[:, :, 17][:, :, np.newaxis]
    class_traffic_sign = r[:, :, 18][:, :, np.newaxis]
    class_ego = np.any(r[:, :, [19, 20]], axis=2)[:, :, np.newaxis]
    class_building = np.any(r[:, :, [21, 22, 23, 24, 25, 26]], axis=2)[:, :, np.newaxis]
    class_unlabeled = np.any(r[:, :, [27, 28]], axis=2)[:, :, np.newaxis]

    concatenated_array = np.concatenate((class_sky, class_road, class_vehicle, class_terrain, class_vegetation,
                                            class_person, class_infa, class_traffic_light, class_traffic_sign, class_ego,
                                            class_building, class_unlabeled), axis=2)
    return concatenated_array

def add_frame(self, image):
    #if "color_frame" not in data_dict:
    img = self.convert_image_to_array(image)
    data_dict["color_frame"] = img
    names_dict["color_frame"] = image.frame


def add_sensor(self,data,sensor_name):
    data_dict[sensor_name] = data
    names_dict[sensor_name] = data.frame

def add_semantic(self, image):
    #if "semantic_segmentation" not in data_dict:
    label_map = self.convert_image_to_array(image)
    data_dict["semantic_segmentation"] = label_map
    names_dict["semantic_segmentation"] = image.frame

def add_gbuffer(self, image, name):
    #if name == 'CustomStencil':
        #self.add_semantic(image)
    #if name not in data_dict:
    data_dict[name] = self.convert_image_to_array(image)
    names_dict[name] = image.frame

    if name == 'CustomStencil': #custom stencil buffer is the same as semantic segmentation camera
        data_dict["semantic_segmentation"] = data_dict[name]
        names_dict["semantic_segmentation"] = names_dict[name]

def make_image(self):
    # img = self.convert_image_to_array(color_frame_list[0])
    img = data_dict['color_frame']

    # renderObject.surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
    result = mat2tensor(img.astype(np.float32) / 255.0)
    return result

# Function 2 to be executed on a separate thread
def make_gtlabels(self):
    # label_map = self.convert_image_to_array(semantic_sementation_list[0])
    label_map = data_dict['semantic_segmentation']

    result = mat2tensor(self.split_gt_label(label_map))
    return result

# Function 3 to be executed on a separate thread
def make_gbuffers(self):
    result = mat2tensor(self.make_gbuffer_matrix().astype(np.float32))
    return result

def render_thread(self, queue, render_screen):
    while True:
        argument = queue.get()
        if argument is None:
            break
        img = (argument[0, ...].clamp(min=0, max=1).permute(1, 2, 0) * 255.0).detach().cpu().numpy().astype(np.uint8)
        render_screen.surface = pygame.surfarray.make_surface(img[:, :, :3].swapaxes(0, 1))

def preprocess_worker(self, name, comp, inputs):
    global result_container
    nameid = 0
    if name == "frame":
        result = self.make_image()
        nameid = 0
    elif name == "gt_labels":
        result = self.make_gtlabels()
        nameid = 2
    else:
        result = self.make_gbuffers()
        nameid = 1
    if comp == 'onnxruntime' or comp == 'tensorrt':
        result_container[name] = np.expand_dims(result.numpy(),axis=0)
        if comp == 'tensorrt':
            inputs[nameid].host = result_container[name]
        elif comp == 'onnxruntime':
            result_container[name] = onnxruntime.OrtValue.ortvalue_from_numpy(result_container[name], 'cuda', 0)
    else:
        result_container[name] = result.pin_memory().to(self.device)

def save_vehicle_status(self, vehicle, frame_id):
    velocity = vehicle.get_velocity()
    speed = 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
    control = vehicle.get_control()
    steering = control.steer
    throttle = control.throttle
    brake = control.brake
    location = vehicle.get_location()
    rotation = vehicle.get_transform().rotation

    vehicle_status = {"velocity": [velocity.x, velocity.y, velocity.z], "speed": speed, "steering": steering,
                        "throttle": throttle, "brake": brake, "location": [location.x, location.y, location.z],
                        "rotation": [rotation.pitch, rotation.yaw, rotation.roll]}

    with open(export_dataset_path + "ADDataset/VehicleStatus/" + str(frame_id) + ".json", "w") as outfile:
        json.dump(vehicle_status, outfile)

def save_world_status(self, town_name, weather_name, vehicle_name, perspective, synchronization, frame_id,
                        real_dataset="cityscapes"):
    world_status = {"town": town_name, "weather": weather_name, "vehicle": vehicle_name, "perspective": perspective,
                    "synchronization_mode": synchronization, "real_dataset": real_dataset}

    with open(export_dataset_path + "ADDataset/WorldStatus/" + str(frame_id) + ".json", "w") as outfile:
        json.dump(world_status, outfile)

def save_frames(self, frame_id, enhanced_frame, carla_config):
    enhanced_frame = (
                enhanced_frame[0, ...].clamp(min=0, max=1).permute(1, 2, 0) * 255.0).detach().cpu().numpy().astype(
        np.uint8)
    im = Image.fromarray(enhanced_frame[:, :, :3])
    im.save(export_dataset_path + "ADDataset/EnhancedFrames/" + str(frame_id) + "." + str(carla_config['dataset_settings']['images_format']))

    im = Image.fromarray(data_dict["color_frame"])
    im.save(export_dataset_path + "ADDataset/RGBFrames/" + str(frame_id) + "." + str(carla_config['dataset_settings']['images_format']))

    if carla_config['dataset_settings']['export_semantic_gt']:
        im = Image.fromarray(data_dict["semantic_segmentation"])
        im.save(export_dataset_path + "ADDataset/SemanticSegmentation/" + str(frame_id) + "." + str(carla_config['dataset_settings']['images_format']))
    if carla_config['dataset_settings']['export_depth']:
        im = Image.fromarray(data_dict["SceneDepth"])
        im.save(export_dataset_path + "ADDataset/Depth/" + str(frame_id) + "." + str(carla_config['dataset_settings']['images_format']))

def random_id_generator(self, n=10, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(n))

def save_rl_stats(self, actor_losses, critic_losses, steps, rewards, current_step,distances,episode,epsilon):
    rl_stats = {"losses": actor_losses, "critic_losses": critic_losses, "steps": steps, "rewards": rewards,"distances":distances,"episodes":episode,"epsilons":epsilon}

    with open("out/rl_stats/rl_stats_"+str(current_step)+".json", "w") as outfile:
        json.dump(rl_stats, outfile)

def on_collision(self, event, environment):
    collision_actor = event.other_actor
    print(collision_actor)
    environment.collision_history.append(collision_actor.type_id)

def is_vehicle_moving(self,vehicle,speed_threshold):
    velocity = vehicle.get_velocity()
    speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
    return speed > speed_threshold

def create_dataset_folders(self,carla_config):
    global export_dataset_path
    if not os.path.isdir(export_dataset_path):
        print('\033[91m'+"Export dataset path is not a valid path in the disk.")
        exit(1)
    if export_dataset_path[-1] == ":":
        export_dataset_path += "/"

    if not os.path.exists(export_dataset_path + "ADDataset" + "/SemanticSegmentation") and carla_config['dataset_settings']['export_semantic_gt']:
        os.makedirs(export_dataset_path + "ADDataset" + "/SemanticSegmentation")
        print(f"Folder SemanticSegmentation created successfully.")
    if not os.path.exists(export_dataset_path + "ADDataset" + "/WorldStatus") and carla_config['dataset_settings']['export_status_json']:
        os.makedirs(export_dataset_path + "ADDataset" + "/WorldStatus")
        print(f"Folder WorldStatus created successfully.")
    if not os.path.exists(export_dataset_path + "ADDataset" + "/VehicleStatus") and carla_config['dataset_settings']['export_status_json']:
        os.makedirs(export_dataset_path + "ADDataset" + "/VehicleStatus")
        print(f"Folder VehicleStatus created successfully.")
    if not os.path.exists(export_dataset_path + "ADDataset" + "/Depth") and carla_config['dataset_settings']['export_depth']:
        os.makedirs(export_dataset_path + "ADDataset" + "/Depth")
        print(f"Folder Depth created successfully.")
    if not os.path.exists(export_dataset_path + "ADDataset" + "/EnhancedFrames"):
        os.makedirs(export_dataset_path + "ADDataset" + "/EnhancedFrames")
        print(f"Folder EnhancedFrames created successfully.")
    if not os.path.exists(export_dataset_path + "ADDataset" + "/RGBFrames"):
        os.makedirs(export_dataset_path + "ADDataset" + "/RGBFrames")
        print(f"Folder RGBFrames created successfully.")
    if not os.path.exists(export_dataset_path + "ADDataset" + "/ObjectDetection") and carla_config['dataset_settings']['export_object_annotations']:
        os.makedirs(export_dataset_path + "ADDataset" + "/ObjectDetection")
        print(f"Folder ObjectDetection created successfully.")

def initialize_gt_labels(self,width=960,height=540,num_channels=29):
    global multi_gt_labels
    specific_classes = [11,1,2,24,25,27,14,15,16,17,18,19,10,9,12,13,6,7,8,21,23,20,3,4,5,26,28,0,22]
    multi_channel_array = np.zeros((num_channels, height, width))
    for channel_index, value in enumerate(specific_classes):
        multi_channel_array[channel_index, :, :] = value
    multi_gt_labels = np.transpose(multi_channel_array, axes=(1, 2, 0))

def stabilize_vehicle(self,world,spectator_camera_mode,camera,num_ticks,data_length):
    global data_dict
    global names_dict

    for i in range(num_ticks):
        data_dict = {}
        names_dict = {}
        world.tick()
        if spectator_camera_mode == 'follow':
            world.get_spectator().set_transform(camera.get_transform())
        while True:
            if len(data_dict) == data_length:
                break
        data_dict = {}
        names_dict = {}

def get_transform_from_field(self,world,scenario_config,field_name):
    if scenario_config[field_name]['init_spawn_point'] == 'random':
        transform = world.get_map().get_spawn_points()[random.randint(0, len(world.get_map().get_spawn_points()) - 1)]
    elif isinstance(scenario_config[field_name]['init_spawn_point'], int):
        transform = world.get_map().get_spawn_points()[scenario_config[field_name]['init_spawn_point']]
    elif isinstance(scenario_config[field_name]['init_spawn_point'], list):
        coords = scenario_config[field_name]['init_spawn_point']
        transform = carla.Transform(carla.Location(x=coords[0][0], y=coords[0][1], z=coords[0][2]), carla.Rotation(coords[1][0],coords[1][1],coords[1][2]))
    return transform

def initialize_movement(self,scenario_config,controls):
    global other_actor
    if scenario_config['other_actor']['static'] == False:
        if isinstance(other_actor, carla.Vehicle):
            init_controls = scenario_config['other_actor'][controls]
            other_actor.apply_control(carla.VehicleControl(throttle=init_controls[1], steer=init_controls[0], brake=init_controls[2]))
        elif isinstance(other_actor, carla.Walker):
            init_controls = scenario_config['other_actor'][controls]
            walker_control = carla.WalkerControl()
            walker_control.speed = init_controls[0]  # Set the desired speed in m/s
            walker_control.direction = carla.Vector3D(x=init_controls[1][0], y=init_controls[1][1], z=init_controls[1][2])  # Set the direction
            other_actor.apply_control(walker_control)
        else:
            print('\033[91m'+"Not compatible actor. Choose a vehicle or pederestrian actor instance.")
            exit(1)

def get_image_point(self,loc, K, w2c):

    point = np.array([loc.x, loc.y, loc.z, 1])

    point_camera = np.dot(w2c, point)

    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    point_img = np.dot(K, point_camera)

    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]

def build_projection_matrix(self,w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_vehicles_mask(self,detected_vehicle_mask,segmentation,detected_id):
    vehicles_ids = [13,14,15,16,18,19]
    for vid in vehicles_ids:
        if vid == detected_id:
            continue
        vid_mask = (segmentation == vid)
        detected_vehicle_mask = np.logical_or(detected_vehicle_mask,vid_mask)
    return detected_vehicle_mask


def is_valid_bbox(self,bbox,segmentation,type,type_pixels_thresh,type_pixels_zero_thresh):
    segmentation = segmentation[:,:,0]
    type_map = {"person":12,"vehicle":14,"truck":15,"bus":16,"traffic_light":7,"traffic_signs":8,"motorcycle":18,"bicycle":19,"rider":13}

    type_max_id = type_map[type]
    type_mask = (segmentation == type_max_id)
    type_mask_single = (segmentation == type_max_id)
    if type_map[type] in [13,14,15,16,18,19]:
        type_mask = self.get_vehicles_mask(type_mask,segmentation,type_map[type])
    xmin, ymin, xmax, ymax = bbox
    bottom_right = (int(xmax), int(ymax))
    top_left = (int(xmin), int(ymin))
    roi = type_mask.astype(np.uint8)[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    roi_single = type_mask_single.astype(np.uint8)[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    count_true_pixels = np.sum(roi == 1)
    count_false_pixels = np.sum(roi == 0)
    if count_true_pixels > type_pixels_thresh[type] and count_false_pixels > type_pixels_zero_thresh[type] and np.sum(roi_single == 1) > type_pixels_thresh[type]:
        return True
    else:
        return False

def is_bbox_overlaping(self,bbox,bbox_list):
    for bb in bbox_list:
        outer_x1, outer_y1, outer_x2, outer_y2 = bbox
        inner_x1, inner_y1, inner_x2, inner_y2 = bb
        is_inside = (outer_x1 <= inner_x1) and (outer_y1 <= inner_y1) and (outer_x2 >= inner_x2) and (outer_y2 >= inner_y2)

        if is_inside:
            return True
    return False

def bbox_from_mask(self,type,writer,carla_config,frame):
    type_pixels_thresh = dict(carla_config['dataset_settings']['object_class_numpixel_threshold'])
    type_map = {"person": 12, "vehicle": 14, "truck": 15, "bus": 16, "traffic_light": 7, "traffic_signs": 8,"motorcycle": 18, "bicycle": 19,"rider":13}
    type_mask = (data_dict['semantic_segmentation'][:,:,0] == type_map[type])

    lbl_0 = label(type_mask)
    props = regionprops(lbl_0)
    for prop in props:
        x_min = prop.bbox[1]
        y_min = prop.bbox[0]
        x_max = prop.bbox[3]
        y_max = prop.bbox[2]
        bottom_right = (int(x_max), int(y_max))
        top_left = (int(x_min), int(y_min))
        roi = type_mask.astype(np.uint8)[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        count_true_pixels = np.sum(roi == 1)

        if count_true_pixels > type_pixels_thresh[type] and (roi.shape[0] * roi.shape[1]) > carla_config['dataset_settings']['object_bbox_shape_threshold']:
            writer.addObject(type, x_min, y_min, x_max, y_max)
            cv2.rectangle(frame, (int(x_max), int(y_max)), (int(x_min), int(y_min)), class_color[type], 2)
            cv2.putText(frame, type, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,class_color[type], 2)

#code from https://github.com/carla-simulator/carla/issues/3801
def get_bounding_box(self, actor, min_extent=0.5):
    """
    Some actors like motorbikes have a zero width bounding box. This is a fix to this issue.
    """
    if not hasattr(actor, "bounding_box"):
        return carla.BoundingBox(carla.Location(0, 0, min_extent),
                                    carla.Vector3D(x=min_extent, y=min_extent, z=min_extent))

    bbox = actor.bounding_box

    buggy_bbox = (bbox.extent.x * bbox.extent.y * bbox.extent.z == 0)
    if buggy_bbox:
        bbox.location = carla.Location(0, 0, max(bbox.extent.z, min_extent))

    if bbox.extent.x < min_extent:
        bbox.extent.x = min_extent
    if bbox.extent.y < min_extent:
        bbox.extent.y = min_extent
    if bbox.extent.z < min_extent:
        bbox.extent.z = min_extent

    return bbox

def analyze_semantic_lidar_occlusion(self,actor_id):
    global data_dict
    if not ( 'lidar_semantic' in data_dict ):
        return True

    for detection in data_dict['lidar_semantic']:
        if detection.object_idx == actor_id:
            return True
    return False


def save_object_detection_annotations(self,camera,world,vehicle,frame_id,carla_config):
    global class_colors
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

    image_w = int(carla_config['ego_vehicle_settings']['camera_width'])
    image_h = int(carla_config['ego_vehicle_settings']['camera_height'])
    image_fov = int(carla_config['ego_vehicle_settings']['camera_fov'])
    listed_classes = list(carla_config['dataset_settings']['object_annotations_classes'])
    object_max_distance = carla_config['dataset_settings']['object_annotation_distance']
    dot_product_threshold = carla_config['dataset_settings']['object_dot_product_threshold']
    bbox_min_extend = carla_config['dataset_settings']['object_bbox_min_extent']
    current_frame = np.ascontiguousarray(data_dict['color_frame'], dtype=np.uint8)
    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

    writer = Writer(export_dataset_path+"ADDataset/Frames/"+ str(frame_id) + '.png', image_w, image_h)
    K = self.build_projection_matrix(image_w, image_h, image_fov)

    vehicles = ['vehicle.dodge.charger_2020','vehicle.dodge.charger_police','vehicle.dodge.charger_police_2020','vehicle.ford.crown','vehicle.ford.mustang','vehicle.jeep.wrangler_rubicon','vehicle.lincoln.mkz_2017','vehicle.lincoln.mkz_2020'
                ,'vehicle.mercedes.coupe','vehicle.mercedes.coupe_2020','vehicle.micro.microlino','vehicle.mini.cooper_s','vehicle.mini.cooper_s_2021','vehicle.nissan.micra','vehicle.nissan.patrol','vehicle.nissan.patrol_2021',
                'vehicle.seat.leon','vehicle.tesla.model3','vehicle.toyota.prius','vehicle.audi.a2','vehicle.audi.etron','vehicle.audi.tt','vehicle.bmw.grandtourer','vehicle.chevrolet.impala','vehicle.citroen.c3']
    trucks = ['vehicle.carlamotors.carlacola','vehicle.carlamotors.european_hgv','vehicle.carlamotors.firetruck','vehicle.tesla.cybertruck','vehicle.ford.ambulance','vehicle.mercedes.sprinter','vehicle.volkswagen.t2','vehicle.volkswagen.t2_2021']
    buses = ['vehicle.mitsubishi.fusorosa']
    motorcycles = ['vehicle.harley-davidson.low_rider','vehicle.kawasaki.ninja','vehicle.vespa.zx125','vehicle.yamaha.yzf']
    bikes = ['vehicle.bh.crossbike','vehicle.diamondback.century','vehicle.gazelle.omafiets']

    if 'rider' in listed_classes:
        self.bbox_from_mask("rider", writer, carla_config, current_frame)

    bboxes_list = []

    for npc in world.get_actors():
        if npc.id != vehicle.id:
            type = None
            if npc.type_id.startswith('walker.pedestrian') and 'person' in listed_classes:
                type = 'person'
            elif npc.type_id in vehicles and 'vehicle' in listed_classes:
                type = 'vehicle'
            elif npc.type_id in trucks and 'truck' in listed_classes:
                type = 'truck'
            elif npc.type_id in buses and 'bus' in listed_classes:
                type = 'bus'
            elif npc.type_id in motorcycles and 'motorcycle' in listed_classes:
                type = 'motorcycle'
            elif npc.type_id in bikes and 'bicycle' in listed_classes:
                type = 'bicycle'
            else:
                continue

            bb = self.get_bounding_box(npc,bbox_min_extend)
            dist = npc.get_transform().location.distance(vehicle.get_transform().location)

            if dist < object_max_distance:
                forward_vec = vehicle.get_transform().get_forward_vector()
                ray = npc.get_transform().location - vehicle.get_transform().location

                if forward_vec.dot(ray) > dot_product_threshold:

                    if self.analyze_semantic_lidar_occlusion(npc.id) == False:
                        continue

                    x_max = -10000
                    x_min = 10000
                    y_max = -10000
                    y_min = 10000
                    verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                    for vert in verts:
                        p = self.get_image_point(vert, K, world_2_camera)
                        if p[0] > x_max:
                            x_max = p[0]
                        if p[0] < x_min:
                            x_min = p[0]
                        if p[1] > y_max:
                            y_max = p[1]
                        if p[1] < y_min:
                            y_min = p[1]

                    if x_min > 0 and x_max < image_w and y_min > 0 and y_max < image_h:
                        if carla_config['dataset_settings']['use_only_semantic_lidar'] and carla_config['dataset_settings']['use_semantic_lidar']:
                            writer.addObject(type, x_min, y_min, x_max, y_max)
                            bboxes_list.append([int(x_min), int(y_min), int(x_max), int(y_max)])
                            cv2.rectangle(current_frame, (int(x_max), int(y_max)), (int(x_min), int(y_min)), class_color[type], 2)
                            cv2.putText(current_frame, type, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_color[type], 2)
                            continue


                        if self.is_valid_bbox([x_min, y_min, x_max, y_max], data_dict['semantic_segmentation'],type,dict(carla_config['dataset_settings']['object_class_numpixel_threshold']),dict(carla_config['dataset_settings']['object_class_numpixel_zero_threshold'])):
                            writer.addObject(type, x_min, y_min, x_max, y_max)
                            bboxes_list.append([int(x_min), int(y_min), int(x_max), int(y_max)])
                            cv2.rectangle(current_frame, (int(x_max), int(y_max)), (int(x_min), int(y_min)), class_color[type], 2)
                            cv2.putText(current_frame, type, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_color[type], 2)

    bounding_box_set = []
    bbnames = []
    if 'traffic_light' in listed_classes:
        bounding_box_set = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
        for n in range(len(bounding_box_set)):
            bbnames.append("traffic_light")
    if 'traffic_sign' in listed_classes:
        bounding_box_set.extend(world.get_level_bbs(carla.CityObjectLabel.TrafficSigns))
        for n in range(len(world.get_level_bbs(carla.CityObjectLabel.TrafficSigns))):
            bbnames.append("traffic_signs")
    if len(bounding_box_set) > 0:
        for bb in range(len(bounding_box_set)):
            if bounding_box_set[bb].location.distance(vehicle.get_transform().location) < object_max_distance:
                forward_vec = vehicle.get_transform().get_forward_vector()
                ray = bounding_box_set[bb].location - vehicle.get_transform().location
                if forward_vec.dot(ray) > dot_product_threshold:

                    x_max = -10000
                    x_min = 10000
                    y_max = -10000
                    y_min = 10000
                    verts = [v for v in bounding_box_set[bb].get_world_vertices(carla.Transform())]
                    for vert in verts:
                        p = self.get_image_point(vert, K, world_2_camera)
                        if p[0] > x_max:
                            x_max = p[0]
                        if p[0] < x_min:
                            x_min = p[0]
                        if p[1] > y_max:
                            y_max = p[1]
                        if p[1] < y_min:
                            y_min = p[1]


                    if x_min > 0 and x_max < image_w and y_min > 0 and y_max < image_h:
                        if self.is_valid_bbox([x_min, y_min, x_max, y_max], data_dict['semantic_segmentation'],bbnames[bb],dict(carla_config['dataset_settings']['object_class_numpixel_threshold']),dict(carla_config['dataset_settings']['object_class_numpixel_zero_threshold'])):
                            writer.addObject(bbnames[bb], x_min, y_min, x_max, y_max)
                            bboxes_list.append([int(x_min), int(y_min), int(x_max), int(y_max)])
                            cv2.rectangle(current_frame, (int(x_max), int(y_max)), (int(x_min), int(y_min)), class_color[bbnames[bb]], 2)
                            cv2.putText(current_frame, bbnames[bb], (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, class_color[bbnames[bb]], 2)

    writer.save(export_dataset_path + "ADDataset/ObjectDetection/" + str(frame_id) + ".xml")

    if carla_config['dataset_settings']['object_visualize_annotations'] == True:
        cv2.imshow('Object Detection Annotations', current_frame)

def initialize_scenario(self,world,scenario_config,vehicle):
    global other_actor

    vehicle.set_target_velocity(carla.Vector3D(x=0, y=0, z=0))
    ego_transform = self.get_transform_from_field(world,scenario_config,'ego_vehicle_settings')
    vehicle.set_transform(ego_transform)

    if other_actor == None and scenario_config['other_actor']['actor_id'] is not None:
        blueprint_library = world.get_blueprint_library()
        available_blueprints = blueprint_library.filter(scenario_config['other_actor']['actor_id'])
        vehicle_blueprint = available_blueprints[0]
        transform = self.get_transform_from_field(world,scenario_config,'other_actor')
        other_actor = world.spawn_actor(vehicle_blueprint, transform)
        self.initialize_movement(scenario_config,'init_controls')
    elif other_actor is not None:
        other_actor.set_target_velocity(carla.Vector3D(x=0, y=0, z=0))
        self.initialize_movement(scenario_config, 'init_controls')
        transform = self.get_transform_from_field(world,scenario_config,'other_actor')
        other_actor.set_transform(transform)

    if scenario_config['general']['traffic_lights'] == 'green':
        list_actor = world.get_actors()
        for actor_ in list_actor:
            if isinstance(actor_, carla.TrafficLight):
                actor_.set_state(carla.TrafficLightState.Green)
                actor_.set_green_time(scenario_config['general']['traffic_lights_time'])
    elif scenario_config['general']['traffic_lights'] == 'red':
        for actor_ in list_actor:
            if isinstance(actor_, carla.TrafficLight):
                actor_.set_state(carla.TrafficLightState.Red)
                actor_.set_red_time(scenario_config['general']['traffic_lights_time'])

    return ego_transform

def compare_distance(self,comparison_critiria,dist,threshold):
    result = False
    comparison_critiria = str(comparison_critiria)
    if comparison_critiria == 'greater':
        result = dist > threshold
    elif comparison_critiria == 'less':
        result = dist < threshold
    elif comparison_critiria == 'equals':
        result = dist == threshold
    elif comparison_critiria == 'not equals':
        result = dist != threshold
    elif comparison_critiria == 'greater_equals':
        result = dist >= threshold
    elif comparison_critiria == 'less_equals':
        result = dist <= threshold
    return result

def manual_controls_apply(self,vehicle,manual_controls):
    keys = pygame.key.get_pressed()

    throttle = 0.0
    steer = 0.0
    brake = 0.0

    if keys[pygame.K_UP] or keys[pygame.K_w]:
        throttle = manual_controls[0]
    elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
        brake = manual_controls[2]
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        steer = -1 * manual_controls[1]
    elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        steer = manual_controls[1]

    vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))

def trigger_scenario(self,world,scenario_config,vehicle):
    global other_actor

    if other_actor is not None and scenario_config['other_actor']['static'] is not None and scenario_config['other_actor']['static'] == False:
        ego_transform = vehicle.get_transform()
        other_transform = other_actor.get_transform()

        ego_location = ego_transform.location
        other_location = other_transform.location

        distance = math.sqrt((ego_location.x - other_location.x) ** 2 + (ego_location.y - other_location.y) ** 2 + (ego_location.z - other_location.z) ** 2)
        print("Distance to other actor: "+ str(round(distance, 2))+ "m")
        #print(self.compare_distance(scenario_config['other_actor']['threshold_critiria'],distance,scenario_config['other_actor']['distance_threshold']))
        if self.compare_distance(scenario_config['other_actor']['threshold_critiria'],distance,scenario_config['other_actor']['distance_threshold']):
            self.initialize_movement(scenario_config,'out_controls')




def validate_parameters(self,carla_config):
    from colorama import Back, Style

    if not carla_config['general']['data_type'] in ['fp32','fp16','tf32','int8']:
        print('\033[91m'+"Invalid parameter data_type. Valid values are ['fp32','fp16','tf32'].")
        exit(1)
    if not carla_config['general']['pygame_output'] in ['enhanced', 'rgb','ad_task']:
        print('\033[91m'+"Invalid parameter pygame_output. Valid values are ['enhanced', 'rgb','ad_task'].")
        exit(1)
    if not type(carla_config['general']['run_enhanced_model']) == bool:
        print('\033[91m'+"Invalid parameter run_enhanced_model. Valid values are True or False.")
        exit(1)
    if not carla_config['general']['compiler'] in ['tensorrt','onnxruntime','pytorch']:
        print('\033[91m'+"Invalid parameter compiler. Valid values are ['tensorrt','onnxruntime','pytorch'].")
        exit(1)
    if not carla_config['carla_world_settings']['camera_output'] in ['enhanced', 'rgb']:
        print('\033[91m'+"Invalid parameter camera_output. Valid values are ['enhanced', 'rgb'].")
        exit(1)
    if not carla_config['carla_world_settings']['driving_mode'] in ['ad_model', 'auto','rl_train','rl_eval','manual']:
        print('\033[91m'+"Invalid parameter driving_mode. Valid values are ['ad_model', 'auto','rl_train','rl_eval','manual'].")
        exit(1)
    if not (carla_config['carla_world_settings']['perspective'] == 0 or carla_config['carla_world_settings']['perspective'] == 1 or carla_config['carla_world_settings']['perspective'] == 2):
        print('\033[91m'+"Invalid parameter perspective. Valid values are [0, 1, 2].")
        exit(1)
    if not carla_config['carla_world_settings']['weather_preset'] in ["Default","ClearNoon","ClearSunset","CloudyNoon","CloudySunset","WetNoon","WetSunset","SoftRainNoon","SoftRainSunset","HardRainNoon","HardRainSunset","WetCloudyNoon","WetCloudySunset"]:
        print('\033[91m'+'Invalid parameter weather_preset. Valid values are ["Default","ClearNoon","ClearSunset","CloudyNoon","CloudySunset","WetNoon","WetSunset","SoftRainNoon","SoftRainSunset","HardRainNoon","HardRainSunset","WetCloudyNoon","WetCloudySunset"].')
        exit(1)
    if not carla_config['carla_world_settings']['spectator_camera_mode'] in ['follow','free']:
        print('\033[91m'+"Invalid parameter spectator_camera_mode. Valid values are ['follow','free'].")
        exit(1)
    if not type(carla_config['carla_world_settings']['load_parked_vehicles']) == bool:
        print('\033[91m'+"Invalid parameter load_parked_vehicles. Valid values are True or False.")
        exit(1)
    if not type(carla_config['carla_world_settings']['sync_mode']) == bool:
        print('\033[91m'+"Invalid parameter sync_mode. Valid values are True or False.")
        exit(1)
    if not type(carla_config['dataset_settings']['export_dataset']) == bool:
        print('\033[91m' + "Invalid parameter export_dataset. Valid values are True or False.")
        exit(1)

    if carla_config['general']['compiler'] == 'onnxruntime' and carla_config['general']['data_type'] == 'fp16':
        print('\033[93m'+"Warning...onnxruntime does not work correctly with fp16. Use fp32 or tensorrt compiler instead.")

    if not carla_config['general']['compiler'] == 'tensorrt' and carla_config['general']['data_type'] == 'int8':
        print('\033[91m'+"INT8 data type is only available for tensorrt compiler.")
        exit(1)

    if not carla_config['dataset_settings']['images_format'] in ['png','jpg']:
        print('\033[91m'+"The supported images formats are png and jpg.")
        exit(1)

    if carla_config['carla_world_settings']['sync_mode'] == False:
        print('\033[93m' + "Warning...Running in asynchronous mode may lead to different problems depending on the system computing capability. We recommend using asynchronous mode only for visualization.")

    if carla_config['carla_world_settings']['sync_mode'] == False and carla_config['dataset_settings']['export_dataset'] == True:
        print('\033[91m' + "Data extraction is supported only is in synchronous mode.")
        exit(1)

    if carla_config['carla_world_settings']['camera_output'] == 'enhanced' and carla_config['general']['run_enhanced_model'] == False:
        print('\033[91m' + "Please enable 'run_enhanced_model' parameter in carla_config if you want to use enhanced camera output.")
        exit(1)

    if carla_config['general']['async_data_transfer'] == True and (not carla_config['general']['compiler'] == 'tensorrt' or not carla_config['carla_world_settings']['camera_output'] == 'enhanced' or carla_config['carla_world_settings']['sync_mode'] == True):
        print('\033[91m' + "Async data transfer is only supported in asynchronous mode using tensorrt compiler and enhanced camera output.")
        exit(1)

    if "Rain" in carla_config['carla_world_settings']['weather_preset'] and "cityscapes" in self.weight_init:
        print('\033[93m' + "Warning...If you select rain weather presets with Cityscapes pretrained models we recommend enabling no_render_mode perameter.")


def evaluate_infer(self):
    global data_dict
    global names_dict
    global weather_mapping
    global export_dataset_path
    global enh_width
    global enh_height
    found_scenario = True


    with open(carla_config_path, 'r') as file:
        carla_config = yaml.safe_load(file)

    self.validate_parameters(carla_config)

    val_ticks = None
    try:
        with open("../scenarios/"+str(carla_config['autonomous_driving']['scenario']+".yaml"), 'r') as file:
            scenario_config = yaml.safe_load(file)
        val_ticks = scenario_config['general']['val_ticks']
    except:
        found_scenario = False

    print("Clearing older ONNX profiles...")
    file_list = os.listdir(os.getcwd())
    for filename in file_list:
        if "onnxruntime_profile__" in filename:
            file_path = os.path.join(os.getcwd(), filename)
            os.remove(file_path)
            print(f"Deleted: {file_path}")

    actor_list = []

    client = carla.Client(carla_config['carla_server_settings']['ip'], carla_config['carla_server_settings']['port'])
    client.set_timeout(carla_config['carla_server_settings']['timeout'])
    sync_mode = carla_config['carla_world_settings']['sync_mode']
    selected_town = carla_config['carla_world_settings']['town']
    selected_weather_preset = carla_config['carla_world_settings']['weather_preset']
    selected_perspective = carla_config['carla_world_settings']['perspective']
    export_dataset = carla_config['dataset_settings']['export_dataset']
    export_dataset_path = carla_config['dataset_settings']['dataset_path']
    driving_mode = carla_config['carla_world_settings']['driving_mode']
    selected_camera_output = carla_config['carla_world_settings']['camera_output']
    num_skip_frames = carla_config['carla_world_settings']['skip_frames']
    rl_action_dim = carla_config['autonomous_driving']['rl_action_dim']
    rl_model_load_episode = carla_config['autonomous_driving']['rl_model_load_episode']
    rl_model_name = carla_config['autonomous_driving']['rl_model_name']
    spectator_camera_mode = carla_config['carla_world_settings']['spectator_camera_mode']
    stabilize_ticks = carla_config['autonomous_driving']['stabilize_num_ticks']
    use_lidar = carla_config['other_sensors_settings']['use_lidar']
    use_radar = carla_config['other_sensors_settings']['use_radar']
    use_imu = carla_config['other_sensors_settings']['use_imu']
    use_gnss = carla_config['other_sensors_settings']['use_gnss']
    compiler = carla_config['general']['compiler']
    async_data_transfer = carla_config['general']['async_data_transfer']
    onnx_path = "..\\checkpoints\\ONNX\\"+self.weight_init+".onnx"
    dtype = carla_config['general']['data_type']
    rl_ego_transform = None
    ticks_counter = 0
    data_length = 10
    inputs = None
    manual_controls = carla_config['carla_world_settings']['manual_controls']

    if '/Game/Carla/Maps/'+selected_town not in client.get_available_maps():
        print('\033[91m'+"The selected Town does not exist. Below is a list with all the available Towns:")
        print(client.get_available_maps())
        exit(1)

    if use_lidar:
        data_length += 1
    if use_radar:
        data_length += 1
    if use_imu:
        data_length += 1
    if use_gnss:
        data_length += 1

    if selected_camera_output == 'enhanced' and carla_config['general']['run_enhanced_model'] == False:
        print('\033[91m'+"When using enhanced camera output the run_enhanced_model option must be enabled.")
        exit(1)

    if export_dataset == True:
        self.create_dataset_folders(carla_config)

    if compiler == 'onnxruntime' or compiler == 'tensorrt':
        if not os.path.exists(onnx_path):

            path = "..\checkpoints\ONNX"
            print('The onnx model path does not exist. Trying to generate at '+path+".")
            if not os.path.exists(path):
                os.makedirs(path)
                print("Created ONNX directory...")
            path = os.path.join(path,self.weight_init+".onnx")
            self.generate_onxx(path,carla_config['ego_vehicle_settings']['camera_width'],carla_config['ego_vehicle_settings']['camera_height'])
            print("ONNX file generated successfully at "+path+".")
    if compiler == 'onnxruntime':
        opts = onnxruntime.SessionOptions()
        opts.enable_profiling = True

        if dtype == 'fp16':
            if not os.path.exists(os.path.join("..\checkpoints\ONNX",self.weight_init+"_fp16.onnx")):
                model = onnx.load(os.path.join("..\checkpoints\ONNX",self.weight_init+".onnx"))
                model_fp16 = onnxconverter_common.convert_float_to_float16(model)
                onnx.save(model_fp16, os.path.join("..\checkpoints\ONNX",self.weight_init+"_fp16.onnx"))
            onnx_path = os.path.join("..\checkpoints\ONNX",self.weight_init+"_fp16.onnx")
        session = onnxruntime.InferenceSession(
            onnx_path,
            opts,
            providers=[
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "cudnn_conv_use_max_workspace": "1",
                        "cudnn_conv_algo_search": "DEFAULT",
                    },
                ),
                "CPUExecutionProvider",
            ],
        )
        io_binding = session.io_binding()
    elif compiler == 'tensorrt':
        try:
            pass
            sys.path.insert(1, carla_config['general']['tensorrt_common_path'])
            import common
            import tensorrt as trt
            tensorrt_success = True
        except:
            print('\033[93m'+"Error importing tensorrt. Please check your TensorRT installation...Setting Pytorch as the compiler.")
            tensorrt_success = False

        if tensorrt_success:
            path, file = os.path.split(onnx_path)
            trt_file = file.split(".")[0]+"_"+str(dtype)+".trt"
            cache_file = file.split(".")[0]+".cache"
            TRT_LOGGER = trt.Logger(trt.Logger.INFO)
            trt.init_libnvinfer_plugins(TRT_LOGGER, '')
            engine = self.get_engine(onnx_path, os.path.join(path,trt_file),dtype,os.path.join(path,cache_file),carla_config['ego_vehicle_settings']['camera_width'],carla_config['ego_vehicle_settings']['camera_height'],carla_config['general']['calibration_dataset'])
            context = engine.create_execution_context()
            inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        else:
            compiler = 'pytorch'

    #if pytorch compiler is not used then delete the pytorch model and free the memory
    if compiler == 'tensorrt' or compiler == 'onnxruntime':
        del self.network.generator

    world = client.get_world()
    world = client.load_world(selected_town, carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)

    if carla_config['carla_world_settings']['load_parked_vehicles'] == False:
        world.unload_map_layer(carla.MapLayer.ParkedVehicles)

    settings = world.get_settings()
    settings.synchronous_mode = sync_mode
    settings.no_rendering_mode = carla_config['carla_world_settings']['no_rendering_mode']
    settings.fixed_delta_seconds = carla_config['carla_world_settings']['fixed_delta_seconds']
    world.apply_settings(settings)

    if str(selected_weather_preset) != "Default":
        world.set_weather(weather_mapping[selected_weather_preset])

    blueprint_library = world.get_blueprint_library()

    cars = ['vehicle.chevrolet.impala', 'vehicle.tesla.model3', 'vehicle.mercedes.coupe_2020']
    car_cam_coord = [[1.2, -0.3, 1.4], [0.8, -0.3, 1.4], [1.0, -0.3, 1.4]]
    if carla_config['ego_vehicle_settings']['vehicle_model'] == 'random':
        selected_random_car = random.randint(0, len(cars) - 1)
        bp = random.choice(blueprint_library.filter(cars[selected_random_car]))
        selected_vehicle_name = cars[selected_random_car]
    else:
        selected_random_car = 0
        bp = blueprint_library.filter(carla_config['ego_vehicle_settings']['vehicle_model'])[0]
        selected_vehicle_name = carla_config['ego_vehicle_settings']['vehicle_model']

    actor_list_remaining = world.get_actors()

    for actor in actor_list_remaining:
        if 'vehicle' in actor.type_id or 'walker.pedestrian' in actor.type_id:
            actor.destroy()

    if bp.has_attribute('color'):
        color = random.choice(bp.get_attribute('color').recommended_values)
        bp.set_attribute('color', color)

    if carla_config['ego_vehicle_settings']['init_spawn_point'] == 'random':
        transform = world.get_map().get_spawn_points()[random.randint(0, len(world.get_map().get_spawn_points()) - 1)]
    elif isinstance(carla_config['ego_vehicle_settings']['init_spawn_point'], int):
        transform = world.get_map().get_spawn_points()[carla_config['ego_vehicle_settings']['init_spawn_point']]
    elif isinstance(carla_config['ego_vehicle_settings']['init_spawn_point'], list):
        coords = carla_config['ego_vehicle_settings']['init_spawn_point']
        transform = carla.Transform(carla.Location(x=coords[0][0], y=coords[0][1], z=coords[0][2]), carla.Rotation(coords[1][0],coords[1][1],coords[1][2]))

    vehicle = world.spawn_actor(bp, transform)

    actor_list.append(vehicle)
    print('created %s' % vehicle.type_id)

    if driving_mode == "auto":
        vehicle.set_autopilot(True)
    else:
        vehicle.set_autopilot(False)

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(carla_config['ego_vehicle_settings']['camera_width']))
    camera_bp.set_attribute('image_size_y', str(carla_config['ego_vehicle_settings']['camera_height']))
    camera_bp.set_attribute('fov', str(carla_config['ego_vehicle_settings']['camera_fov']))

    if selected_perspective == 1:
        camera_transform = carla.Transform(
            carla.Location(x=car_cam_coord[selected_random_car][0], y=car_cam_coord[selected_random_car][1],
                            z=car_cam_coord[selected_random_car][2]))
    elif selected_perspective == 0:
        camera_transform = carla.Transform(carla.Location(x=1.5, z=1.4))
    elif selected_perspective == 2:
        camera_transform = carla.Transform(carla.Location(x=carla_config['ego_vehicle_settings']['camera_location'][0],y=carla_config['ego_vehicle_settings']['camera_location'][1] ,z=carla_config['ego_vehicle_settings']['camera_location'][2]))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    actor_list.append(camera)
    print('created %s' % camera.type_id)

    #camera_bp2 = blueprint_library.find('sensor.camera.semantic_segmentation')
    #camera_bp2.set_attribute('image_size_x', str(carla_config['ego_vehicle_settings']['camera_width']))
    #camera_bp2.set_attribute('image_size_y', str(carla_config['ego_vehicle_settings']['camera_height']))
    #camera_bp2.set_attribute('fov', str(carla_config['ego_vehicle_settings']['camera_fov']))
    #camera_semseg = world.spawn_actor(camera_bp2, camera_transform, attach_to=vehicle)

    #actor_list.append(camera_semseg)
    #print('created %s' % camera_semseg.type_id)

    if use_lidar:
        lidar_transform_conf = carla_config['lidar_settings']['transform']
        lidar_transform = carla.Transform(carla.Location(x=lidar_transform_conf[0][0],y=lidar_transform_conf[0][1] ,z=lidar_transform_conf[0][2]),carla.Rotation(lidar_transform_conf[0][0],lidar_transform_conf[0][1],lidar_transform_conf[0][2]))
        lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels',str(carla_config['lidar_settings']['channels']))
        lidar_bp.set_attribute('range', str(carla_config['lidar_settings']['range']))
        lidar_bp.set_attribute('points_per_second', str(carla_config['lidar_settings']['points_per_second']))
        lidar_bp.set_attribute('rotation_frequency', str(carla_config['lidar_settings']['rotation_frequency']))
        lidar_bp.set_attribute('upper_fov', str(carla_config['lidar_settings']['upper_fov']))
        lidar_bp.set_attribute('horizontal_fov', str(carla_config['lidar_settings']['horizontal_fov']))
        lidar_bp.set_attribute('atmosphere_attenuation_rate', str(carla_config['lidar_settings']['atmosphere_attenuation_rate']))
        lidar_bp.set_attribute('dropoff_general_rate', str(carla_config['lidar_settings']['dropoff_general_rate']))
        lidar_bp.set_attribute('dropoff_intensity_limit', str(carla_config['lidar_settings']['dropoff_intensity_limit']))
        lidar_bp.set_attribute('dropoff_zero_intensity', str(carla_config['lidar_settings']['dropoff_zero_intensity']))
        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
        actor_list.append(lidar)
        print('created %s' % lidar.type_id)

    if use_radar:
        radar_transform_conf = carla_config['radar_settings']['transform']
        radar_transform = carla.Transform(carla.Location(x=radar_transform_conf[0][0],y=radar_transform_conf[0][1] ,z=radar_transform_conf[0][2]),carla.Rotation(radar_transform_conf[0][0],radar_transform_conf[0][1],radar_transform_conf[0][2]))
        radar_bp = world.get_blueprint_library().find('sensor.other.radar')
        radar_bp.set_attribute('horizontal_fov',str(carla_config['radar_settings']['horizontal_fov']))
        radar_bp.set_attribute('vertical_fov', str(carla_config['radar_settings']['vertical_fov']))
        radar_bp.set_attribute('points_per_second', str(carla_config['radar_settings']['points_per_second']))
        radar_bp.set_attribute('range', str(carla_config['radar_settings']['range']))
        radar = world.spawn_actor(radar_bp, radar_transform, attach_to=vehicle)
        actor_list.append(radar)
        print('created %s' % radar.type_id)

    if use_imu:
        imu_bp = world.get_blueprint_library().find('sensor.other.imu')
        imu = world.spawn_actor(imu_bp, camera_transform, attach_to=vehicle)
        actor_list.append(imu)
        print('created %s' % imu.type_id)

    if use_gnss:
        gnss_bp = world.get_blueprint_library().find('sensor.other.gnss')
        gnss = world.spawn_actor(gnss_bp, camera_transform, attach_to=vehicle)
        actor_list.append(gnss)
        print('created %s' % gnss.type_id)


    if carla_config['dataset_settings']['use_semantic_lidar'] and export_dataset:
        lidar_semantic_transform_conf = carla_config['dataset_settings']['semantic_lidar_transform']
        lidar_semantic_sensor = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
        lidar_semantic_transform = carla.Transform(carla.Location(x=lidar_semantic_transform_conf[0][0],y=lidar_semantic_transform_conf[0][1] ,z=lidar_semantic_transform_conf[0][2]),carla.Rotation(lidar_semantic_transform_conf[0][0],lidar_semantic_transform_conf[0][1],lidar_semantic_transform_conf[0][2]))
        lidar_semantic_sensor.set_attribute('channels', str(carla_config['dataset_settings']['semantic_lidar_channels']))
        lidar_semantic_sensor.set_attribute('range', str(carla_config['dataset_settings']['semantic_lidar_range']))
        lidar_semantic_sensor.set_attribute('upper_fov', str(carla_config['dataset_settings']['semantic_lidar_upper_fov']))
        lidar_semantic_sensor.set_attribute('points_per_second', str(carla_config['dataset_settings']['semantic_lidar_points_per_second']))
        lidar_semantic_sensor.set_attribute('lower_fov', str(carla_config['dataset_settings']['semantic_lidar_lower_fov']))
        lidar_semantic_sensor.set_attribute('rotation_frequency', str(carla_config['dataset_settings']['semantic_lidar_rotation_frequency']))
        lidar_semantic_sensor.set_attribute('horizontal_fov', str(carla_config['dataset_settings']['semantic_lidar_horizontal_fov']))
        lidar_semantic_sensor = world.spawn_actor(lidar_semantic_sensor, lidar_semantic_transform, attach_to=vehicle)
        actor_list.append(lidar_semantic_sensor)
        lidar_semantic_sensor.listen(lambda data: self.add_sensor(data, 'lidar_semantic'))
        data_length += 1


    try:

        camera.listen(lambda image: self.add_frame(image))
        camera.listen_to_gbuffer(carla.GBufferTextureID.SceneColor,
                                    lambda image: self.add_gbuffer(image, "SceneColor"))
        camera.listen_to_gbuffer(carla.GBufferTextureID.SceneDepth,
                                    lambda image: self.add_gbuffer(image, "SceneDepth"))
        camera.listen_to_gbuffer(carla.GBufferTextureID.GBufferA,
                                    lambda image: self.add_gbuffer(image, "GBufferA"))
        camera.listen_to_gbuffer(carla.GBufferTextureID.GBufferB,
                                    lambda image: self.add_gbuffer(image, "GBufferB"))
        camera.listen_to_gbuffer(carla.GBufferTextureID.GBufferC,
                                    lambda image: self.add_gbuffer(image, "GBufferC"))
        camera.listen_to_gbuffer(carla.GBufferTextureID.GBufferD,
                                    lambda image: self.add_gbuffer(image, "GBufferD"))

        camera.listen_to_gbuffer(carla.GBufferTextureID.SSAO,
                                    lambda image: self.add_gbuffer(image, "GBufferSSAO"))

        camera.listen_to_gbuffer(carla.GBufferTextureID.CustomStencil,
                                    lambda image: self.add_gbuffer(image, "CustomStencil"))

        #camera_semseg.listen(lambda image: self.add_semantic(image))

        if use_lidar:
            lidar.listen(lambda data: self.add_sensor(data,'lidar'))
        if use_radar:
            radar.listen(lambda data: self.add_sensor(data,'radar'))
        if use_gnss:
            gnss.listen(lambda data: self.add_sensor(data, 'gnss'))
        if use_imu:
            imu.listen(lambda data: self.add_sensor(data, 'imu'))

        enh_width = carla_config['ego_vehicle_settings']['camera_width']
        enh_height = carla_config['ego_vehicle_settings']['camera_height']

        renderObject = RenderObject(enh_width, enh_height)
        # Initialise the display
        pygame.init()
        gameDisplay = pygame.display.set_mode((enh_width, enh_height), pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.initialize_gt_labels(enh_width,enh_height,29)

        frame_queue = queue.Queue()
        worker_thread = threading.Thread(target=self.render_thread, args=(frame_queue, renderObject,))
        worker_thread.start()

        if async_data_transfer:
            transfer_thread = threading.Thread(target=self.host_device_thread, args=(inputs,data_length))
            transfer_thread.start()

        done_simulation = False
        self.network.eval()

        if dtype == "fp16" or dtype == 'tf32':
            forward_data_type = torch.float16
        else:
            forward_data_type = torch.float32

        if dtype == 'tf32':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        if driving_mode == "ad_model":
            autonomous_model = ADModel().to(self.device)
            autonomous_model.load()
            autonomous_model.eval()

        elif driving_mode == "rl_train":
            rl_environment = AutonomousDrivingEnvironment()
            rl_num_episodes_save = carla_config['autonomous_driving']['rl_num_episodes_save']
            current_episode = 0
            autonomous_agent = RLModel(rl_action_dim,replay_buffer_size=carla_config['autonomous_driving']['rl_buffer_max_size'])
            rl_total_reward = 0
            done = False
            state = rl_environment.reset(vehicle)
            rl_steps_counter = 0
            steps_per_episode = []
            rewards_per_episode = []
            critic_loss = []
            actor_loss = []
            vehicle_distance = []
            critic_loss_per_episode = []
            actor_loss_per_episode = []
            vehicle_distance_per_episode = []
            epsilon_per_episode = []
            epsilons = []
            best_total_reward = 0

            collision_sensor_bp = blueprint_library.find('sensor.other.collision')
            collision_sensor = world.spawn_actor(collision_sensor_bp, camera_transform, attach_to=vehicle)
            collision_sensor.listen(lambda event: self.on_collision(event, rl_environment))
            actor_list.append(collision_sensor)
        elif driving_mode == "rl_eval":
            rl_environment = AutonomousDrivingEnvironment()
            autonomous_agent = RLModel(rl_action_dim,replay_buffer_size=carla_config['autonomous_driving']['rl_buffer_max_size'])
            autonomous_agent.load(rl_model_name,rl_model_load_episode)
            autonomous_agent.eval_mode()
            state = rl_environment.reset(vehicle)

        if carla_config['general']['pygame_output'] == 'ad_task':
            ad_task_ref = ADTask()


        if found_scenario:
            rl_ego_transform = self.initialize_scenario(world,scenario_config,vehicle)
            self.stabilize_vehicle(world,spectator_camera_mode,camera,stabilize_ticks,data_length)
        elif found_scenario == False and driving_mode == 'rl_train':
            print('\033[91m'+"You need to select a valid scenario for reinforcement learning training...")
            sys.exit(0)

        while not done_simulation:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done_simulation = True

            gameDisplay.blit(renderObject.surface, (0, 0))
            pygame.display.flip()

            if spectator_camera_mode == 'follow':
                world.get_spectator().set_transform(camera.get_transform())

            if sync_mode == True or sync_mode == False:
                start_timer = time.time()

                if driving_mode == "rl_train":
                    action = autonomous_agent.select_action(state, carla_config['autonomous_driving']['rl_use_exploration'])
                    rl_environment.apply_action(action, vehicle)

                elif driving_mode == "rl_eval":
                    action = autonomous_agent.select_action(state, False)
                    rl_environment.apply_action(action, vehicle)

                if sync_mode == True:
                    if num_skip_frames > 0:
                        for i in range(num_skip_frames):
                            data_dict = {}
                            names_dict = {}
                            world.tick()
                            if spectator_camera_mode == 'follow':
                                world.get_spectator().set_transform(camera.get_transform())
                            if driving_mode == "manual":
                                self.manual_controls_apply(vehicle,manual_controls)
                            while True:
                                if len(data_dict) == data_length:
                                    break
                            data_dict = {}
                            names_dict = {}

                    world.tick()
                ticks_counter += 1
                if driving_mode == "auto":
                    vehicle.set_autopilot(True)

                if driving_mode == "manual":
                    self.manual_controls_apply(vehicle,manual_controls)

                if spectator_camera_mode == 'follow':
                    world.get_spectator().set_transform(camera.get_transform())


                if found_scenario:
                    self.trigger_scenario(world,scenario_config,vehicle)

                if sync_mode == True:
                    frame_found = False
                    gt_labels_found = False
                    gbuffers_found = False
                    frame_thread = None
                    gt_labels_thread = None
                    gbuffers_thread = None
                    while True:
                        if "color_frame" in data_dict and frame_found == False:
                            frame_found = True
                            frame_thread = threading.Thread(target=self.preprocess_worker, args=("frame",compiler,inputs,))
                            frame_thread.start()

                        if "semantic_segmentation" in data_dict and gt_labels_found == False:
                            gt_labels_found = True
                            gt_labels_thread = threading.Thread(target=self.preprocess_worker, args=("gt_labels",compiler,inputs,))
                            gt_labels_thread.start()

                        if "SceneColor" in data_dict and "SceneDepth" in data_dict and "GBufferA" in data_dict and "GBufferB" in data_dict and "GBufferC" in data_dict and "GBufferD" in data_dict and "GBufferSSAO" in data_dict and "CustomStencil" in data_dict and gbuffers_found == False:
                            gbuffers_found = True
                            gbuffers_thread = threading.Thread(target=self.preprocess_worker, args=("gbuffers",compiler,inputs,))
                            gbuffers_thread.start()
                        if len(data_dict) == data_length:
                            if gbuffers_found == False:
                                gbuffers_thread = threading.Thread(target=self.preprocess_worker, args=("gbuffers",compiler,inputs,))
                                gbuffers_thread.start()
                            if frame_found == False:
                                frame_thread = threading.Thread(target=self.preprocess_worker, args=("frame",compiler,inputs,))
                                frame_thread.start()
                            if gt_labels_found == False:
                                gt_labels_thread = threading.Thread(target=self.preprocess_worker, args=("gt_labels",compiler,inputs,))
                                gt_labels_thread.start()
                            break
                else:
                    if async_data_transfer == False:
                        frame_thread = threading.Thread(target=self.preprocess_worker,args=("frame", compiler, inputs,))
                        frame_thread.start()
                        gbuffers_thread = threading.Thread(target=self.preprocess_worker,args=("gbuffers", compiler, inputs,))
                        gbuffers_thread.start()
                        gt_labels_thread = threading.Thread(target=self.preprocess_worker,args=("gt_labels", compiler, inputs,))
                        gt_labels_thread.start()
                if async_data_transfer == False:
                    frame_thread.join()
                    gt_labels_thread.join()
                    gbuffers_thread.join()
                    img = result_container['frame']
                    label_map = result_container['gt_labels']
                    gbuffers = result_container['gbuffers']


                #print(names_dict)
                if carla_config['general']['visualize_buffers'] == True:
                    self.visualize_buffers(enh_width, enh_height)
                if len(data_dict) == data_length:

                    if carla_config['general']['run_enhanced_model'] == True:
                        if compiler == 'onnxruntime':
                            tdtype = np.float32
                            if dtype == 'fp16':
                                tdtype = np.float16
                            io_binding.bind_input(name='input', device_type=img.device_name(), device_id=0,
                                                    element_type=tdtype,
                                                    shape=img.shape(), buffer_ptr=img.data_ptr())
                            io_binding.bind_input(name='gbuffers', device_type=gbuffers.device_name(), device_id=0,
                                                    element_type=tdtype,
                                                    shape=gbuffers.shape(), buffer_ptr=gbuffers.data_ptr())
                            io_binding.bind_input(name='onnx::Gather_2', device_type=label_map.device_name(), device_id=0,
                                                    element_type=tdtype,
                                                    shape=label_map.shape(), buffer_ptr=label_map.data_ptr())
                            io_binding.bind_output('output', device_type='cuda', device_id=0, element_type=tdtype)

                            infer_timer = time.time()
                            session.run_with_iobinding(io_binding)
                            print("Inference time: " + str(time.time() - infer_timer))

                            new_img = torch.from_numpy(io_binding.copy_outputs_to_cpu()[0])
                        elif compiler == 'tensorrt':
                            infer_timer = time.time()
                            [output] = common.do_inference_v2(context, bindings=bindings, inputs=inputs , outputs=outputs, stream=stream)
                            print("Inference time: " + str(time.time() - infer_timer))
                            new_img = torch.from_numpy(output.reshape(1, 3, 540, 960))
                        else:
                            with torch.no_grad():
                                with torch.cuda.amp.autocast_mode.autocast(dtype=forward_data_type):
                                    batch = EPEBatch(img, gbuffers=gbuffers, gt_labels=label_map, robust_labels=None, path=None,coords=None).to(self.device)
                                    infer_timer = time.time()
                                    new_img = self.network.generator(batch)
                                    print("Inference time: " + str(time.time() - infer_timer))
                                    pass


                    if carla_config['general']['pygame_output'] == 'enhanced' and carla_config['general']['run_enhanced_model'] == True:
                        frame_queue.put(new_img)
                    elif (carla_config['general']['pygame_output'] == 'enhanced' and carla_config['general']['run_enhanced_model'] == False) or carla_config['general']['pygame_output'] == 'rgb':
                        renderObject.surface = pygame.surfarray.make_surface(data_dict['color_frame'].swapaxes(0, 1))
                    else:
                        if selected_camera_output == "enhanced" and carla_config['general']['run_enhanced_model'] == True:
                            img = (new_img[0, ...].clamp(min=0, max=1).permute(1, 2, 0) * 255.0).detach().cpu().numpy().astype(np.uint8)
                            ad_task_frame = ad_task_ref.predict_output(img,np.ascontiguousarray(data_dict['semantic_segmentation']),world,vehicle,camera,data_dict)
                        else:
                            img = np.ascontiguousarray(data_dict['color_frame'])
                            ad_task_frame = ad_task_ref.predict_output(img,np.ascontiguousarray(data_dict['semantic_segmentation']),world,vehicle,camera,data_dict)
                        renderObject.surface = pygame.surfarray.make_surface(ad_task_frame.swapaxes(0, 1))

                    if selected_camera_output == "enhanced" and driving_mode == "ad_model":
                        enhanced_frame = (new_img[0, ...].clamp(min=0, max=1) * 255.0).cpu().detach().numpy().astype(np.uint8)
                        enhanced_frame = np.transpose(enhanced_frame, axes=(1, 2, 0))
                        controls_predicted = autonomous_model.test_single(enhanced_frame)
                        print(controls_predicted)
                    elif selected_camera_output == "rgb" and driving_mode == "ad_model":
                        controls_predicted = autonomous_model.test_single(data_dict['color_frame'])
                        print(controls_predicted)

                    if driving_mode == "ad_model":
                        if float(controls_predicted['brake']) > carla_config['autonomous_driving']['ad_brake_threshold']:
                            controls_predicted['brake'] = 1.0
                        else:
                            controls_predicted['brake'] = 0.0
                        vehicle.apply_control(carla.VehicleControl(throttle=float(controls_predicted['throttle']),
                                                                    steer=float(controls_predicted['steering']),
                                                                    brake=float(controls_predicted['brake'])))
                        print(controls_predicted)


                    if driving_mode == "rl_train" or driving_mode == "rl_eval":
                        if selected_camera_output == "enhanced":
                            next_state = (new_img[0, ...].clamp(min=0, max=1) * 255.0).cpu().detach().numpy().astype(np.uint8)
                            next_state = np.transpose(next_state, axes=(1, 2, 0))
                            next_state = autonomous_agent.preprocess_camera_frame(next_state)
                        else:
                            rgb_frame = autonomous_agent.preprocess_camera_frame(np.ascontiguousarray(data_dict['color_frame']))
                            next_state = rgb_frame


                        if driving_mode == "rl_train":
                            reward, done,next_state, _ = rl_environment.calculate_reward(action, vehicle, world, ticks_counter, next_state, data_dict)
                            autonomous_agent.add_observation(state, action, next_state, reward, done)

                            actorloss,criticloss = autonomous_agent.train()

                            critic_loss.append(criticloss)
                            actor_loss.append(actorloss)
                            epsilons.append(autonomous_agent.epsilon)
                            vehicle_distance.append(vehicle.get_location().distance(rl_ego_transform.location))
                            rl_total_reward += reward
                            rl_steps_counter += 1
                            print("total reward:" + str(rl_total_reward))
                            if done == True:
                                critic_loss_per_episode.append(critic_loss)
                                actor_loss_per_episode.append(actor_loss)
                                vehicle_distance_per_episode.append(vehicle_distance)
                                epsilon_per_episode.append(epsilons)
                                critic_loss = []
                                actor_loss = []
                                vehicle_distance = []
                                epsilons = []
                                rewards_per_episode.append(rl_total_reward)
                                if rl_total_reward > best_total_reward:
                                    best_total_reward = rl_total_reward
                                    autonomous_agent.save(rl_model_name,"best")
                                rl_total_reward = 0
                                steps_per_episode.append(rl_steps_counter)
                                rl_steps_counter = 0
                                state = rl_environment.reset(vehicle)
                                self.initialize_scenario(world, scenario_config, vehicle)
                                self.stabilize_vehicle(world, spectator_camera_mode, camera,int(stabilize_ticks/2),data_length)
                                self.initialize_scenario(world, scenario_config, vehicle)
                                self.stabilize_vehicle(world, spectator_camera_mode, camera, int(stabilize_ticks/2),data_length)
                                current_episode+=1
                                state = next_state

                                if current_episode % rl_num_episodes_save == 0:
                                    autonomous_agent.save(rl_model_name,current_episode)
                                    self.save_rl_stats(actor_losses=actor_loss_per_episode,
                                                        critic_losses=critic_loss_per_episode,
                                                        steps=steps_per_episode, rewards=rewards_per_episode,
                                                        current_step=current_episode,
                                                        distances=vehicle_distance_per_episode,episode=current_episode,epsilon=epsilon_per_episode)
                            else:
                                state = next_state
                        elif driving_mode == "rl_eval":
                            state = next_state
                    if (driving_mode == "rl_eval" or driving_mode == "ad_model") and val_ticks is not None: #driving_mode == "rl_eval" or driving_mode == "ad_model"
                        if val_ticks < ticks_counter:
                            sys.exit(0)

                else:
                    print("--Unsync--")
                    print(names_dict)
                    print("-------")

                    # Update PyGame window
                    gameDisplay.fill((0, 0, 0))
                    gameDisplay.blit(renderObject.surface, (0, 0))
                    pygame.display.flip()

                if export_dataset and carla_config['general']['run_enhanced_model'] == True and driving_mode != 'rl_train':
                    if carla_config['dataset_settings']['capture_when_static'] == True or (carla_config['dataset_settings']['capture_when_static']==False and self.is_vehicle_moving(vehicle,carla_config['dataset_settings']['speed_threshold'])):
                        random_id = self.random_id_generator()
                        if carla_config['dataset_settings']['export_status_json']:
                            self.save_vehicle_status(vehicle, random_id + str(names_dict['color_frame']))
                            self.save_world_status(selected_town, selected_weather_preset, selected_vehicle_name,
                                                    selected_perspective, sync_mode,
                                                    random_id + str(names_dict['color_frame']))
                        self.save_frames(random_id + str(names_dict['color_frame']), new_img, carla_config)

                        if carla_config['dataset_settings']['export_object_annotations']:
                            self.save_object_detection_annotations(camera,world,vehicle,random_id + str(names_dict['color_frame']),carla_config)
                elif export_dataset and carla_config['general']['run_enhanced_model'] == False:
                    print("Warning: To export frames you need to enable run_enhanced_model from the carla_config file.")
                elif export_dataset and driving_mode == 'rl_train':
                    print("Warning: To export frames you need to be in evaluation or auto drive mode.")

                if sync_mode == True:
                    names_dict = {}
                    data_dict = {}

                print("Executin Delay: " + str(time.time() - start_timer))


    except Exception as error: #(KeyboardInterrupt,SystemExit,subprocess.CalledProcessError)
        print(error)
        settings.synchronous_mode = False
        world.apply_settings(settings)
        if compiler == 'tensorrt':
            common.free_buffers(inputs, outputs, stream)
        print('Destroying actors...')
        camera.destroy()
        #camera_semseg.destroy()
        if driving_mode == "rl_train":
            collision_sensor.destroy()
        vehicle.destroy()
        if other_actor is not None:
            other_actor.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('Done...')
        frame_queue.put(None)
        print('Terminating render thread...')
        print('Exiting...')
        sys.exit()