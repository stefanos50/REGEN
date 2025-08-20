#python carla_test.py --dataroot ./data --name epe2 --gpu_id 0 --label_nc 0 --no_instance

import carla
import random
import time
import numpy as np
import cv2
import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
import torch

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0      
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)


if not opt.engine and not opt.onnx:
    model = create_model(opt)
    if opt.data_type == 16:
        model.half()
    elif opt.data_type == 8:
        model.type(torch.uint8)
            
    if opt.verbose:
        print(model)
else:
    from run_engine import run_trt_engine, run_onnx

def main():
    # Connect to the CARLA server
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.load_world("Town10HD")
    #world = client.get_world()

    # Enable synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / 20.0  # 20 FPS
    world.apply_settings(settings)

    # Set up traffic manager
    tm = client.get_trafficmanager()
    tm_port = tm.get_port()
    tm.set_synchronous_mode(True)

    blueprint_library = world.get_blueprint_library()

    # Spawn vehicle
    #vehicle_bp = random.choice(blueprint_library.filter("vehicle.*"))
    vehicle_bp = blueprint_library.find("vehicle.tesla.model3")
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("No spawn points available.")
        return

    spawn_point = random.choice(spawn_points)
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(True, tm_port)

    # Spawn camera
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", "960")
    camera_bp.set_attribute("image_size_y", "540")
    camera_bp.set_attribute("fov", "90")

    camera_transform = carla.Transform(carla.Location(x=1.5, z=1.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # Shared variable for latest frame
    latest_image = {"frame": None}

    # Camera callback just stores frame
    def save_image(image):
        latest_image["frame"] = image

    camera.listen(save_image)

    try:
        while True:
            # Step simulation forward
            world.tick()

            # Ensure autopilot stays on
            vehicle.set_autopilot(True, tm_port)

            # Process latest frame if available
            if latest_image["frame"] is not None:
                image = latest_image["frame"]

                # Convert CARLA raw image to RGB NumPy
                array = np.frombuffer(image.raw_data, dtype=np.uint8)
                array = array.reshape((image.height, image.width, 4))
                rgb_array = array[:, :, :3][:, :, ::-1]  # BGRA → RGB
                rgb_array_copy = rgb_array.copy()

                # Convert to torch tensor
                image_np = torch.from_numpy(rgb_array_copy).permute(2, 0, 1).unsqueeze(0).float()

                # Run model
                generated = model.inference(image_np, image_np, image_np)
                generated = util.tensor2im(generated.data[0])

                # Show result
                bgr_array = generated[:, :, ::-1]  # RGB → BGR for OpenCV
                cv2.imshow("CARLA Enhanced Camera", bgr_array)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                    break

    except KeyboardInterrupt:
        print("Exiting...")

    finally:
        camera.stop()
        vehicle.destroy()
        camera.destroy()

        # Restore async mode
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        tm.set_synchronous_mode(False)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
