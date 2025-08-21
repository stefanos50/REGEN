import cv2
import numpy as np
import torch
from options.test_options import TestOptions
from models.models import create_model
import util.util as util
import mss
import win32gui

opt = TestOptions().parse(save=False)
opt.nThreads = 1
opt.batchSize = 1
opt.serial_batches = True
opt.no_flip = True

model = create_model(opt)
if opt.data_type == 16:
    model.half()
elif opt.data_type == 8:
    model.type(torch.uint8)

def get_window_bbox(window_name):
    hwnd = win32gui.FindWindow(None, window_name)
    if not hwnd:
        raise Exception(f"Window not found: {window_name}")
    rect = win32gui.GetWindowRect(hwnd)
    left, top, right, bottom = rect
    return {"top": top, "left": left, "width": right - left, "height": bottom - top}

def enum_window_titles():
    titles = []

    def callback(hwnd, extra):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:  # skip empty titles
                titles.append(title)
    win32gui.EnumWindows(callback, None)
    return titles

windows = enum_window_titles()
for w in windows:
    print(w)
#exit(1)

import numpy as np

def crop_image(img: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:
    return img[top: img.shape[0] - bottom, left: img.shape[1] - right]


def main():
    window_name = "Grand Theft Auto V" 
    monitor = get_window_bbox(window_name)
    sct = mss.mss()

    try:
        while True:
            img = np.array(sct.grab(monitor))
            img = crop_image(img, 120,70,120,120)

            rgb_array = img[:, :, :3][:, :, ::-1]  # BGRA → RGB
            rgb_array_copy = rgb_array.copy()

            image_np = torch.from_numpy(rgb_array_copy).permute(2, 0, 1).unsqueeze(0).float()

            generated = model.inference(image_np, image_np, image_np)
            generated = util.tensor2im(generated.data[0])

            bgr_array = generated[:, :, ::-1]  # RGB → BGR for OpenCV


            cv2.imshow("Enhanced Window", bgr_array)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                break

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()