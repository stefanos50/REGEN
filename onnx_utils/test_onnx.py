import argparse
import cv2
import numpy as np
import onnxruntime as ort


print("ORT:", ort.__version__)
print("Providers:", ort.get_available_providers())


def preprocess(image_path, height, width):
    image = cv2.imread(image_path)

    if image is None:
        raise RuntimeError(f"Cannot load image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_LINEAR
    )

    image = image.astype(np.float32) / 255.0

    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    image = (image - mean) / std

    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    return np.ascontiguousarray(image)


def postprocess(output_tensor):
    image_numpy = output_tensor[0]

    image_numpy = (
        (np.transpose(image_numpy, (1, 2, 0)) + 1)
        / 2.0
        * 255.0
    )

    image_numpy = np.clip(image_numpy, 0, 255)
    image_numpy = image_numpy.astype(np.uint8)

    image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)

    return image_numpy


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--onnx", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--output", default="output.png")

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Session
    # ------------------------------------------------------------
    session = ort.InferenceSession(
        args.onnx,
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ]
    )

    print("Active providers:", session.get_providers())

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name


    input_tensor = preprocess(
        args.image,
        args.height,
        args.width
    )

    io_binding = session.io_binding()

    providers = session.get_providers()

    use_cuda = "CUDAExecutionProvider" in providers

    if use_cuda:
        input_ort = ort.OrtValue.ortvalue_from_numpy(
            input_tensor,
            "cuda",
            0
        )

        io_binding.bind_ortvalue_input(
            input_name,
            input_ort
        )

        io_binding.bind_output(
            output_name,
            "cuda",
            0
        )

        session.run_with_iobinding(io_binding)

        outputs = io_binding.copy_outputs_to_cpu()

    else:
        # CPU fallback
        outputs = session.run(
            [output_name],
            {input_name: input_tensor}
        )


    output_tensor = outputs[0]

    output_image = postprocess(output_tensor)

    cv2.imwrite(args.output, output_image)

    print(f"Saved output image to: {args.output}")


if __name__ == "__main__":
    main()