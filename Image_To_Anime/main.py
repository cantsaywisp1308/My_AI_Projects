import sys
import time
import os
from pathlib import Path
import urllib.request

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov
from IPython.display import HTML, display

try:
    import paddle
    from paddle.static import InputSpec
    from ppgan.apps import AnimeGANPredictor
except NameError:
    if sys.platform == "win32":
        install_message = ("To use this notebook, please install the free Microsoft "
            "Visual C++ redistributable from <a href='https://aka.ms/vs/16/release/vc_redist.x64.exe'>"
            "https://aka.ms/vs/16/release/vc_redist.x64.exe</a>")
    else:
        install_message = (
            "To use this notebook, please install a C++ compiler. On macOS, "
            "`xcode-select --install` installs many developer tools, including C++. On Linux, "
            "install gcc with your distribution's package manager."
        )
        display(
            HTML(
                f"""<div class="alert alert-danger" ><i>
            <b>Error: </b>PaddlePaddle requires installation of C++. {install_message}"""
            )
        )
        raise

MODEL_DIR = "model"
MODEL_NAME = "paddlegan_anime"

os.makedirs(MODEL_DIR, exist_ok=True)
# Create filenames of the models that will be converted in this notebook.
model_path = Path(f"{MODEL_DIR}/{MODEL_NAME}")
ir_path = model_path.with_suffix(".xml")
onnx_path = model_path.with_suffix(".onnx")

def resize_to_max_wodth(image, max_width):
    if image.shape[1] > max_width:
        hw_ratio = image.shape[0] / image.shape[1]
        new_height = int(max_width * hw_ratio)
        image = cv2.resize(image, (max_width, new_height))
    return image


# This cell will initialize the AnimeGANPredictor() and download the weights from PaddlePaddle.
# This may take a while. The weights are stored in a cache and are downloaded once.
predictor = AnimeGANPredictor()

PADDLEGAN_INFERENCE = True
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)
#Step1: Load the image and convert to RGB
image_path = Path('pics/messi.jpg')
#urllib.request.urlretrieve('https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_bricks.png', image_path)
image = cv2.cvtColor(cv2.imread(str(image_path), flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

## Inference takes a long time on large images. Resize to a max width of 600.
image= resize_to_max_wodth(image,600)

# Step 2. Transform the image.
transformed_image = predictor.transform(image)
input_tensor = paddle.to_tensor(transformed_image[None,::])

if PADDLEGAN_INFERENCE:
    # Step 3. Do inference.
    predictor.generator.eval()
    with paddle.no_grad():
        result = predictor.generator(input_tensor)
        # Step 4. Convert the inference result to an image, following the same steps as
        # PaddleGAN's predictor.run() function.
    result_image_pg = (result *0.5 + 0.5)[0].numpy() * 255
    result_image_pg = result_image_pg.transpose((1,2,0))
    # Step 5. Resize the result image.
    result_image_pg = cv2.resize(result_image_pg, image.shape[:2][::-1])
    # Step 6. Adjust the brightness.
    result_image_pg = predictor.adjust_brightness(result_image_pg, image)
    # Step 7. Save the result image.
    anime_image_path_pg = Path(f"{OUTPUT_DIR}/{image_path.stem}_anime_pg").with_suffix(".jpg")
    if cv2.imwrite(str(anime_image_path_pg), result_image_pg[:, :, (2,1,0)]):
        print(f"The anime image was saved to {anime_image_path_pg}")

#Show Inference Results on PaddleGAN model
if PADDLEGAN_INFERENCE:
    fig, ax = plt.subplots(1,2, figsize=(25,15))
    ax[0].imshow(image)
    ax[1].imshow(result_image_pg)
else:
    print(
        "PADDLEGAN_INFERENCE is not enabled. Set PADDLEGAN_INFERENCE = True in the previous cell and run that cell to show inference results.")



#onnx is only compatible with python under 3.8
# target_height, target_width = transformed_image.shape[1:]
# predictor.generator.eval()
# x_spec = InputSpec([None, 3, target_height, target_width], "float32", "x")
# paddle.onnx.export(predictor.generator, str(model_path), input_spec=[x_spec], opset_version=11)
#
# # predictor.__init__()
# # t = predictor.transform.transforms[0]
#
# print("Exporting ONNX model to OpenVINO IR... This may take a few minutes.")
# model = ov.convert_model(onnx_path, input=[1,3,target_height,target_width],)
# ov.save_model(model, str(ir_path))
# predictor.adjust_brightness(image)
# predictor.calc_avg_brightness(image)
#
# def calc_avg_brightness(img):
#     R = img[...,0].mean()
#     G = img[...,1].mean()
#     B = img[...,2].mean()
#     brightness = 0.299* R + 0.587 * G + 0.114 * B
#     return brightness, B, G ,R
#
# def adjust_brightness(dst, src):
#     brightness1, B1, G1, R1 = AnimeGANPredictor.calc_avg_brightness(src)
#     brightness2, B2, G2, R2 = AnimeGANPredictor.calc_avg_brightness(dst)
#     brightness_difference = brightness1 / brightness2
#     dstf = dst * brightness_difference
#     dstf = np.clip(dstf, 0, 255)
#     dstf = np.uint8(dstf)
#     return dstf
#
# import ipywidgets as widgets
# core = ov.Core()
# device = widgets.Dropdown(
#     options= core.available_devices + ["AUTO"],
#     value='AUTO',
#     description = 'Device',
#     disabled = False,
# )
