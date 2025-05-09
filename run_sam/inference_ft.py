import sys
import os
import pickle
# Get the parent directory of the current script
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to the Python path
sys.path.append(parent_directory)
import numpy as np
import cv2
import scipy
from skimage.transform import resize
from skimage import io
from tqdm import tqdm
import itertools

import torch
from torchvision import transforms
from torchvision.transforms import functional as F


from samgeo.common import *
import rasterio
import matplotlib.pyplot as plt

device = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else ("mps" if torch.mps.is_available() else "cpu")
)
if torch.cuda.is_available():
    torch.distributed.init_process_group(backend="nccl")


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, ann=None):
        if ann is None:
            for t in self.transforms:
                image = t(image)
            return image
        for t in self.transforms:
            image, ann = t(image, ann)
        return image, ann


class Resize(object):
    def __init__(self, image_height, image_width, ann_height, ann_width):
        self.image_height = image_height
        self.image_width = image_width
        self.ann_height = ann_height
        self.ann_width = ann_width

    def __call__(self, image, ann):
        image = resize(image, (self.image_height, self.image_width))
        image = np.array(image, dtype=np.float32) / 255.0

        sx = self.ann_width / ann["width"]
        sy = self.ann_height / ann["height"]
        ann["junc_ori"] = ann["junctions"].copy()
        ann["junctions"][:, 0] = np.clip(
            ann["junctions"][:, 0] * sx, 0, self.ann_width - 1e-4
        )
        ann["junctions"][:, 1] = np.clip(
            ann["junctions"][:, 1] * sy, 0, self.ann_height - 1e-4
        )
        ann["width"] = self.ann_width
        ann["height"] = self.ann_height
        ann["mask_ori"] = ann["mask"].copy()
        ann["mask"] = cv2.resize(
            ann["mask"].astype(np.uint8), (int(self.ann_width), int(self.ann_height))
        )

        return image, ann


class ResizeImage(object):
    def __init__(self, image_height, image_width):
        self.image_height = image_height
        self.image_width = image_width

    def __call__(self, image, ann=None):
        image = resize(image, (self.image_height, self.image_width))
        image = np.array(image, dtype=np.float32) / 255.0
        if ann is None:
            return image
        return image, ann


class ToTensor(object):
    def __call__(self, image, anns=None):
        if anns is None:
            return F.to_tensor(image)

        for key, val in anns.items():
            if isinstance(val, np.ndarray):
                anns[key] = torch.from_numpy(val)
        return F.to_tensor(image), anns


def inference_image(image, model):
    print("start inferencing image ... ")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    # h_stride, w_stride = 600, 600

    h_stride, w_stride = 800, 800  # bigger the stride, fewer moves need to be done (make big for high resolution)
    h_crop, w_crop = 1024, 1024
    h_img, w_img, _ = image.shape
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    pred_whole_img = np.zeros([h_img, w_img], dtype=np.float32)
    count_mat = np.zeros([h_img, w_img])

    patch_weight = np.ones((h_crop + 2, w_crop + 2))
    patch_weight[0, :] = 0
    patch_weight[-1, :] = 0
    patch_weight[:, 0] = 0
    patch_weight[:, -1] = 0

    patch_weight = scipy.ndimage.distance_transform_edt(patch_weight)
    patch_weight = patch_weight[1:-1, 1:-1]

    print(f"h grids: {h_grids}, w grids: {w_grids}")
    total_grid = h_grids * w_grids
    for h_idx, w_idx in tqdm(itertools.product(range(h_grids), range(w_grids)), total=total_grid, miniters=total_grid//50):
        y1 = h_idx * h_stride
        x1 = w_idx * w_stride
        y2 = min(y1 + h_crop, h_img)
        x2 = min(x1 + w_crop, w_img)
        y1 = max(y2 - h_crop, 0)
        x1 = max(x2 - w_crop, 0)

        crop_img = image[y1:y2, x1:x2, :]
        crop_img = crop_img.astype(np.float32)
        crop_img_tensor = transform(crop_img)[None].to(device)

        with torch.no_grad():
            mask_pred = model.infer(crop_img_tensor)
            mask_pred = torch.sigmoid(mask_pred)
            mask_pred = mask_pred.detach().cpu().numpy()[0, 0]
        mask_pred *= patch_weight
        pred_whole_img += np.pad(
            mask_pred,
            (
                (int(y1), int(pred_whole_img.shape[0] - y2)),
                (int(x1), int(pred_whole_img.shape[1] - x2)),
            ),
        )
        count_mat[y1:y2, x1:x2] += patch_weight
    pred_whole_img = pred_whole_img / count_mat

    return pred_whole_img


def get_binary_mask(pred_mask, thres):

    binar_mask = pred_mask > thres

    return binar_mask


def save_fig(data, data_name, path_out):

    # create figure
    plt.figure(figsize=(50, 50))
    plt.axis("off")

    # save data
    path_out_ = os.path.join(path_out, data_name + ".png")
    plt.imshow(data)
    plt.savefig(path_out_, bbox_inches="tight")


def tiff_to_shp(tiff_path, output, simplify_tolerance=0, **kwargs):
    """Convert a tiff file to a shapefile.
    Args:
        tiff_path (str): The path to the tiff file.
        output (str): The path to the shapefile.
        simplify_tolerance (float, optional): The maximum allowed geometry displacement.
            The higher this value, the smaller the number of vertices in the resulting geometry.
    """
    raster_to_shp(tiff_path, output, simplify_tolerance=simplify_tolerance, **kwargs)


def save_predicted_probability_mask_shapefile(
    pred_whole_img, thres, path_out, upsample, img_path
):

    # read spatial information from test image
    with rasterio.open(img_path) as src:
        ras_meta = src.profile
        # ras_meta["crs"] = src.crs  # for vector
        ras_meta["count"] = 1  # for raster
        ras_meta["dtype"] = "float32"
        ras_meta["compress"] = "lzw"
        ras_meta["photometric"] = "MINISBLACK"

    pred_whole_img_prob = np.expand_dims(pred_whole_img[...], axis=0)
    pred_whole_img_bin = pred_whole_img_prob > thres

    # save probability as png
    save_fig(pred_whole_img, "mask_prob", path_out)
    # save binary dense mask as png
    save_fig(pred_whole_img > thres, "mask_binary", path_out)

    # save probability map
    if upsample != "SR":
        path_mask_out = path_out + "/pred_mask_prob.tif"
        with rasterio.open(path_mask_out, "w", **ras_meta) as dst:
            dst.write(pred_whole_img_prob)

    # save predicted binary mask
    path_mask_out_bin = path_out + "/pred_mask_bin" + str(thres) + ".tif"
    with rasterio.open(path_mask_out_bin, "w", **ras_meta) as dst:
        dst.write(pred_whole_img_bin)

    # save polygons as shapefile if the polgyons are not None.
    num_positive = np.sum(pred_whole_img_bin)
    num_total = pred_whole_img_bin.shape[1] * pred_whole_img_bin.shape[2]

    if num_positive > num_total * 1e-4:
        path_shp_out = path_out + "/pred_mask_bin" + str(thres) + ".shp"
        tiff_to_shp(path_mask_out_bin, path_shp_out)


# main function
def inference_main(model, config, thres, path_out, upsample, img_path=""):
    print("read in test image ... ")
    # read image for testing (a large geotiff data)
    if not img_path:
        img_path = config["test_dataset"]["dataset"]["args"]["root_path_1"]
    image = io.imread(img_path)
    print(f"image size {image.shape}")
    if image.shape[2] > 3:
        image = image[:, :, :3]
    image = (image - image.min()) / (image.max() - image.min())

    # probability map
    prob_mask = inference_image(image, model)
    with open(f"{path_out}/prob_mask.pkl", "wb") as f:
        pickle.dump(prob_mask, f)

    print("Predicted probability map.")

    # save probability map, binary mask, shapefile
    save_predicted_probability_mask_shapefile(
        prob_mask, thres, path_out, upsample, img_path
    )
    print(
        "Save predicted probability map, binary map with threshold at {} and shapefile to {}.".format(
            thres, path_out
        )
    )
