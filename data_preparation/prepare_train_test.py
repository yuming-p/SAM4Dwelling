
"""
- make sure train and test aerial image is converted to EPSG:4326 already, if not:
gdalwarp -t_srs EPSG:4326 {input_img_path} {output_img_path}

- input image needs to RGB and type unit8

# Data preparation

- This notebook includes data preprocessing steps for [SAM-Adapter](https://github.com/tianrun-chen/SAM-Adapter-PyTorch).
- This workflow is only suitable for ***binary segmentation***. Feel free to adapt it for multiclass segmentation.
- You can upscale images (4 times) by a super resolution model ([EDSR](https://github.com/aswintechguy/Deep-Learning-Projects/tree/main/Super%20Resolution%20-%20OpenCV)) by OpenCV.
- The default structure and format of your input datasets are:<br>
Here we aim to convert a large geotiff image/label data into small patches for deep learning models.<br>

- **Data Structure:** <br>

    Dataset1<br>
    - raw
        - train
            - images  (geotiff, uint8, 3 bands (RGB), you can create and enhance image data in GIS software in advance)
            - gt      (geotiff, uint8, value:0(background), 255(targets)(not necessary to have to be 255 if it is a binary segmentation but have to be distinctive from background))
        - test
            - images
            - gt
    
    Dataset2<br>
        ... ...<br>

"""
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


import geopandas as gpd
import rasterio.mask
import argparse
import os
import rasterio
from rasterio.features import rasterize
from DataProcessing import data_process_sam_seg_final, upscale_data_by_cubic_final


def get_gt_bbox(dataset_name, gt_path, buffer_factor=0.002):
    # get buffered bounding box of ground truth shapefile
    gt = gpd.read_file(gt_path)
    gt = gt.to_crs("EPSG:4326")
    bbox = gpd.GeoDataFrame([gt.union_all().envelope], columns=["geometry"], crs=gt.crs)
    logger.info(f"bounding box: {bbox}")
    bbox_buffered = bbox.copy()
    bbox_buffered.geometry = bbox_buffered.geometry.apply(
        lambda x: x.buffer(distance=x.length * buffer_factor, join_style="mitre")
    )
    gpd.GeoDataFrame(bbox_buffered).to_file(
        f"./output/bbox_{dataset_name}.geojson", driver="GeoJSON"
    )
    return gt, bbox_buffered


def format_and_clip_aerial_img(dataset_name, shapes, img_path, data_type):
    with rasterio.open(img_path) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        out_image = out_image[:3, :, :]
        out_meta = src.meta
        out_meta.update({"count": 3})
    logger.info(f"resolution of image: {src.res}")
    logger.info(f"cropped image meta: {out_meta}")
    assert out_meta["dtype"] == "uint8", "data type of input raster should be unit8!"
    out_meta.update(
        {
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        }
    )
    # Save the masked image and remove band 4
    output_path = f"../data/{dataset_name}/raw/{data_type}/images/{dataset_name}.tif"
    os.makedirs(f"../data/{dataset_name}/raw/{data_type}/images/", exist_ok=True)
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(out_image)
    logger.info(f"clipped image saved to path {output_path} .")
    return out_meta


# generate ground truth raster from shapefile NHAG, that matches the size of corresponding satellite image
def generate_gt_raster(gt_shp, total_bounds, output_tiff_path, width, height):
    minx, miny, maxx, maxy = total_bounds
    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)
    rasterized_gt = rasterize(
        [(geom, 255) for geom in gt_shp.geometry],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    logger.info(rasterized_gt.shape)
    os.makedirs(os.path.dirname(output_tiff_path), exist_ok=True)
    with rasterio.open(
        output_tiff_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=rasterized_gt.dtype,
        crs=gt_shp.crs,
        transform=transform,
    ) as dst:
        dst.write(rasterized_gt, 1)
    logger.info(f"ground truth raster saved to {output_tiff_path} ! ")


def prepare_gt_img(dataset_name, gt_path, img_path, data_type):
    settlement_gt, bbox_buffered_settlement = get_gt_bbox(
        dataset_name=dataset_name, gt_path=gt_path
    )
    shapes_settlement = list(bbox_buffered_settlement.geometry)
    out_meta_settlement = format_and_clip_aerial_img(
        dataset_name=dataset_name,
        shapes=shapes_settlement,
        img_path=img_path,
        data_type=data_type
    )
    logger.info("rasterize ground truth shapefile ...")
    output_tiff_path_settlement = (
        f"../data/{dataset_name}/raw/{data_type}/gt/{dataset_name}_gt.tif"
    )
    total_bounds_settlement = bbox_buffered_settlement.total_bounds
    generate_gt_raster(
        settlement_gt,
        total_bounds_settlement,
        output_tiff_path_settlement,
        width=out_meta_settlement["width"],
        height=out_meta_settlement["height"],
    )


def generate_upsacled_patches():
    pass

# python prepare_train_test.py --img-path --gt-path --dataname 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img-path",
        help="aerial image path to be clipped by ground truth boundary and preprocessed",
    )
    parser.add_argument(
        "--gt-path", help="path to ground truth shapefile to be rasterized"
    )
    parser.add_argument(
        "--is-test", default=False, action="store_true", help="input being test data, only upscaling no generating patches"
    )
    parser.add_argument("--dataname", type=str, help="dataset names")
    parser.add_argument("--upsample", default="", help="cubic, SR to be implemented, if emtpy, no upsample")
    parser.add_argument("--patch-size", type=int, default=256, help="first split img into patches of {patch_size} then upscale to 1024x1024 for SAM input")

    args = parser.parse_args()
    logger.info(args)
    img_path = args.img_path
    gt_path = args.gt_path
    is_test = args.is_test
    dataset_name = args.dataname
    upsample = args.upsample
    patch_size = args.patch_size
    data_type = "test" if is_test else "train"

    prepare_gt_img(dataset_name, gt_path, img_path, data_type)
    path_database = "put_your_path_to_database_here"
    data_list = [dataset_name]
    if not is_test:
        logger.info("segmenting image into patches ...")
        data_process_sam_seg_final(path_database, dataset_name, data_list, data_type, patch_size)
    if upsample == "cubic":
        logger.info(f"upscaling {data_list} patches to 1024x1024 ... ")
        upscale_data_by_cubic_final(path_database, data_list, [data_type])
