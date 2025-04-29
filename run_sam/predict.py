import yaml
import torch
import argparse
from inference_ft import inference_main
import os
import sys
import datetime

# Get the parent directory of the current script
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_directory)
sys.stdout.flush()

import models
from Data_Preparation_SAM.DataProcessing import upscale_testing_data_cubic_final


device = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else ("mps" if torch.mps.is_available() else "cpu")
)
# python run_sam/predict.py --version dataset_name/1024/20250327_1719 --path examples/test.tif --use-epoch 12


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        type=str,
        help='combination of dataset name, size, upsample and train timestamp, e.g. "dataset_name/1024/20250327_1719"',
    )
    parser.add_argument("--path", type=str, help="path to image to predict on")
    parser.add_argument(
        "--use-epoch",
        type=int,
        help="use model state from selected epoch to generate prediction",
    )
    parser.add_argument(
        "--threshold",
        default=0.5,
        type=float,
        help="0 to 1 as threshold for binarizing prediction",
    )
    parser.add_argument(
        "--upscale-test", default=False, action="store_true", help="upscale test image by cubic, no need to do that for drone image"
    )
    args = parser.parse_args()

    trained_version = args.version
    test_img_path = args.path
    selected_epoch = args.use_epoch
    thres = args.threshold
    upscale = args.upscale_test

    src_path = f"{parent_directory}/outputs/{trained_version}"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    path_out_pred = f"{src_path}/predicted/{timestamp}"
    config_path = f"{src_path}/config.yaml"
    os.makedirs(path_out_pred, exist_ok=True)

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if upscale:
        upscale_testing_data_cubic_final(test_file_path=test_img_path)
        test_img_path = test_img_path.replace(".tif", "_upscaled.tif")

    model = models.make(config["model"]).to(device)
    saved_path = f"save_model/{trained_version}/model_epoch{selected_epoch}.pth"
    checkpoint = torch.load(saved_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        inference_main(
            model, config, thres, path_out_pred, upsample="1024", img_path=test_img_path
        )


if __name__ == "__main__":
    main()
