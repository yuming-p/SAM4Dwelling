import sys
import os

# Get the parent directory of the current script
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to the Python path
sys.path.append(parent_directory)

import warnings

"""the version of mmcv used here is 1.7. current updated version is 2.0"""
warnings.filterwarnings("ignore", category=DeprecationWarning, module="mmcv")
import datetime
import argparse
import yaml
import pathlib
import glob
import shutil

import datasets
import models
from helpers import utils
from statistics import mean

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import OneCycleLR
import torch
from inference_ft import inference_main


if torch.cuda.is_available():
    torch.distributed.init_process_group(backend="nccl")


def write_yaml(data_yaml, path_out):
    """A function to write YAML file"""

    path_out_yaml = os.path.join(path_out, "config.yaml")
    with open(path_out_yaml, "w") as f:
        yaml.dump(data_yaml, f)


def make_data_loader(spec, tag=""):
    if spec is None:
        return None

    dataset = datasets.make(spec["dataset"])
    dataset = datasets.make(spec["wrapper"], args={"dataset": dataset})

    log("{} dataset: size={}".format(tag, len(dataset)))
    for k, v in dataset[-1].items():
        log("  {}: shape={}".format(k, tuple(v.shape)))
    loader = DataLoader(
        dataset,
        batch_size=spec["batch_size"],
        shuffle=False,
        sampler=None,
        pin_memory=True,
    )

    return loader


def make_data_loaders(config):
    train_loader = make_data_loader(config.get("train_dataset"), tag="train")
    val_loader = make_data_loader(config.get("val_dataset"), tag="val")
    return train_loader, val_loader


def eval_psnr(loader, model, eval_type=None):
    model.eval()

    if eval_type == "f1":
        metric_fn = utils.calc_f1
        metric1, metric2, metric3, metric4 = "f1", "auc", "none", "none"
    elif eval_type == "fmeasure":
        metric_fn = utils.calc_fmeasure
        metric1, metric2, metric3, metric4 = "f_mea", "mae", "none", "none"
    elif eval_type == "ber":
        metric_fn = utils.calc_ber
        metric1, metric2, metric3, metric4 = "shadow", "non_shadow", "ber", "none"
    elif eval_type == "cod":
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = "sm", "em", "wfm", "mae"

    pbar = tqdm(total=len(loader), leave=True, desc="val")

    pred_list = []
    gt_list = []
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.mps.is_available() else "cpu"
    )
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                batch[k] = v.to(device)

            inp = batch["inp"]
            pred = torch.sigmoid(model.infer(inp))

            pred_list.append(pred.detach().cpu())
            gt_list.append(batch["gt"].detach().cpu())

            pbar.update(1)
            pbar.refresh()

    pbar.close()

    pred_list = torch.cat(pred_list, 1)
    gt_list = torch.cat(gt_list, 1)
    result1, result2, result3, result4 = metric_fn(pred_list, gt_list)

    return result1, result2, result3, result4, metric1, metric2, metric3, metric4


def prepare_training(config):
    if torch.cuda.is_available():
        model = models.make(config["model"]).cuda()
    elif torch.mps.is_available():
        model = models.make(config["model"]).to("mps")
    optimizer = utils.make_optimizer(model.parameters(), config["optimizer"])

    epoch_start = config.get("resume") + 1 if config.get("resume") is not None else 1
    print(f"epoch start: {epoch_start}")
    log("model: #params={}".format(utils.compute_num_params(model, text=True)))

    return model, optimizer, epoch_start


def train(train_loader, model, lr_scheduler):
    model.train()
    model.to("mps")

    pbar = tqdm(total=len(train_loader), leave=False, desc="train")
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.mps.is_available() else "cpu"
    )
    loss_list = []
    for batch in train_loader:
        for k, v in batch.items():
            v = v.to(device)
            batch[k] = v
        inp = batch["inp"]
        gt = batch["gt"]
        print(f"Model is on device: {next(model.parameters()).device}")
        model.set_input(inp, gt)
        model.optimize_parameters()
        lr_scheduler.step()

        loss_list.append(model.loss_G.clone())
        # batch_loss = [torch.zeros_like(model.loss_G) for _ in range(dist.get_world_size())]
        # dist.all_gather(batch_loss, model.loss_G)
        # loss_list.extend(batch_loss)
        pbar.update(1)
    pbar.close()

    loss = [i.item() for i in loss_list]
    return mean(loss)


def save(config, model, epoch, optimizer, loss, save_path):
    name = "epoch" + str(epoch)
    if (
        config["model"]["name"] == "segformer" or config["model"]["name"] == "setr"
    ) and config["model"]["args"]["encoder_mode"]["name"] == "evp":
        prompt_generator = model.encoder.backbone.prompt_generator.state_dict()
        decode_head = model.encoder.decode_head.state_dict()
        torch.save(
            {"prompt": prompt_generator, "decode_head": decode_head},
            os.path.join(save_path, f"prompt_epoch_{name}.pth"),
        )
    else:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            os.path.join(save_path, f"model_{name}.pth"),
        )


def delete_checkpoint_folder_datafolder(path_data):

    img_folder_path = os.path.join(path_data, "images", ".ipynb_checkpoints")
    gt_folder_path = os.path.join(path_data, "gt", ".ipynb_checkpoints")

    if os.path.exists(img_folder_path):
        shutil.rmtree(img_folder_path)
        print("ipynb_checkpoints folder of images deleted successfully.")
    elif os.path.exists(gt_folder_path):
        shutil.rmtree(gt_folder_path)
        print("ipynb_checkpoints folder deleted successfully.")
    else:
        print("ipynb_checkpoints folder not found.")


def main(config, path_output, path_model, save_last_n_epoch, thres, upsample):

    global log, writer, log_info
    log, writer = utils.set_save_path(path_output, remove=False)

    train_loader, val_loader = make_data_loaders(config)
    if config.get("data_norm") is None:
        config["data_norm"] = {
            "inp": {"sub": [0], "div": [1]},
            "gt": {"sub": [0], "div": [1]},
        }
    epoch_max = config["epoch_max"]
    model, optimizer, epoch_start = prepare_training(config)

    model.optimizer = optimizer
    # lr_scheduler = CosineAnnealingLR(
    #     model.optimizer, config["epoch_max"], eta_min=config.get("lr_min")
    # )

    lr_scheduler = OneCycleLR(
                optimizer,
                max_lr=0.0001,
                steps_per_epoch=len(train_loader),
                epochs=epoch_max,
            )

    sam_checkpoint = torch.load(config["sam_checkpoint"])
    model.load_state_dict(sam_checkpoint, strict=False)

    for name, para in model.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)

    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        "model_grad_params:" + str(model_grad_params),
        "\nmodel_total_params:" + str(model_total_params),
    )

    model.train()
    if torch.cuda.is_available():
        model.cuda()
    elif torch.mps.is_available():
        model.to("mps")
    print(f"Model is on device: {next(model.parameters()).device}")

    epoch_max = config["epoch_max"]
    max_val_v = -1e18 if config["eval_type"] != "ber" else 1e-8

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):

        t_epoch_start = timer.t()
        train_loss_G = train(train_loader, model, lr_scheduler)

        log_info = ["epoch {}/{}".format(epoch, epoch_max)]
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        log_info.append("train G: loss={:.4f}".format(train_loss_G))
        writer.add_scalars("loss", {"train G": train_loss_G}, epoch)

        model_spec = config["model"]
        model_spec["sd"] = model.state_dict()
        optimizer_spec = config["optimizer"]
        optimizer_spec["sd"] = optimizer.state_dict()

        # save the trained model
        if (epoch_max - epoch) <= save_last_n_epoch:
            save(config, model, epoch, optimizer, train_loss_G, path_model)

        # validation
        result1, result2, result3, result4, metric1, metric2, metric3, metric4 = (
            eval_psnr(val_loader, model, eval_type=config.get("eval_type"))
        )
        # f1, fmeasure, cod metrics all range from 0 to 1, bigger the better
        log_info.append("val: {}={:.4f}".format(metric1, result1))
        writer.add_scalars(metric1, {"val": result1}, epoch)
        log_info.append("val: {}={:.4f}".format(metric2, result2))
        writer.add_scalars(metric2, {"val": result2}, epoch)
        log_info.append("val: {}={:.4f}".format(metric3, result3))
        writer.add_scalars(metric3, {"val": result3}, epoch)
        log_info.append("val: {}={:.4f}".format(metric4, result4))
        writer.add_scalars(metric4, {"val": result4}, epoch)

        if config["eval_type"] != "ber":
            if result1 > max_val_v:
                max_val_v = result1
                print(f"max val v: {max_val_v}")
                save(config, model, epoch, optimizer, train_loss_G, path_model)
        else:
            if result3 < max_val_v:
                max_val_v = result3
                save(config, model, epoch, optimizer, train_loss_G, path_model)

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append("{} {}/{}".format(t_epoch, t_elapsed, t_all))

        log(", ".join(log_info))
        writer.flush()

    # inference - testing data
    path_output_pred = os.path.join(path_output, "eval_test")
    pathlib.Path(path_output_pred).mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        inference_main(model, config, thres, path_output_pred, upsample)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/config_sam_vit_h.yaml",
        help="use the hyperparameters provided by SAM-Adapter",
    )
    parser.add_argument("--data", default=None, help="different datasets")
    parser.add_argument("--upsample", default="1024", help="1024 or SR")
    parser.add_argument("--epoch", default=13, type=int, help="epochs for training")
    parser.add_argument(
        "--save-last-epoch",
        default=5,
        help="save model state, optimizer and scheduler for last n epochs.",
    )
    parser.add_argument(
        "--thres", default=0.5, help="the threshold to determine the binary map"
    )
    path_data = f"{parent_directory}/data"

    args = parser.parse_args()
    path_cfg = args.config
    data_name = args.data
    upsample = args.upsample
    save_last_n_epoch = args.save_last_epoch
    thres = args.thres
    epoch = args.epoch
    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else ("mps" if torch.mps.is_available() else "cpu")
    )

    # open the base config file to read hyperparameters
    with open(path_cfg, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # number of epochs for training
    config["epoch_max"] = epoch

    # training data
    path_train_data = os.path.join(path_data, data_name, "SAM", upsample, "train")
    delete_checkpoint_folder_datafolder(path_train_data)

    config["train_dataset"]["dataset"]["args"]["root_path_1"] = os.path.join(
        path_train_data, "images"
    )
    config["train_dataset"]["dataset"]["args"]["root_path_2"] = os.path.join(
        path_train_data, "gt"
    )

    # validation data - to save time for training, we skip validation process
    path_val_data = os.path.join(path_data, data_name, "SAM", upsample, "val")
    config["val_dataset"]["dataset"]["args"]["root_path_1"] = os.path.join(
        path_val_data, "images"
    )
    config["val_dataset"]["dataset"]["args"]["root_path_2"] = os.path.join(
        path_val_data, "gt"
    )

    # testing data
    if upsample == "1024":
        path_test_data = os.path.join(path_data, data_name, "SAM", upsample, "test")
        print(path_test_data)
        print(glob.glob(os.path.join(path_test_data, "images") + "/*.tif"))
        path_img = glob.glob(os.path.join(path_test_data, "images") + "/*.tif")[0]

    elif upsample == "SR":
        path_test_data = os.path.join(path_data, data_name, "SAM", upsample, "test")
        path_img = glob.glob(os.path.join(path_test_data) + "/*.tif")[0]

    path_gt = glob.glob(os.path.join(path_test_data, "gt") + "/*.tif")[0]

    config["test_dataset"]["dataset"]["args"]["root_path_1"] = path_img
    config["test_dataset"]["dataset"]["args"]["root_path_2"] = path_gt

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    # defin path of outputs of predicted results and models to be saved
    path_output = os.path.join("outputs", data_name, upsample, timestamp)
    path_model = os.path.join("save_model", data_name, upsample, timestamp)

    pathlib.Path(path_output).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_model).mkdir(parents=True, exist_ok=True)

    write_yaml(config, path_output)
    print("config saved.")

    all_paths_to_clean = [path_train_data, path_val_data, path_test_data]
    all_paths_to_clean = [p + "/images/.DS_Store" for p in all_paths_to_clean] + [
        p + "/gt/.DS_Store" for p in all_paths_to_clean
    ]
    for p in all_paths_to_clean:
        try:
            print(f"removing DS store file from path {p}...")
            os.remove(p)
        except FileNotFoundError:
            continue
    main(config, path_output, path_model, save_last_n_epoch, thres, upsample)
