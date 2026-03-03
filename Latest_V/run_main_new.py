import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from models import (
    Autoformer,
    DLinear,
    TimeLLM,
    LongitudinalLLM,
    LongitudinalLLM_nacc,
    LongitudinalLLM_nacc1
)

from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os
import pandas as pd
from datetime import datetime


os.environ["CURL_CA_BUNDLE"] = ""
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64" # deprecated
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import (
    del_files,
    EarlyStopping,
    adjust_learning_rate,
    vali,
    load_content,
)

parser = argparse.ArgumentParser(description="Time-LLM")

# --- Alzheimer-specific options ---
parser.add_argument(
    "--feature_names",
    type=str,
    default=None,
    help="Comma-sep or Python list of time-varying feature names",
)
parser.add_argument(
    "--static_feature_names",
    type=str,
    default=None,
    help="Comma-sep or Python list of static feature names",
)
parser.add_argument(
    "--visit_order",
    type=str,
    default=None,
    help="Comma-sep or Python list of visits in chronological order",
)
parser.add_argument(
    "--mark_options",
    type=str,
    default=None,
    help="Python dict or JSON string of mark_options",
)
parser.add_argument(
    "--add_missing_mask",
    type=int,
    default=1,
    help="Whether to append missingness mask as a feature",
)
parser.add_argument(
    "--impute_value",
    type=float,
    default=0.0,
    help="Value to impute missing entries with (or leave NaN if None)",
)
parser.add_argument(
    "--static_in_prompt",
    type=int,
    default=1,
    help="If 1, static demographics go only into the prompt; if 0, appended to features",
)


# basic config
parser.add_argument(
    "--task_name",
    type=str,
    required=True,
    default="long_term_forecast",
    help="task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]",
)
parser.add_argument("--is_training", type=int, required=True, default=1, help="status")
parser.add_argument(
    "--model_id", type=str, required=True, default="test", help="model id"
)
parser.add_argument(
    "--model_comment",
    type=str,
    required=True,
    default="none",
    help="prefix when saving test results",
)
parser.add_argument(
    "--model",
    type=str,
    required=True,
    default="Autoformer",
    help="model name, options: [Autoformer, DLinear]",
)
parser.add_argument("--seed", type=int, default=2025, help="random seed")

# data loader
parser.add_argument(
    "--data", type=str, default="Alzheimer", help='Must be "Alzheimer" for this script'
)
parser.add_argument(
    "--data_path", type=str, required=True, help="Path to the flattened Alzheimer CSV"
)
parser.add_argument(
    "--features",
    type=str,
    default="M",
    help="forecasting task, options:[M, S, MS]; "
    "M:multivariate predict multivariate, S: univariate predict univariate, "
    "MS:multivariate predict univariate",
)
parser.add_argument(
    "--target", type=str, default="CDRSB", help="target feature in S or MS task"
)
parser.add_argument("--loader", type=str, default="modal", help="dataset type")
parser.add_argument(
    "--checkpoints",
    type=str,
    default="./checkpoints/",
    help="location of model checkpoints",
)

# forecasting task
parser.add_argument("--seq_len",
                    type=int, 
                    default=5, 
                    help="input sequence length")
parser.add_argument("--label_len", 
                    type=int, 
                    default=2, 
                    help="start token length")
parser.add_argument(
    "--pred_len", 
    type=int, 
    default=2, 
    help="prediction sequence length"
)

# model define
parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
parser.add_argument("--c_out", type=int, default=7, help="output size")
parser.add_argument("--d_model", type=int, default=32, help="dimension of model")
parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
parser.add_argument("--d_ff", type=int, default=64, help="dimension of fcn")
parser.add_argument(
    "--moving_avg", type=int, default=2, help="window size of moving average"
)
parser.add_argument("--factor", type=int, default=1, help="attn factor")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
parser.add_argument(
    "--embed",
    type=str,
    default="timeF",
    help="time features encoding, options:[timeF, fixed, learned]",
)
parser.add_argument("--activation", type=str, default="gelu", help="activation")
parser.add_argument(
    "--output_attention",
    action="store_true",
    help="whether to output attention in encoder",
)
parser.add_argument("--patch_len", type=int, default=2, help="patch length")
parser.add_argument("--stride", type=int, default=1, help="stride")
parser.add_argument("--prompt_domain", type=int, default=0, help="")
parser.add_argument(
    "--llm_model", type=str, default="LLAMA", help="LLM model"
)  # LLAMA, GPT2, BERT
parser.add_argument(
    "--llm_dim", type=int, default="4096", help="LLM model dimension"
)  # LLama7b:4096; GPT2-small:768; BERT-base:768


# optimization
parser.add_argument(
    "--num_workers", type=int, default=10, help="data loader num workers"
)
parser.add_argument("--itr", type=int, default=1, help="experiments times")
parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
parser.add_argument("--align_epochs", type=int, default=10, help="alignment epochs")
parser.add_argument(
    "--batch_size", type=int, default=32, help="batch size of train input data"
)
parser.add_argument(
    "--eval_batch_size", type=int, default=32, help="batch size of model evaluation"
)
parser.add_argument("--patience", type=int, default=10, help="early stopping patience")
parser.add_argument(
    "--learning_rate", type=float, default=0.001, help="optimizer learning rate"
)
parser.add_argument("--des", type=str, default="test", help="exp description")
parser.add_argument("--loss", type=str, default="MSE", help="loss function")
parser.add_argument("--lradj", type=str, default="type1", help="adjust learning rate")
parser.add_argument("--pct_start", type=float, default=0.2, help="pct_start")
parser.add_argument(
    "--use_amp",
    action="store_true",
    help="use automatic mixed precision training",
    default=False,
)
parser.add_argument("--llm_layers", type=int, default=6)


#############################################################
# Added: training size and validation size as input arguments

parser.add_argument(
    "-ts", "--training_size",
    type = float,
    default = 0.1,
    help = "fraction of data samples used for training"
)
parser.add_argument(
    "-vs", "--validation_size",
    type = float,
    default = 0.0,
    help = "fraction of samples used for validation"
)
#############################################################


args = parser.parse_args()
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config="./ds_config_zero2.json")
accelerator = Accelerator(
    kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin
)

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)


def masked_mse_loss(output, target, mask):
    """
    output: Tensor of shape [B, seq, n_feat]
    target: Tensor of shape [B, seq, n_feat]
    mask:   Tensor of shape [B, seq, n_feat] (1 if observed, 0 if missing)
    Returns:
        Scalar tensor (masked MSE averaged over observed values only)
    """
    mse = (output - target) ** 2  # elementwise MSE
    masked_mse = mse * mask  # zero out missing
    loss = masked_mse.sum() / (mask.sum() + 1e-8)
    return loss


def masked_mae_loss(output, target, mask):
    """
    output, target, mask: same as above
    Returns:
        Scalar tensor (masked MAE averaged over observed values only)
    """
    mae = torch.abs(output - target)
    masked_mae = mae * mask
    loss = masked_mae.sum() / (mask.sum() + 1e-8)
    return loss


#############################################################
# Added:
# new metric
#### fix it for missing and zero true values
def masked_mare_loss(output, target, mask):
    """
    Mean Absolute Relative Error

    output, target, mask: same as above
    Returns:
        Scalar tensor (masked MAE averaged over observed values only)
    """
    if target == 0:
        mare = torch.abs( (output - target) / (target + 1e-8) )
    else:
        mare = torch.abs( (output - target) / target)
    masked_mare = mare * mask
    loss = masked_mare.sum() / (mask.sum() + 1e-8)
    return loss
#############################################################


results = []

for ii in range(args.itr):
    # setting record of experiments
    setting = "{}_{}_{}_{}_ts{}_vs{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}".format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.training_size,
        args.validation_size,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.des,
        ii,
    )

    full_dataset, _ = data_provider(args, flag=None)  # flag ignored for Alzheimer
    num_subjects = len(full_dataset)

    # train_size = int(0.1 * num_subjects)
    # val_size = int(0.2 * num_subjects)
    train_size = int(args.training_size * num_subjects)
    val_size = int(args.validation_size * num_subjects)

    test_size = num_subjects - train_size - val_size

    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set, test_set = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )
    vali_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    args.content = load_content(args)

    if args.model == "Autoformer":
        model = Autoformer.Model(args).float()
    elif args.model == "DLinear":
        model = DLinear.Model(args).float()
    elif args.model == "TimeLLM":
        model = TimeLLM.Model(args).float()
    else:
        model = LongitudinalLLM_nacc.Model(args).float()

    path = os.path.join(
        args.checkpoints, setting + "-" + args.model_comment
    )  # unique checkpoint saving path
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)

    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    if args.lradj == "COS":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            model_optim, T_max=20, eta_min=1e-8
        )
    else:
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=args.pct_start,
            epochs=args.train_epochs,
            max_lr=args.learning_rate,
        )

    train_loader, vali_loader, test_loader, model, model_optim, scheduler = (
        accelerator.prepare(
            train_loader, vali_loader, test_loader, model, model_optim, scheduler
        )
    )

    if args.use_amp:
        # scaler = torch.cuda.amp.GradScaler()
        scaler = torch.amp.GradScaler('cuda')

    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        try:
            for i, batch in enumerate(tqdm(train_loader, desc="Training")):
                batch_x, batch_y, batch_x_mark, batch_y_mark, *rest = batch

                mask_seq_x = mask_seq_y = statics = None
                if args.static_in_prompt and len(rest) == 3:
                    mask_seq_x, mask_seq_y, statics = rest
                    mask_seq_x = mask_seq_x.float().to(accelerator.device)
                    mask_seq_y = mask_seq_y.float().to(accelerator.device)
                elif not args.static_in_prompt and len(rest) == 2:
                    mask_seq_x, mask_seq_y = rest
                    mask_seq_x = mask_seq_x.float().to(accelerator.device)
                    mask_seq_y = mask_seq_y.float().to(accelerator.device)

                # --- Assert shapes match ---
                if mask_seq_x is not None:
                    assert (
                        mask_seq_x.shape == batch_x.shape
                    ), f"mask_seq_x.shape {mask_seq_x.shape} != batch_x.shape {batch_x.shape}"
                if mask_seq_y is not None:
                    assert (
                        mask_seq_y.shape == batch_y.shape
                    ), f"mask_seq_y.shape {mask_seq_y.shape} != batch_y.shape {batch_y.shape}"

                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(accelerator.device)
                batch_y = batch_y.float().to(accelerator.device)
                batch_x_mark = batch_x_mark.float().to(accelerator.device)
                batch_y_mark = batch_y_mark.float().to(accelerator.device)

                # decoder input
                dec_inp = (
                    torch.zeros_like(batch_y[:, -args.pred_len :, :])
                    .float()
                    .to(accelerator.device)
                )
                dec_inp = (
                    torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(accelerator.device)
                )

                # encoder - decoder
                if args.use_amp:
                    # with torch.cuda.amp.autocast():
                    with torch.amp.autocast('cuda'):
                        if args.output_attention:
                            outputs = model(
                                batch_x,
                                batch_x_mark,
                                dec_inp,
                                batch_y_mark,
                                mask_seq_x,
                                mask_seq_y,
                                statics,
                            )[0]
                        else:
                            outputs = model(
                                batch_x,
                                batch_x_mark,
                                dec_inp,
                                batch_y_mark,
                                mask_seq_x,
                                mask_seq_y,
                                statics,
                            )

                        f_dim = -1 if args.features == "MS" else 0
                        outputs = outputs[:, -args.pred_len :, f_dim:]
                        batch_y = batch_y[:, -args.pred_len :, f_dim:].to(
                            accelerator.device
                        )
                        mask_seq_y = mask_seq_y[:, -args.pred_len :, f_dim:].to(
                            accelerator.device
                        )
                        loss = masked_mse_loss(outputs, batch_y, mask_seq_y)
                        train_loss.append(loss.item())
                else:
                    if args.output_attention:
                        outputs = model(
                            batch_x,
                            batch_x_mark,
                            dec_inp,
                            batch_y_mark,
                            mask_seq_x,
                            mask_seq_y,
                            statics,
                        )[0]
                    else:
                        outputs = model(
                            batch_x,
                            batch_x_mark,
                            dec_inp,
                            batch_y_mark,
                            mask_seq_x,
                            mask_seq_y,
                            statics,
                        )

                    f_dim = -1 if args.features == "MS" else 0
                    outputs = outputs[:, -args.pred_len :, f_dim:]
                    batch_y = batch_y[:, -args.pred_len :, f_dim:]
                    mask_seq_y = mask_seq_y[:, -args.pred_len :, f_dim:].to(
                        accelerator.device
                    )
                    loss = masked_mse_loss(outputs, batch_y, mask_seq_y)
                    train_loss.append(loss.item())

                if (i + 1) % 50 == 0:
                    accelerator.print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    accelerator.print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                if args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    accelerator.backward(loss)
                    model_optim.step()

                if args.lradj == "TST":
                    adjust_learning_rate(
                        accelerator,
                        model_optim,
                        scheduler,
                        epoch + 1,
                        args,
                        printout=False,
                    )
                    scheduler.step()
        except Exception as e:
            # print(f"Exception at index {batch}: {e}")
            raise

        accelerator.print(
            "Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time)
        )
        train_loss = np.average(train_loss)
        vali_loss, vali_mae_loss, _ = vali(
            args,
            accelerator,
            model,
            val_set,
            vali_loader,
            masked_mse_loss,
            masked_mae_loss,
            desc="Validation",
        )
        test_loss, test_mae_loss, _ = vali(
            args,
            accelerator,
            model,
            test_set,
            test_loader,
            masked_mse_loss,
            masked_mae_loss,
            desc="Testing",
        )
        accelerator.print(
            "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} MAE Loss: {4:.7f}".format(
                epoch + 1, train_loss, vali_loss, test_loss, test_mae_loss
            )
        )

        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            break

        if args.lradj != "TST":
            if args.lradj == "COS":
                scheduler.step()
                accelerator.print(
                    "lr = {:.10f}".format(model_optim.param_groups[0]["lr"])
                )
            else:
                if epoch == 0:
                    args.learning_rate = model_optim.param_groups[0]["lr"]
                    accelerator.print(
                        "lr = {:.10f}".format(model_optim.param_groups[0]["lr"])
                    )
                adjust_learning_rate(
                    accelerator, model_optim, scheduler, epoch + 1, args, printout=True
                )

        else:
            accelerator.print(
                "Updating learning rate to {}".format(scheduler.get_last_lr()[0])
            )

    # After training and early stopping:

    # 1. Reload best checkpoint
    if args.model == "Autoformer":
        best_model = Autoformer.Model(args).float()
    elif args.model == "DLinear":
        best_model = DLinear.Model(args).float()
    elif args.model == "TimeLLM":
        best_model = TimeLLM.Model(args).float()
    else:
        best_model = LongitudinalLLM_nacc.Model(args).float()

    checkpoint_path = os.path.join(
        args.checkpoints, setting + "-" + args.model_comment, "checkpoint"
    )
    best_model.load_state_dict(
        torch.load(checkpoint_path, map_location=accelerator.device)
    )
    best_model = best_model.to(dtype=torch.bfloat16)
    best_model.to(accelerator.device)
    best_model.eval()

    # 2. Evaluate test MAE
    test_loss, test_mae_loss, test_per_var_mae = vali(
        args,
        accelerator,
        best_model,
        test_set,
        test_loader,
        masked_mse_loss,
        masked_mae_loss,
        desc="Testing (Best Checkpoint)",
    )

    accelerator.print(f"Trial {ii}: Test MAE (best checkpoint): {test_mae_loss:.6f}")

    # 3. Save results for CSV
    run_result = vars(args).copy()  # save all current settings
    run_result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_result["trial"] = ii
    run_result["test_mae"] = float(test_mae_loss)

    # If feature_names is a string, parse it
    if isinstance(args.feature_names, str):
        feature_names = [x.strip() for x in args.feature_names.split(",")]
    else:
        feature_names = list(args.feature_names)
    for i, mae in enumerate(test_per_var_mae):
        var_name = feature_names[i] if i < len(feature_names) else f"var_{i}"
        run_result[f"test_mae_{var_name}"] = mae

    results.append(run_result)

accelerator.wait_for_everyone()
if accelerator.is_local_main_process:
    # 1. Create DataFrame from trial results
    df = pd.DataFrame(results)
    # Desired order: timestamp, test_mae, test_mae_{var1}, ...
    cols = ["timestamp", "test_mae"] + [f"test_mae_{name}" for name in feature_names]

    # If your DataFrame has more columns (hyperparams, etc.), you can add them after:
    remaining_cols = [c for c in df.columns if c not in cols]
    df = df[cols + remaining_cols]  # reorder columns

    # 2. Set up variable names for columns
    # per_var_cols = [
    #     f"test_mae_{feature_names[i]}" for i, mae in enumerate(feature_names)
    # ]

    # all_mae_cols = ["test_mae"] + per_var_cols

    # 3. Compute mean and std for only those columns
    # avg_row = {col: np.nan for col in df.columns}
    # std_row = {col: np.nan for col in df.columns}
    # for col in all_mae_cols:
    #     if col in df.columns:
    #         avg_row[col] = df[col].astype(float).mean()
    #         std_row[col] = df[col].astype(float).std()

    # 4. Label the summary rows in the first column (e.g., 'trial' or whatever your first column is)
    # label_col = df.columns[0]
    # avg_row[label_col] = "AVG"
    # std_row[label_col] = "STD"

    # 5. Append summary rows and save to CSV with timestamp
    # df = df.append([avg_row, std_row], ignore_index=True)
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # csv_path = f"results_{args.model_id}_{timestamp}.csv"
    csv_path = f"results_{args.model_id}.csv"
    # Check if the file already exists
    if os.path.exists(csv_path):
        # Read existing CSV
        old_df = pd.read_csv(csv_path)
        # Append new rows
        combined_df = pd.concat([old_df, df], ignore_index=True)
    else:
        combined_df = df
    combined_df.to_csv(csv_path, index=False)
    print(f"Saved all trial results and per-variable stats to {csv_path}.")

    path = "./checkpoints"  # unique checkpoint saving path
    del_files(path)  # delete checkpoint files
    accelerator.print("success delete checkpoints")
