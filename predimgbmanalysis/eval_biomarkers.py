import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib as mpl


import seaborn as sns

import numpy as np

import torch
from torch.utils.data import DataLoader
import math

from predimgbmanalysis.get_toydata import dataset_dict
from predimgbmanalysis.train import ToyModelImgModule
from predimgbmanalysis.utils import update_nested_dict

import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy
from sklearn.model_selection import KFold


import yaml

import tqdm.notebook as tq

import pytorch_lightning as pl

import binsreg


def binscatter(**kwargs):
    # Estimate binsreg (scatter-binning)
    est = binsreg.binsreg(**kwargs)
    # print(est)
    # Retrieve estimates
    df_est = pd.concat([d.dots for d in est.data_plot])
    # print(df_est)

    df_est = df_est.rename(columns={"x": kwargs.get("x"), "fit": kwargs.get("y")})

    # Add confidence intervals
    if "ci" in kwargs:
        df_est = pd.merge(df_est, pd.concat([d.ci for d in est.data_plot]))
        df_est = df_est.drop(columns=["x"])
        df_est["ci"] = df_est["ci_r"] - df_est["ci_l"]

    # Rename groups
    if "by" in kwargs:
        df_est["group"] = df_est["group"].astype(df_est[kwargs.get("by")].dtype)
        df_est = df_est.rename(columns={"group": kwargs.get("by")})

    return df_est


def get_experiment_results(
    experiment_dir,
    fit_type,
    final_activation=None,
    model_name=None,
    save_df_results=False,
    get_saliency_maps=None,
    get_reference_fit_results=False,
    get_progreference_fit_results=False,
    get_predictions=False,
    env="test",
    k_fold_splits=None,
    fold_idx=None,
    **kwargs,
):
    suffix = "" if env == "test" else env
    df_results_filename = os.path.join(experiment_dir, f"df_results{suffix}.pkl")
    skip_evaluation = os.path.isfile(df_results_filename)
    saliency_suffix = get_saliency_maps if isinstance(get_saliency_maps, str) else ""
    if get_saliency_maps is not None and get_saliency_maps is not False:
        skip_evaluation = skip_evaluation & os.path.isfile(
            os.path.join(experiment_dir, f"saliency_maps{saliency_suffix}{suffix}.pt")
        )
    if get_reference_fit_results:
        skip_evaluation = skip_evaluation & os.path.isfile(
            os.path.join(experiment_dir, f"df_results_reference{suffix}.pkl")
        )
    if get_progreference_fit_results:
        skip_evaluation = skip_evaluation & os.path.isfile(
            os.path.join(experiment_dir, f"df_results_prognostic_reference{suffix}.pkl")
        )
    if get_predictions:
        skip_evaluation = skip_evaluation & os.path.isfile(
            os.path.join(experiment_dir, f"data_pred{suffix}.pt")
        )
    if skip_evaluation:
        print(
            f"Results already exist in {experiment_dir}. Loading from file {df_results_filename} instead."
        )
        # df_results = pd.read_csv(os.path.join(experiment_dir,"df_results.txt"))
        df_results = pd.read_pickle(
            os.path.join(experiment_dir, f"df_results{suffix}.pkl")
        )
        if get_saliency_maps is not None and get_saliency_maps is not False:
            saliency_maps = torch.load(
                os.path.join(
                    experiment_dir, f"saliency_maps{saliency_suffix}{suffix}.pt"
                )
            )
            df_results = (
                df_results,
                saliency_maps,
            )
        if get_reference_fit_results:
            df_results_ref = pd.read_pickle(
                os.path.join(experiment_dir, f"df_results_reference{suffix}.pkl")
            )
            if not isinstance(df_results, tuple):
                df_results = (df_results,)
            df_results = df_results + (df_results_ref,)
        if get_progreference_fit_results:
            df_progresults_ref = pd.read_pickle(
                os.path.join(
                    experiment_dir, f"df_results_prognostic_reference{suffix}.pkl"
                )
            )
            if not isinstance(df_results, tuple):
                df_results = (df_results,)
            df_results = df_results + (df_progresults_ref,)
        if get_predictions:
            data_pred = pd.read_pickle(
                os.path.join(experiment_dir, f"data_pred{suffix}.pt")
            )
            if not isinstance(df_results, tuple):
                df_results = (df_results,)
            df_results = df_results + (data_pred,)
    else:
        outputs = []
        saliency_maps = []
        outputs_ref = []
        outputs_progref = []
        outputs_data_pred = []
        for file in tq.tqdm(os.listdir(experiment_dir), position=0, leave=True):
            filename = os.fsdecode(file)
            if not os.path.isdir(
                os.path.join(
                    experiment_dir,
                    filename,
                    "lightning_logs",
                    "version_0",
                    "checkpoints",
                )
            ):
                continue
            boolean_flag = any(
                fname.endswith(".ckpt")
                for fname in os.listdir(
                    os.path.join(
                        experiment_dir,
                        filename,
                        "lightning_logs",
                        "version_0",
                        "checkpoints",
                    )
                )
            )

            # process all files if final_activation is not specified
            if final_activation is not None:
                boolean_flag = boolean_flag & filename.endswith(final_activation)
            if model_name is not None:
                boolean_flag = boolean_flag & (model_name in filename)
            if boolean_flag:
                output = get_predictions_fit_results(
                    experiment_dir,
                    log_name=filename,
                    # load_params=True,
                    fit_type=fit_type,
                    # final_activation=final_activation_,
                    get_saliency_maps=get_saliency_maps,
                    get_reference_fit_results=get_reference_fit_results,
                    get_progreference_fit_results=get_progreference_fit_results,
                    get_predictions=get_predictions,
                    env=env,
                    k_fold_splits=k_fold_splits,
                    fold_idx=fold_idx,
                    **kwargs,
                )
                if get_predictions:
                    if isinstance(output, tuple) or isinstance(output, list):
                        *output, data_pred = output
                        outputs_data_pred.append(data_pred)
                    else:
                        print("Error: data_pred not found")
                if get_progreference_fit_results:
                    if isinstance(output, tuple) or isinstance(output, list):
                        *output, output_progref = output
                        outputs_progref.append(output_progref)
                    else:
                        print("Error: output_progref not found")
                if get_reference_fit_results:
                    if isinstance(output, tuple) or isinstance(output, list):
                        *output, output_ref = output
                        outputs_ref.append(output_ref)
                    else:
                        print("Error: output_ref not found")
                if get_saliency_maps is not None and get_saliency_maps is not False:
                    output, saliency_map = output
                    saliency_maps.append(saliency_map)
                if isinstance(output, list):
                    output = output[0]
                outputs.append(output)

            else:
                continue

        newoutput = [value for value in outputs if type(value) != int]
        df_results = pd.DataFrame(newoutput)
        if save_df_results:
            df_results.to_pickle(
                os.path.join(experiment_dir, f"df_results{suffix}.pkl")
            )
            if get_saliency_maps is not None and get_saliency_maps is not False:
                torch.save(
                    saliency_maps,
                    os.path.join(
                        experiment_dir, f"saliency_maps{saliency_suffix}{suffix}.pt"
                    ),
                )
                df_results = (df_results, saliency_maps)
            if get_reference_fit_results:
                df_results_ref = pd.DataFrame(outputs_ref)
                df_results_ref.to_pickle(
                    os.path.join(experiment_dir, f"df_results_reference{suffix}.pkl")
                )
                if not isinstance(df_results, tuple):
                    df_results = (df_results,)
                df_results = df_results + (df_results_ref,)
            if get_progreference_fit_results:
                df_results_progref = pd.DataFrame(outputs_progref)
                df_results_progref.to_pickle(
                    os.path.join(
                        experiment_dir, f"df_results_prognostic_reference{suffix}.pkl"
                    )
                )
                if not isinstance(df_results, tuple):
                    df_results = (df_results,)
                df_results = df_results + (df_results_progref,)
            if get_predictions:
                data_pred = pd.DataFrame(outputs_data_pred)
                data_pred.to_pickle(
                    os.path.join(experiment_dir, f"data_pred{suffix}.pt")
                )
                if not isinstance(df_results, tuple):
                    df_results = (df_results,)
                df_results = df_results + (data_pred,)

    return df_results


def get_predictions_fit_results(
    experiment_dir,
    log_name,
    fit_type="linear",
    formula="y ~ z + treat + z * treat",
    get_input=False,
    b=None,
    env="test",
    data_model_type=None,
    n_batch=None,
    print_stats=False,
    LR_test=False,
    use_cuda=False,
    method=None,
    get_saliency_maps=None,
    dataset_root="/absolute/path/to/datasets",
    num_workers=0,
    pin_memory=False,
    get_reference_fit_results=False,
    get_progreference_fit_results=False,
    additional_key_path_to_save=None,  # key path or list of key paths separated by "." to save from the checkpoint
    custom_params=None,
    get_predictions=False,
    k_fold_splits=None,
    fold_idx=None,
    checkpoint_num=-1,
):
    log_dir = os.path.join(experiment_dir, log_name)

    with open(
        os.path.join(
            log_dir,
            "lightning_logs",
            "version_0",
            "hparams.yaml",
        ),
        "r",
    ) as stream:
        params = yaml.load(stream, Loader=yaml.CLoader)
        if isinstance(params, list):
            params = params[-1]
        if b is None:
            b = params["data_params"]["b"]

        data_model_type = params["data_params"]["data_model_type"]

    params["data_params"]["save_num_data_dir"] = None
    params["data_params"]["root"] = dataset_root
    params["data_params"]["get_treatment"] = True

    if custom_params is not None:
        params = update_nested_dict(params, custom_params)

    pred_feature = params["data_params"]["pred_feature"]
    prog_feature = params["data_params"]["prog_feature"]

    if k_fold_splits is None:
        k_fold_splits = params.get("k_fold_splits", None)
    if fold_idx is None:
        fold_idx = params.get("fold_idx", None)

    n_batch = params["loader_params"]["batch_size"] if n_batch is None else n_batch

    # load model
    checkpoint_filename = [
        os.path.join("lightning_logs", "version_0", "checkpoints", fname)
        for fname in os.listdir(
            os.path.join(log_dir, "lightning_logs", "version_0", "checkpoints")
        )
        if fname.endswith(".ckpt")
    ]
    if len(checkpoint_filename) > 1:
        # sort checkpoints by epoch (checkpoint filename format: max_epochs-epoch=399-step=9600.ckpt)
        checkpoint_filename = sorted(
            checkpoint_filename,
            key=lambda x: int(x.split("-")[1].split("=")[1].split(".")[0]),
        )
        print(
            f"Warning: More than one checkpoint file found {checkpoint_filename}. Using {checkpoint_filename[checkpoint_num]} (No. {checkpoint_num})."
        )
    checkpoint = torch.load(os.path.join(log_dir, checkpoint_filename[checkpoint_num]))
    is_mtl_model = checkpoint["hyper_parameters"]["model_type"].endswith("mtl") or (
        "mtl" in checkpoint["hyper_parameters"]["model_type"]
        or checkpoint["hyper_parameters"]["model_type"].endswith("tarnet")
        or checkpoint["hyper_parameters"]["model_type"].endswith("learner")
    )
    model = ToyModelImgModule.load_from_checkpoint(
        os.path.join(log_dir, checkpoint_filename[0])
    )

    model.eval()

    if k_fold_splits is not None and "traincv" not in env:
        print(
            f"Model trained on k-fold {fold_idx}, using {env} set for evaluation instead"
        )
    dataset = dataset_dict[checkpoint["hyper_parameters"]["data_type"]](
        env="traincv"
        if (k_fold_splits is not None and env.endswith("val") and "traincv" in env)
        else env,
        train=False,
        load_1d_covariates=True,
        get_counterfactuals=get_reference_fit_results,
        **params["data_params"],
    )
    dataset_len = len(dataset)
    if k_fold_splits is not None:
        kfold = KFold(n_splits=k_fold_splits, shuffle=True, random_state=42)
        train_indices, val_indices = list(kfold.split(np.arange(dataset_len)))[fold_idx]
        indices = val_indices if env.endswith("val") else train_indices

    else:
        indices = None

    dl = DataLoader(
        dataset=torch.utils.data.Subset(dataset, indices)
        if indices is not None
        else dataset,
        batch_size=n_batch,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    predictions = [[], []] if is_mtl_model else []
    y_list = []
    T_list = []

    if use_cuda:
        device = torch.device(
            "cuda" if (torch.cuda.is_available() and use_cuda) else "cpu"
        )
        model.to(device)

    for input, targets in dl:
        with torch.no_grad():
            y = targets[:, 0]
            T = targets[:, 1].long()
            if use_cuda:
                input = input.to(device)
        out = model(input)
        if is_mtl_model:
            predictions[0].append(out[0].cpu().detach())
            predictions[1].append(out[1].cpu().detach())
        else:
            predictions.append(out.cpu().detach())
        y_list.append(y)
        T_list.append(T)

    # concat "patches" if using minibatches
    out = (
        [torch.cat(predictions[0]), torch.cat(predictions[1])]
        if is_mtl_model
        else torch.cat(predictions)
    )
    y = torch.cat(y_list)
    T = torch.cat(T_list)

    mse = torch.nn.MSELoss(reduction="none")

    if is_mtl_model:
        # treatment effect
        z_pred = out[1].numpy() - out[0].numpy()
        out = torch.cat(out, 1)
        estimator_mse = mse(out[range(out.shape[0]), T], y).mean().numpy()
        y_pred = out[range(out.shape[0]), T].numpy()

        y_counterf = dataset.y_counterf
        if k_fold_splits is not None:
            y_counterf = y_counterf[indices]
        cate_gt = (y - y_counterf) * (2 * T - 1.0)
        pehe = torch.sqrt(
            mse(torch.from_numpy(z_pred.squeeze(1)), cate_gt).mean()
        ).item()
    else:
        z_pred = out.numpy()
        estimator_mse = mse(out.squeeze(1), y).mean().numpy()
        y_pred = out.squeeze(1).numpy()
        pehe = -1

    data = {"y": y, "z": z_pred.squeeze(1), "treat": T}
    if get_predictions:
        data["y_pred"] = y_pred

    if (not np.isnan(z_pred).any()) and (z_pred.sum() != 0):
        result, LRp_val = evaluate_fit(
            data,
            fit_type=fit_type,
            formula=formula,
            print_stats=print_stats,
            LR_test=LR_test,
            method=method,
        )
    else:
        print("Error, output z_pred contains NaNs")
        result = None
        LRp_val = None

    if len(b) == 4 or data_model_type == "simple":
        bprog = b[2]
        bpred = b[3]
        bmix = 0
    elif len(b) == 6 or data_model_type == "full":
        bprog = b[2]
        bpred = b[4]
        if b[3] != b[5]:
            bmix = (b[3] + b[5]) / 2
        else:
            bmix = b[3]

    if get_predictions:
        x_pred = dataset.x[:, 1]
        x_prog = dataset.x[:, 0]
        if k_fold_splits is not None:
            x_pred = x_pred[indices]
            x_prog = x_prog[indices]

        data["x_pred"] = x_pred
        data["x_prog"] = x_prog

        # saving model predictions
        if is_mtl_model:
            data["y1_pred"] = out[:, 1].numpy()
            data["y0_pred"] = out[:, 0].numpy()
        else:
            data["y_pred"] = out.numpy()

    if result is not None:
        output = {
            "b": b,
            "bprog": bprog,
            "bpred": bpred,
            "bmix": bmix,
            "weight_decay": params["optimizer_params"]["weight_decay"],
            "pehe": pehe,
            "mse_accuracy": estimator_mse,
            "pval": result.pvalues.tolist(),
            "tval": result.tvalues.tolist(),
            "fit_params": result.params.tolist(),
            "fit_params_bse": result.bse.tolist(),
            "fit_params_ci_l": result.conf_int(alpha=0.05)[0].tolist(),
            "fit_params_ci_h": result.conf_int(alpha=0.05)[1].tolist(),
            "pred_feature": pred_feature,
            "prog_feature": prog_feature,
            "k_fold_splits": k_fold_splits,
            "fold_idx": fold_idx,
        }
    else:
        output = {
            "b": b,
            "bprog": bprog,
            "bpred": bpred,
            "bmix": bmix,
            "weight_decay": params["optimizer_params"]["weight_decay"],
            "pehe": pehe,
            "mse_accuracy": estimator_mse,
            "pval": -1,
            "tval": -1,
            "fit_params": -1,
            "fit_params_bse": -1,
            "fit_params_ci_l": -1,
            "fit_params_ci_h": -1,
            "pred_feature": pred_feature,
            "prog_feature": prog_feature,
            "k_fold_splits": k_fold_splits,
            "fold_idx": fold_idx,
        }

    if get_reference_fit_results:
        x = dataset.x
        if k_fold_splits is not None:
            x = x[indices]
        data_ref = {"y": y, "z": x[:, 1], "treat": T}
        result_ref, LRp_val_ref = evaluate_fit(
            data_ref,
            fit_type=fit_type,
            formula=formula,
            print_stats=print_stats,
            LR_test=LR_test,
            method=method,
        )
        # get predictions
        pred_res = result_ref.predict(data_ref)
        # counterfactual predictions
        data_ref_counterf = data_ref.copy()
        data_ref_counterf["treat"] = 1 - data_ref["treat"]
        pred_res_counterf = result_ref.predict(data_ref_counterf)
        # get mse
        mse_err = mse(torch.Tensor(pred_res), y).mean()
        # get pehe using predictions from linear model
        y_counterf = dataset.y_counterf
        if k_fold_splits is not None:
            y_counterf = y_counterf[indices]
        cate_gt = (y - y_counterf) * (2 * T - 1.0)
        cate_pred = (torch.Tensor(pred_res) - torch.Tensor(pred_res_counterf)) * (
            2 * T - 1.0
        )
        pehe = torch.sqrt(mse(cate_pred, cate_gt).mean()).item()
        output_ref = {
            "b": b,
            "bprog": bprog,
            "bpred": bpred,
            "bmix": bmix,
            "weight_decay": 0.0,
            "mse_accuracy": mse_err.item(),
            "pval": result_ref.pvalues.tolist(),
            "tval": result_ref.tvalues.tolist(),
            "fit_params": result_ref.params.tolist(),
            "fit_params_bse": result_ref.bse.tolist(),
            "fit_params_ci_l": result_ref.conf_int(alpha=0.05)[0].tolist(),
            "fit_params_ci_h": result_ref.conf_int(alpha=0.05)[1].tolist(),
            "pred_feature": pred_feature,
            "prog_feature": prog_feature,
            "pehe": pehe,
            "k_fold_splits": k_fold_splits,
            "fold_idx": fold_idx,
        }

    # fit with prognostic covariate as input
    if get_progreference_fit_results:
        x = dataset.x
        if k_fold_splits is not None:
            x = x[indices]
        data_progref = {"y": y, "z": x[:, 0], "treat": T}
        result_progref, LRp_val_progref = evaluate_fit(
            data_progref,
            fit_type=fit_type,
            formula=formula,
            print_stats=print_stats,
            LR_test=LR_test,
            method=method,
        )
        # get predictions
        pred_res = result_progref.predict(data_progref)
        # counterfactual predictions
        data_progref_counterf = data_progref.copy()
        data_progref_counterf["treat"] = 1 - data_progref["treat"]
        pred_res_counterf = result_progref.predict(data_progref_counterf)
        # get mse
        mse_err = mse(torch.Tensor(pred_res), y).mean()
        # get pehe using predictions from linear model
        y_counterf = dataset.y_counterf
        if k_fold_splits is not None:
            y_counterf = y_counterf[indices]
        cate_gt = (y - y_counterf) * (2 * T - 1.0)
        cate_pred = (torch.Tensor(pred_res) - torch.Tensor(pred_res_counterf)) * (
            2 * T - 1.0
        )
        pehe = torch.sqrt(mse(cate_pred, cate_gt).mean()).item()
        output_progref = {
            "b": b,
            "bprog": bprog,
            "bpred": bpred,
            "bmix": bmix,
            "weight_decay": 0.0,
            "mse_accuracy": mse_err.item(),
            "pval": result_progref.pvalues.tolist(),
            "tval": result_progref.tvalues.tolist(),
            "fit_params": result_progref.params.tolist(),
            "fit_params_bse": result_progref.bse.tolist(),
            "fit_params_ci_l": result_progref.conf_int(alpha=0.05)[0].tolist(),
            "fit_params_ci_h": result_progref.conf_int(alpha=0.05)[1].tolist(),
            "pred_feature": pred_feature,
            "prog_feature": prog_feature,
            "pehe": pehe,
            "k_fold_splits": k_fold_splits,
            "fold_idx": fold_idx,
        }

    # if get_saliency_maps is not None:
    if get_saliency_maps:
        y_list = []
        T_list = []
        saliency_maps = []
        # for x, targets in tq.tqdm(dl, position=1):
        for input, targets in dl:
            with torch.no_grad():
                y = targets[:, 0]
                T = targets[:, 1].long()
                y_list.append(y.cpu().detach())
                T_list.append(T)
                if use_cuda:
                    device = torch.device(
                        "cuda" if (torch.cuda.is_available() and use_cuda) else "cpu"
                    )
                    model.to(device)
                    input = input.to(device)
                    y = y.to(device)
            input.requires_grad_()
            out = model(input)
            if get_saliency_maps == "from_output":
                if is_mtl_model:
                    # out = torch.cat(out, 1)
                    (out[1] - out[0]).mean().backward()
                else:
                    out.mean().backward()
                saliency_maps.append(input.grad.clone())
            else:
                if is_mtl_model:
                    out = torch.cat(out, 1)
                    error_e = mse(out[range(out.shape[0]), T], y)
                else:
                    error_e = mse(out, y)
                error_e.mean().backward()
                saliency_maps.append(input.grad.clone())
            if get_saliency_maps != "all":
                break  # only get saliency_map for first batch
        # concat "patches" if using minibatches
        out = (
            [torch.cat(predictions[0]), torch.cat(predictions[1])]
            if is_mtl_model
            else torch.cat(predictions)
        )
        y = torch.cat(y_list)[:, None]
        T = torch.cat(T_list)
        saliency_maps = torch.cat(saliency_maps)

    if result is not None:
        if fit_type == "logit":
            output.update(
                {"pseudo_rsq": result.prsquared, "llr_pvalue": result.llr_pvalue}
            )
        elif fit_type == "linear":
            output.update(
                {
                    "tval": result.tvalues.tolist(),
                    "f_pvalue": result.f_pvalue.tolist(),
                    "use_t": result.use_t,
                    # "pseudo_rsq": result.prsquared,
                }
            )
            if get_reference_fit_results:
                output_ref.update(
                    {
                        "tval": result_ref.tvalues.tolist(),
                        "f_pvalue": result_ref.f_pvalue.tolist(),
                        "use_t": result_ref.use_t,
                    }
                )

    if additional_key_path_to_save is not None:
        if not isinstance(additional_key_path_to_save, list):
            additional_key_path_to_save = [additional_key_path_to_save]

        for key_path in additional_key_path_to_save:
            if isinstance(key_path, list):
                key_path = ".".join(key_path)

            keys = key_path.split(".")
            value = checkpoint
            for key in keys:
                if key in value:
                    value = value[key]
                else:
                    raise KeyError(f"Key '{key}' not found in the checkpoint.")
            output.update({key_path: value})

            if get_reference_fit_results:
                output_ref.update({key_path: value})
            if get_progreference_fit_results:
                output_progref.update({key_path: value})

    if get_input:
        output = (output, input, data)
    if get_saliency_maps:
        if not isinstance(output, tuple):
            output = (output,)
        output = output + (saliency_maps,)
    if get_reference_fit_results:
        if not isinstance(output, tuple):
            output = (output,)
        output = output + (output_ref,)
    if get_progreference_fit_results:
        if not isinstance(output, tuple):
            output = (output,)
        output = output + (output_progref,)
    if get_predictions:
        if not isinstance(output, tuple):
            output = (output,)
        output = output + (data,)
    return output


def get_interpretation(
    experiment_dir,
    log_name,
    b=None,
    env="test",
    data_model_type=None,
    n_batch=None,
    dataset_root="/absolute/path/to/datasets",
    num_workers=0,
    pin_memory=False,
    k_fold_splits=None,
    fold_idx=None,
    get_segmentation_dl=False,
    **kwargs,
):
    log_dir = os.path.join(experiment_dir, log_name)

    with open(
        os.path.join(
            log_dir,
            "lightning_logs",
            "version_0",
            "hparams.yaml",
        ),
        "r",
    ) as stream:
        params = yaml.load(stream, Loader=yaml.CLoader)
        if isinstance(params, list):
            params = params[-1]
        if b is None:
            b = params["data_params"]["b"]

        # input_type = params["data_params"]["input_type"]
        data_model_type = params["data_params"]["data_model_type"]

    params["data_params"]["save_num_data_dir"] = None
    params["data_params"]["root"] = dataset_root

    n_batch = params["loader_params"]["batch_size"] if n_batch is None else n_batch

    # load model
    checkpoint_filename = [
        os.path.join("lightning_logs", "version_0", "checkpoints", fname)
        for fname in os.listdir(
            os.path.join(log_dir, "lightning_logs", "version_0", "checkpoints")
        )
        if fname.endswith(".ckpt")
    ]
    if len(checkpoint_filename) > 1:
        print(
            f"Warning: More than one checkpoint file found {checkpoint_filename}. Using {checkpoint_filename[0]}."
        )
    checkpoint = torch.load(os.path.join(log_dir, checkpoint_filename[0]))
    model = ToyModelImgModule.load_from_checkpoint(
        os.path.join(log_dir, checkpoint_filename[0])
    )

    model.eval()

    dataset = dataset_dict[checkpoint["hyper_parameters"]["data_type"]](
        env=env,
        train=False,
        load_1d_covariates=True,
        **params["data_params"],
    )
    dataset_len = len(dataset)
    if k_fold_splits is not None:
        kfold = KFold(n_splits=k_fold_splits, shuffle=True, random_state=42)
        train_indices, val_indices = list(kfold.split(np.arange(dataset_len)))[fold_idx]
        indices = val_indices if env.endswith("val") else train_indices
    else:
        indices = None
    dl = DataLoader(
        dataset=torch.utils.data.Subset(dataset, indices)
        if indices is not None
        else dataset,
        batch_size=n_batch,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    if get_segmentation_dl:
        # check if data_type is NSCLC otherwise raise not implemented error
        if checkpoint["hyper_parameters"]["data_type"] != "nsclcradiomics":
            raise NotImplementedError(
                "Segmentation maps are only available for NSCLC dataset"
            )
        else:
            # load segmentation maps
            seg_dataset = dataset_dict["nsclcradiomicsseg"](
                env=env,
                train=False,
                **params["data_params"],
            )
            seg_dl = DataLoader(
                dataset=torch.utils.data.Subset(seg_dataset, indices)
                if indices is not None
                else seg_dataset,
                batch_size=n_batch,
                num_workers=num_workers,
                pin_memory=pin_memory,
                shuffle=False,
            )

    if len(b) == 4 or data_model_type == "simple":
        bprog = b[2]
        bpred = b[3]
    elif len(b) == 6 or data_model_type == "full":
        # b = [0.0, 0.0, b_prog, b_mix, b_pred, b_mix]
        bprog = b[2]
        bpred = b[4]

    if get_segmentation_dl:
        return model, dl, bprog, bpred, seg_dl
    else:
        return model, dl, bprog, bpred


def evaluate_fit(
    data,
    fit_type="logit",
    formula="y ~ z + treat + z * treat",
    print_stats=False,
    LR_test=False,
    method=None,
):
    assert fit_type in ["logit", "linear"], "Unknown type of fit model"

    if fit_type == "logit":
        # fit logistic model
        if method is None:
            method = "newton"  # default method
        result = smf.logit(formula=formula, data=data).fit(disp=0, method=method)

        if print_stats:
            print(result.summary())
            print(result.params)
            print(result.bse)  # standard error
            print(result.pvalues)
            print(result.pvalues < 0.05)
            print(result.conf_int(alpha=0.05))
            print(result.resid_generalized.mean())

        if LR_test:
            # full
            formula = "y ~ z + treat + z * treat"
            res1 = smf.logit(formula=formula, data=data).fit(disp=0)
            # reduced
            formula = "y ~ z + treat"
            res2 = smf.logit(formula=formula, data=data).fit(disp=0)
            full_ll = res1.llf
            reduced_ll = res2.llf
            LR_statistic = -2 * (reduced_ll - full_ll)
            LRp_val = scipy.stats.chi2.sf(LR_statistic, 2)
            if print_stats:
                print(f"LR test for z*treat term: p-val = {LRp_val}")
        else:
            LRp_val = -1
        return result, LRp_val

    elif fit_type == "linear":
        # fit ordinary least squares
        result = smf.ols(formula=formula, data=data).fit()
        if print_stats:
            print(result.summary())

        if LR_test:
            # full
            formula = "y ~ z + treat + z * treat"
            res1 = smf.ols(formula=formula, data=data).fit()
            # reduced
            formula = "y ~ z + treat"
            res2 = smf.ols(formula=formula, data=data).fit()
            full_ll = res1.llf
            reduced_ll = res2.llf
            LR_statistic = -2 * (reduced_ll - full_ll)
            LRp_val = scipy.stats.chi2.sf(LR_statistic, 2)
            if print_stats:
                print(f"LR test for z*treat term: p-val = {LRp_val}")

        else:
            LRp_val = -1

        return result, LRp_val


def plot_coeff(
    df_results,
    bconst="pred",
    w_ind=0,
    coeff_ind=[1, 3],
    model="logit",
    bmix_ind=None,
    use_t=None,
    subplots=[3, 3],
    ylim=None,
    xlim=None,
    ylim_list=None,
    use_abs=False,
):
    bvar_dict = {"pred": "prog", "prog": "pred"}
    bvar = bvar_dict[bconst]
    if model in ["logit", "linear"]:
        colours = ["darkgrey", "cornflowerblue", "slategray", "orange"]
        coloursp = ["black", "darkblue", "black", "darkorange"]
        labels = np.array(["intercept", "z", "treat", "z*treat"])
    else:
        colours = ["cornflowerblue", "slategray", "orange"]
        coloursp = ["darkblue", "black", "darkorange"]
        labels = np.array(["z", "treat", "z*treat"])

    b_list = sorted(df_results[f"b{bconst}"].unique())
    assert subplots[0] * subplots[1] >= len(b_list)
    plt.figure(figsize=(subplots[1] * 6, subplots[0] * 6))

    for b_ind, b in enumerate(b_list):
        plt.subplot(subplots[0], subplots[1], b_ind + 1)
        wd = sorted(pd.unique(df_results["weight_decay"]))[w_ind]

        idx = (df_results["weight_decay"] == wd) & (df_results[f"b{bconst}"] == b)
        if bmix_ind is not None:
            bm = sorted(pd.unique(df_results.get("bmix", [0])))[bmix_ind]
            if "bmix" in df_results.columns:
                idx = idx & (df_results["bmix"] == bm)

        for k, i in enumerate(coeff_ind):
            xs = df_results[f"b{bvar}"][idx]
            ys = df_results["fit_params"][idx].str[i]
            errs = df_results["fit_params_bse"][idx].str[i]
            ci_l = df_results["fit_params_ci_l"][idx].str[i]
            ci_h = df_results["fit_params_ci_h"][idx].str[i]

            if use_t if use_t is not None else df_results.get("use_t", [False])[0]:
                p = df_results["tval"][idx].str[i].tolist()
            else:
                p = df_results["pval"][idx].str[i].tolist()
            xs, ys, errs, ci_l, ci_h, p = zip(*sorted(zip(xs, ys, errs, ci_l, ci_h, p)))
            if use_abs:
                ys = np.abs(ys)
                ci_l = np.abs(ci_l)
                ci_h = np.abs(ci_h)
            plt.fill_between(xs, ci_l, ci_h, color=colours[i], alpha=0.2)
            plt.errorbar(
                xs,
                ys,
                c=colours[i],
                marker="*",
                linestyle="--",
                yerr=errs,
                label=labels[i],
            )
            plt.hlines(0, min(xs), max(xs), linestyle=":", color="grey")

            for j in range(len(xs)):
                if p[j] < 0.05:
                    plt.scatter(xs[j], ys[j], marker="s", s=100, c=coloursp[i])

        plt.legend()
        if ylim is not None:
            plt.ylim(ylim)
        if xlim is not None:
            plt.xlim(xlim)
        elif ylim_list is not None:
            plt.ylim(ylim_list[b_ind])
        if bmix_ind is not None:
            plt.title(
                f"$wd$={wd:.2f}, b$_{{{bconst}}}$=log({math.exp(b):.3f})$\\approx${b:.3f}, b$_{{mix}}\\approx${bm:.3f}"
            )
        else:
            plt.title(
                f"$wd$={wd:.2f}, b$_{{{bconst}}}$=log({math.exp(b):.3f})$\\approx${b:.3f}"
            )
        plt.xlabel(f"$b_{{{bvar}}}$")
    plt.show()


def plot_coeffratios(
    df_results,
    w_ind=0,
    model="logit",
    bmix_ind=None,
    use_t=None,
    ylim=None,
    xlim=None,
    use_abs=False,
    use_cmap=True,
    labels=None,
    linear_fit=None,
    use_log_scale=False,
    use_sns_palette=False,
    savefig=False,
    figsize=None,
    plot_errorbars=False,
    plot_ci=False,
):
    if model in ["logit", "linear"]:
        predprog_ind = [3, 1]
    else:
        predprog_ind = [2, 0]

    df_results = [df_results] if not isinstance(df_results, list) else df_results
    markers = ["o", "s", "+", "x", ">", "<"]
    # only use subset of colourmap to avoid having to light colors
    cmaps = []
    for cmap in [
        mpl.cm.jet(np.linspace(0, 1, 22)),
        mpl.cm.Oranges(np.linspace(0, 1, 50)),
        mpl.cm.Blues(np.linspace(0, 1, 50)),
        mpl.cm.Greys(np.linspace(0, 1, 50)),
        mpl.cm.Purples(np.linspace(0, 1, 50)),
        mpl.cm.Reds(np.linspace(0, 1, 50)),
    ]:
        cmap = mpl.colors.ListedColormap(cmap[0:, :-1])
        cmaps.append(cmap)

    if use_sns_palette is not None and not isinstance(use_sns_palette, str):
        colors = ["green", "orange", "blue", "grey", "purple", "red"]
    else:
        colors = sns.color_palette(use_sns_palette)

    subplots = [1, 1]
    # assert subplots[0] * subplots[1] >= len(b_list)
    add_figwidth = len(df_results) if use_cmap else 0
    if figsize is None:
        figsize = (subplots[1] * 6 + add_figwidth, subplots[0] * 6)
    plt.figure(figsize=figsize)
    fit_res = []

    for i, df in enumerate(df_results):
        wd = sorted(pd.unique(df["weight_decay"]))[w_ind]

        idx = df["weight_decay"] == wd

        if bmix_ind is not None:
            bm = sorted(pd.unique(df.get("bmix", [0])))[bmix_ind]
            if "bmix" in df.columns:
                idx = idx & (df["bmix"] == bm)

        xs = df[f"bpred"][idx] / df[f"bprog"][idx]
        bs = df[f"bpred"][idx]
        errs_pred = df["fit_params_bse"][idx].str[predprog_ind[0]]
        ci_l_pred = df["fit_params_ci_l"][idx].str[predprog_ind[0]]
        ci_h_pred = df["fit_params_ci_h"][idx].str[predprog_ind[0]]
        errs_prog = df["fit_params_bse"][idx].str[predprog_ind[1]]
        ci_l_prog = df["fit_params_ci_l"][idx].str[predprog_ind[1]]
        ci_h_prog = df["fit_params_ci_h"][idx].str[predprog_ind[1]]
        if not use_t:
            ys = (
                df["fit_params"][idx].str[predprog_ind[0]]
                / df["fit_params"][idx].str[predprog_ind[1]]
            )
            ys_err = ys * 0  # not available
        elif use_t == "pred_fit_param":
            ys = df["fit_params"][idx].str[predprog_ind[0]]
            ys_err = errs_pred
        elif use_t == "pred_tval":
            ys = df["tval"][idx].str[predprog_ind[0]]
            ys_err = errs_pred * 0  # not available
        elif use_t == "prog_fit_param":
            ys = df["fit_params"][idx].str[predprog_ind[1]]
            ys_err = errs_prog
        elif use_t == "prog_tval":
            ys = df["tval"][idx].str[predprog_ind[1]]
            ys_err = errs_prog * 0  # not available
        else:
            ys = (
                df["tval"][idx].str[predprog_ind[0]]
                / df["tval"][idx].str[predprog_ind[1]]
            )

            ys_err = ys * np.sqrt(
                (errs_pred / df["fit_params"][idx].str[predprog_ind[0]]) ** 2
                + (errs_prog / df["fit_params"][idx].str[predprog_ind[1]]) ** 2
            )
        ys_ci_l = ci_l_pred / ci_h_prog
        ys_ci_h = ci_h_pred / ci_l_prog

        xs, ys, bs, ys_err, ys_ci_l, ys_ci_h = zip(
            *sorted(zip(xs, ys, bs, ys_err, ys_ci_l, ys_ci_h))
        )
        if use_abs:
            ys = np.abs(ys)
            ys_ci_l = np.abs(ys_ci_l)
            ys_ci_h = np.abs(ys_ci_h)

        label = (
            labels[i] if ((labels is not None) and isinstance(labels, list)) else None
        )

        if use_cmap:
            plt.scatter(
                xs,
                ys,
                c=bs,
                cmap=cmaps[i],
                marker=markers[i],
                label=label,
                alpha=0.7,
            )
            plt.colorbar()
        else:
            plt.scatter(
                xs,
                ys,
                marker=markers[i],
                color=colors[i],
                label=label,
                alpha=0.7,
            )
        if plot_errorbars:
            print(f"relative error {np.nanmean(np.array(ys_err)/np.array(ys))}")
            plt.errorbar(
                xs,
                ys,
                c=colors[i],
                marker="",
                linestyle="",
                yerr=ys_err,
                label=None,
            )
        if plot_ci:
            plt.fill_between(xs, ys_ci_l, ys_ci_h, color=colors[i], alpha=0.2)

        if linear_fit is not None:
            df_xy = pd.DataFrame(list(zip(xs, ys)), columns=["xs", "ys"])
            df_xy.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_xy.dropna(subset=["xs", "ys"], how="any", inplace=True)
            if linear_fit == "add_constant":
                res = sm.OLS(
                    df_xy["ys"], sm.add_constant(df_xy["xs"]), missing="drop"
                ).fit()
            else:
                res = sm.OLS(df_xy["ys"], df_xy["xs"], missing="drop").fit()
            # print(res.summary(slim=True))
            fit_res.append([res.params.values, res.bse.values])
            pred_ols = res.get_prediction()
            iv_l = pred_ols.summary_frame()["obs_ci_lower"]
            iv_u = pred_ols.summary_frame()["obs_ci_upper"]
            plt.plot(df_xy["xs"], res.fittedvalues, ":", color=colors[i])

            if linear_fit == "plot_confidence_intervals":
                plt.plot(df_xy["xs"], iv_u, "--", color=colors[i])
                plt.plot(df_xy["xs"], iv_l, "--", color=colors[i])

        xs_min = 0.0  # np.min(np.array(xs)[np.isfinite(xs)])
        xs_max = (
            np.max(np.array(xs)[np.isfinite(xs)]) if np.any(np.isfinite(xs)) else 1.0
        )
        plt.hlines(0.0, xs_min, xs_max, linestyle=":", color="lightgrey")
        plt.hlines(1.0, xs_min, xs_max, linestyle=":", color="lightgrey")

        if use_log_scale:
            plt.yscale("log")
        if ylim is not None:
            plt.ylim(ylim)
        if xlim is not None:
            plt.xlim(xlim)

        plt.xlabel(f"$b_{{pred}}/b_{{prog}}$")
        if not use_t:
            plt.ylabel(f"$coeff(z*treat) / coeff(z), abs={use_abs}$")
        elif use_t == "pred_fit_param":
            plt.ylabel(f"$coeff(z*treat), abs={use_abs}$")
        elif use_t == "pred_tval":
            plt.ylabel(f"$t_{{z*treat}}, abs={use_abs}$")
        elif use_t == "prog_fit_param":
            plt.ylabel(f"$coeff(z), abs={use_abs}$")
        elif use_t == "prog_tval":
            plt.ylabel(f"$t_z, abs={use_abs}$")
        else:
            plt.ylabel(f"$t_{{z*treat}} / t_z$, abs={use_abs}")

        if labels is not None:
            plt.legend()

    # plt.ylabel("coefficient value")
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, bbox_inches="tight")
    plt.show()

    if linear_fit is not None:
        return fit_res


def plot_binnedcoeffratios_withbaselines_subplots(
    df_results_list,
    w_ind=0,
    model="logit",
    bmix_ind=None,
    use_t=None,
    ylim=None,
    ylim_list=None,
    xlim=None,
    xlim_list=None,
    use_abs=False,
    use_cmap=True,
    labels=None,
    linear_fit=None,
    use_log_scale=False,
    use_xlog_scale=False,
    use_sns_palette=False,
    bin_mode="fixed_binwidth",
    binwidth=1.0,
    xmax=10,
    nbins=10,
    savefig=False,
    figsize=None,
    plot_datapoints=True,
    start_idx=0,
    title=None,
    alpha=0.9,
    legend_ncol=1,
    subplots=[3, 2],
    return_ymeansmedians=False,
):
    if model in ["logit", "linear"]:
        predprog_ind = [3, 1]
    else:
        predprog_ind = [2, 0]

    markers = ["o", "s", "+", "x", ">", "<"]
    # only use subset of colourmap to avoid having to light colors
    cmaps = []
    for cmap in [
        mpl.cm.Greens(np.linspace(0, 1, 50)),
        mpl.cm.Oranges(np.linspace(0, 1, 50)),
        mpl.cm.Blues(np.linspace(0, 1, 50)),
        mpl.cm.Greys(np.linspace(0, 1, 50)),
        mpl.cm.Purples(np.linspace(0, 1, 50)),
        mpl.cm.Reds(np.linspace(0, 1, 50)),
    ]:
        cmap = mpl.colors.ListedColormap(cmap[15:, :-1])
        cmaps.append(cmap)

    if use_sns_palette is not None and not isinstance(use_sns_palette, str):
        colors = ["green", "orange", "blue", "grey", "purple", "red"]
        colors_dark = [
            "darkgreen",
            "darkorange",
            "darkblue",
            "black",
            "indigo",
            "darkred",
        ]
        colors_light = [
            "lightgreen",
            "yellow",
            "lightblue",
            "lightgrey",
            "mediumpuple",
            "salmon",
        ]
    else:
        colors = sns.color_palette(use_sns_palette)
        colors_dark = sns.color_palette("dark")
        colors_light = sns.color_palette("pastel")

    if figsize is None:
        figsize = (subplots[1] * 4, subplots[0] * 4)
    plt.figure(figsize=figsize)

    SMALL_SIZE = 11
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 16

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
    fit_res = []
    bplots = []
    lines = []
    sns.set_style("ticks")
    y_means_list = []
    y_medians_list = []
    for j, df_results in enumerate(df_results_list):
        plt.subplot(subplots[0], subplots[1], j + 1)
        for i, df in enumerate(df_results):
            wd = sorted(pd.unique(df["weight_decay"]))[w_ind]

            # Only one plot without subplots
            idx = df["weight_decay"] == wd

            if bmix_ind is not None:
                bm = sorted(pd.unique(df.get("bmix", [0])))[bmix_ind]
                if "bmix" in df.columns:
                    idx = idx & (df["bmix"] == bm)

            xs = df[f"bpred"][idx] / df[f"bprog"][idx]
            bs = df[f"bpred"][idx]
            if use_t:
                ys = (
                    df["tval"][idx].str[predprog_ind[0]]
                    / df["tval"][idx].str[predprog_ind[1]]
                )
            else:
                ys = (
                    df["fit_params"][idx].str[predprog_ind[0]]
                    / df["fit_params"][idx].str[predprog_ind[1]]
                )

            xs, ys, bs = zip(*sorted(zip(xs, ys, bs)))
            xs = np.array(xs)
            ys = np.array(ys)
            if use_abs:
                ys = np.abs(ys)
            label = (
                labels[i]
                if ((labels is not None) and isinstance(labels, list))
                else None
            )
            if plot_datapoints:
                if use_cmap:
                    plt.scatter(
                        xs,
                        ys,
                        c=bs,
                        cmap=cmaps[i],
                        marker=".",
                        label=None,
                        alpha=0.4,  # edgecolors="white"
                    )
                    plt.colorbar()
                else:
                    plt.scatter(
                        xs,
                        ys,
                        marker=".",
                        color=colors[i],
                        label=None,
                        alpha=0.4,  # edgecolors="white"
                    )

            df_xy = pd.DataFrame(list(zip(xs, ys)), columns=["xs", "ys"])
            df_xy.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_xy.dropna(subset=["xs", "ys"], how="any", inplace=True)
            df_xy = df_xy.sort_values("xs")

            if bin_mode == "fixed_binwidth":
                binmeans = [(0.5 + i) * binwidth for i in range(int(xmax / binwidth))]
                y_means = [
                    np.mean(ys[(xs > i - binwidth / 2) & (xs <= i + binwidth / 2)])
                    for i in binmeans
                ]
                y_stds = [
                    np.std(ys[(xs > i - binwidth / 2) & (xs <= i + binwidth / 2)])
                    for i in binmeans
                ]
                plt.scatter(
                    x=binmeans,
                    y=y_means,
                    alpha=0.9,
                    color=colors[i],
                    marker=markers[i],
                    label=label,
                )
                plt.errorbar(
                    x=binmeans,
                    y=y_means,
                    yerr=y_stds,
                    ls="",
                    lw=2,
                    alpha=0.2,
                    color=colors[i],
                    label=None,
                )
            elif bin_mode == "fixed_nbinpoints":
                df_splits = np.array_split(df_xy, nbins)
                x_means = [np.mean(df_splits[i]["xs"]) for i in range(nbins)]
                y_means = [np.mean(df_splits[i]["ys"]) for i in range(nbins)]
                y_stds = [np.std(df_splits[i]["ys"]) for i in range(nbins)]
                x_stds = [np.std(df_splits[i]["xs"]) for i in range(nbins)]
                plt.scatter(
                    x=x_means,
                    y=y_means,
                    alpha=0.9,
                    color=colors[i],
                    marker=markers[i],
                    label=label,
                )
                plt.errorbar(
                    x=x_means,
                    y=y_means,
                    yerr=y_stds,
                    xerr=x_stds,
                    ls="",
                    lw=2,
                    alpha=0.2,
                    color=colors[i],
                    label=None,
                )
            elif bin_mode == "binscatter":
                df_est = binscatter(x="xs", y="ys", data=df_xy, ci=(3, 3), nbins=nbins)
                plt.scatter(
                    x=df_est["xs"],
                    y=df_est["ys"],
                    alpha=0.9,
                    color=colors[i],
                    marker=markers[i],
                    label=label,
                )
                plt.errorbar(
                    x=df_est["xs"],
                    y=df_est["ys"],
                    yerr=df_est["ci"],
                    ls="",
                    lw=2,
                    alpha=0.2,
                    color=colors[i],
                    label=None,
                )
            elif bin_mode == "boxplots":
                binmeans = [(0.5 + i) * binwidth for i in range(int(xmax / binwidth))]
                y_means = [
                    ys[(xs > i - binwidth / 2) & (xs <= i + binwidth / 2)]
                    for i in binmeans
                ]

                if i < 2:
                    meanpointprops = dict(
                        marker="o",
                        markeredgecolor=colors_light[i],
                        markerfacecolor="black",
                        markeredgewidth=1.3,
                        markersize=5,
                    )
                    flierprops = dict(marker="o", markeredgecolor=colors_dark[i])
                    bplot = plt.boxplot(
                        x=y_means,
                        positions=binmeans,
                        widths=binwidth / 2,
                        patch_artist=True,
                        manage_ticks=False,
                        showmeans=True,
                        meanprops=meanpointprops,
                        flierprops=flierprops,
                    )
                    bplots.append(bplot)
                    for element in ["means", "medians"]:
                        plt.setp(bplot[element], color=colors_dark[i])
                    for patch in bplot["boxes"]:
                        patch.set_facecolor(colors[i] + (alpha,))
                    plt.errorbar(
                        x=binmeans,
                        y=[np.median(y_) for y_ in y_means],
                        xerr=[binwidth / 2] * len(binmeans),
                        ls="",
                        lw=2,
                        alpha=0.9,
                        color=colors_dark[i],
                        label=None,
                    )

                    print(f"{i} y-medians: {[np.median(y_) for y_ in y_means]}")
                    print(f"{i} y-means: {[np.mean(y_) for y_ in y_means]}")
                    if return_ymeansmedians:
                        y_medians_list.append([np.median(y_) for y_ in y_means])
                        y_means_list.append([np.mean(y_) for y_ in y_means])

            elif bin_mode == "boxplots_fixed_nbinpoints":
                df_splits = np.array_split(df_xy, nbins)
                x_means = [np.mean(df_splits[i]["xs"]) for i in range(nbins)]
                y_means = [df_splits[i]["ys"] for i in range(nbins)]
                # y_stds = [np.std(df_splits[i]['ys']) for i in range(nbins)]
                x_stds = [np.std(df_splits[i]["xs"]) for i in range(nbins)]
                if i < 2:
                    meanpointprops = dict(
                        marker="o",
                        markeredgecolor=colors_light[i],
                        markerfacecolor="black",
                        markeredgewidth=1.3,
                        markersize=5,
                    )
                    flierprops = dict(marker="o", markeredgecolor=colors_dark[i])
                    bplot = plt.boxplot(
                        x=y_means,
                        positions=x_means,
                        widths=x_stds,
                        patch_artist=True,
                        manage_ticks=False,
                        showmeans=True,
                        meanprops=meanpointprops,
                        flierprops=flierprops,
                    )
                    plt.errorbar(
                        x=x_means,
                        y=[np.median(y_) for y_ in y_means],
                        xerr=x_stds,
                        ls="",
                        lw=2,
                        alpha=0.4,
                        color=colors_dark[i],
                        label=None,
                    )
                    print(f"{i} y-medians: {[np.median(y_) for y_ in y_means]}")
                    print(f"{i} y-means: {[np.mean(y_) for y_ in y_means]}")
                    if return_ymeansmedians:
                        y_medians_list.append([np.median(y_) for y_ in y_means])
                        y_means_list.append([np.mean(y_) for y_ in y_means])
                    bplots.append(bplot)
                    for element in ["means", "medians"]:
                        plt.setp(bplot[element], color=colors_dark[i])
                    for patch in bplot["boxes"]:
                        patch.set_facecolor(colors[i] + (alpha,))
                    plt.gca().xaxis.set_major_formatter(
                        mtick.StrMethodFormatter("{x:.1f}")
                    )

            else:
                print(f"Unknown bin_mode {bin_mode}")

            if linear_fit is not None:
                if linear_fit == "add_constant":
                    res = sm.OLS(
                        df_xy["ys"], sm.add_constant(df_xy["xs"]), missing="drop"
                    ).fit()
                else:
                    res = sm.OLS(df_xy["ys"], df_xy["xs"], missing="drop").fit()
                fit_res.append([res.params.values, res.bse.values])
                pred_ols = res.get_prediction()
                iv_l = pred_ols.summary_frame()["obs_ci_lower"]
                iv_u = pred_ols.summary_frame()["obs_ci_upper"]

                if i > 1:
                    linestyles = ["--", "-."]
                    start_idx = start_idx if use_log_scale else 0
                    current_label = labels[i] if labels is not None else None
                    (line,) = plt.plot(
                        df_xy["xs"][start_idx:],
                        res.fittedvalues[start_idx:],
                        linestyles[i - 2],
                        color=colors[i],
                        label=current_label,
                    )
                    lines.append(line)
                    if linear_fit == "plot_confidence_intervals":
                        plt.fill_between(
                            df_xy["xs"], iv_u, iv_l, color=colors[i], alpha=0.2
                        )

            xs_min = 0.0  # np.min(np.array(xs)[np.isfinite(xs)])
            xs_max = (
                np.max(np.array(xs)[np.isfinite(xs)])
                if np.any(np.isfinite(xs))
                else 1.0
            )
            plt.hlines(0.0, xs_min, xs_max, linestyle="-", color="lightgrey")
            plt.hlines(1.0, xs_min, xs_max, linestyle=":", color="lightgrey")

            if use_log_scale:
                plt.yscale("log")
            if use_xlog_scale:
                plt.xscale("log")
            plt.minorticks_on()
            if ylim is not None and ylim_list is None:
                plt.ylim(ylim)
            elif ylim_list is not None:
                plt.ylim(ylim_list[j])
            if xlim is not None and xlim_list is None:
                plt.xlim(xlim)
            elif xlim_list is not None:
                plt.xlim(xlim_list[j])

            plt.xlabel(f"$b_{{pred}}/b_{{prog}}$")
            if use_t:
                if use_abs:
                    plt.ylabel(f"$|t_{{pred}} / t_{{prog}}|$")
                else:
                    plt.ylabel(f"$t_{{pred}} / t_{{prog}}$")
            else:
                if use_abs:
                    plt.ylabel("$|\\beta_{pred} / \\beta_{prog}|$")
                else:
                    plt.ylabel("$\\beta_{{pred}} / \\beta_{{prog}}$")
            if labels is not None:
                if bin_mode in ["boxplots", "boxplots_fixed_nbinpoints"]:
                    plt.legend(
                        [bplot["boxes"][0] for bplot in bplots] + lines,
                        labels,
                        ncol=legend_ncol,
                    )
                else:
                    plt.legend(ncol=legend_ncol)

        plt.tight_layout()
        if title is not None:
            plt.title(title)
    if savefig:
        if savefig.endswith(".svg"):
            new_rc_params = {"text.usetex": False, "svg.fonttype": "none"}
            mpl.rcParams.update(new_rc_params)
        plt.savefig(savefig, bbox_inches="tight")

    plt.show()

    output = []
    if linear_fit is not None:
        output.append(fit_res)
    if return_ymeansmedians:
        output.append({"means": y_means_list, "medians": y_medians_list})
    if len(output) > 1:
        return output
