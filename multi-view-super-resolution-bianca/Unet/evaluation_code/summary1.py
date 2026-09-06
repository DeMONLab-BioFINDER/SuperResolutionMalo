import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MODELS = [
    ("axis0_on_UnetModel0", "Coronal"),
    ("axis1_on_UnetModel1", "Axial"),
    ("axis2_on_UnetModel2", "Sagittal"),
    ("average_UnetModel", "Average Fusion"),
    ("fusionCNN", "CNN Fusion"),
    ("3T_baseline", "3T Baseline"),
]

EVAL_AXES = [
    (0, "Evaluation on 0 coronal"),
    (1, "Evaluation on 1 axial"),
    (2, "Evaluation on 2 sagittal"),
    ("3D", "Evaluation on 3D"),
]


def read_sample_values(csv_path, column):
    df = pd.read_csv(csv_path)

    if column not in df.columns:
        raise ValueError(f"{csv_path} does not contain column: {column}")

    values = pd.to_numeric(
        df[column],
        errors="coerce",
    ).dropna().to_numpy()

    # Remove the last row if it contains a precomputed summary value
    if len(values) > 1:
        values = values[:-1]

    return values


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    mean = np.mean(values)

    if len(values) == 1:
        return mean, mean, mean

    sem = np.std(values, ddof=1) / np.sqrt(len(values))
    low = mean - 1.96 * sem
    high = mean + 1.96 * sem

    return mean, low, high


def format_ci(values):
    mean, low, high = mean_ci(values)

    if np.isnan(mean):
        return ""

    return f"{mean:.3f} ({low:.3f}, {high:.3f})"


def find_normal_csv(args, model_key, eval_axis):
    if model_key == "3T_baseline":
        if eval_axis == "3D":
            return Path(args.baseline_3d_csv)

        return Path(
            args.baseline_2d_pattern.format(
                axis=eval_axis
            )
        )

    if eval_axis == "3D":
        return Path(
            args.model_3d_pattern.format(
                model=model_key
            )
        )

    return Path(
        args.model_2d_pattern.format(
            model=model_key,
            axis=eval_axis,
        )
    )


def find_lpips_csv(args, model_key, eval_axis):
    if eval_axis == "3D":
        return None

    if model_key == "3T_baseline":
        return Path(
            args.baseline_lpips_pattern.format(
                axis=eval_axis
            )
        )

    return Path(
        args.model_lpips_pattern.format(
            model=model_key,
            axis=eval_axis,
        )
    )


def find_dice_csv(args, model_key):
    if model_key == "3T_baseline":
        return Path(args.baseline_dice_csv)

    return Path(
        args.model_dice_pattern.format(
            model=model_key
        )
    )


def build_metric_tables(
    args,
    metric_name,
    column_name,
    is_lpips=False,
):
    mean_table = pd.DataFrame(
        index=[label for _, label in MODELS],
        columns=[label for _, label in EVAL_AXES],
    )

    ci_table = pd.DataFrame(
        index=[label for _, label in MODELS],
        columns=[label for _, label in EVAL_AXES],
    )

    all_values = {}

    for model_key, model_label in MODELS:
        for eval_axis, eval_label in EVAL_AXES:
            if metric_name == "dice":
                if eval_axis != "3D":
                    mean_table.loc[
                        model_label,
                        eval_label
                    ] = ""

                    ci_table.loc[
                        model_label,
                        eval_label
                    ] = ""

                    continue

                csv_path = find_dice_csv(
                    args,
                    model_key,
                )

            elif is_lpips:
                csv_path = find_lpips_csv(
                    args,
                    model_key,
                    eval_axis,
                )

            else:
                csv_path = find_normal_csv(
                    args,
                    model_key,
                    eval_axis,
                )

            if csv_path is None:
                mean_table.loc[
                    model_label,
                    eval_label
                ] = ""

                ci_table.loc[
                    model_label,
                    eval_label
                ] = ""

                continue

            if not csv_path.exists():
                print(f"[Missing] {csv_path}")

                mean_table.loc[
                    model_label,
                    eval_label
                ] = ""

                ci_table.loc[
                    model_label,
                    eval_label
                ] = ""

                continue

            if eval_axis == "3D":
                use_column = column_name

            else:
                if metric_name in ["psnr", "ssim"]:
                    use_column = f"{column_name}_2d"
                else:
                    use_column = column_name

            values = read_sample_values(
                csv_path,
                use_column,
            )

            mean, _, _ = mean_ci(values)

            mean_table.loc[
                model_label,
                eval_label
            ] = round(mean, 3)

            ci_table.loc[
                model_label,
                eval_label
            ] = format_ci(values)

            all_values[
                (
                    model_label,
                    eval_label,
                    eval_axis,
                )
            ] = values

    return mean_table, ci_table, all_values


def plot_boxplot_grid(
    all_values,
    metric_label,
    output_path,
    mode,
):
    axis_label_fontsize = 15
    tick_fontsize = 12
    subplot_title_fontsize = 14

    if mode == "2D":
        fig, axes = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=(12, 10),
            sharex=True,
            sharey=False,
        )

        eval_axes_2d = [
            (
                0,
                "Evaluation on 0 coronal",
                "Evaluated on coronal",
            ),
            (
                1,
                "Evaluation on 1 axial",
                "Evaluated on axial",
            ),
            (
                2,
                "Evaluation on 2 sagittal",
                "Evaluated on sagittal",
            ),
        ]

        bottom_labels = []

        for row_index, (
            ax,
            (
                eval_axis,
                eval_label,
                eval_title,
            ),
        ) in enumerate(
            zip(
                axes,
                eval_axes_2d,
            )
        ):
            labels = []
            data = []

            for _, model_label in MODELS:
                key = (
                    model_label,
                    eval_label,
                    eval_axis,
                )

                if key in all_values:
                    labels.append(model_label)
                    data.append(all_values[key])

            if data:
                ax.boxplot(
                    data,
                    showfliers=True,
                )

                ax.set_ylabel(
                    metric_label,
                    fontsize=axis_label_fontsize,
                )

                ax.set_title(
                    eval_title,
                    fontsize=subplot_title_fontsize,
                )

                ax.tick_params(
                    axis="y",
                    labelsize=tick_fontsize,
                )

                ax.grid(
                    axis="y",
                    linestyle="--",
                    alpha=0.4,
                )

                if row_index < 2:
                    ax.tick_params(
                        axis="x",
                        which="both",
                        bottom=False,
                        labelbottom=False,
                    )

                else:
                    bottom_labels = labels

                    ax.set_xticks(
                        range(
                            1,
                            len(labels) + 1,
                        )
                    )

                    ax.set_xticklabels(
                        labels,
                        rotation=0,
                        ha="center",
                        fontsize=tick_fontsize,
                    )

                    ax.set_xlabel(
                        "Model",
                        fontsize=axis_label_fontsize,
                        labelpad=12,
                    )

            else:
                ax.set_title(
                    f"{eval_title} - no data",
                    fontsize=subplot_title_fontsize,
                )

        if bottom_labels:
            axes[-1].set_xticks(
                range(
                    1,
                    len(bottom_labels) + 1,
                )
            )

            axes[-1].set_xticklabels(
                bottom_labels,
                rotation=0,
                ha="center",
                fontsize=tick_fontsize,
            )

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()
        return

    if mode == "3D":
        labels = []
        data = []

        for _, model_label in MODELS:
            key = (
                model_label,
                "Evaluation on 3D",
                "3D",
            )

            if key in all_values:
                labels.append(model_label)
                data.append(all_values[key])

        if not data:
            print(
                f"[Skip] No data for {metric_label}"
            )
            return

        plt.figure(
            figsize=(12, 7)
        )

        plt.boxplot(
            data,
            showfliers=True,
        )

        plt.xticks(
            range(
                1,
                len(labels) + 1,
            ),
            labels,
            rotation=0,
            ha="center",
            fontsize=tick_fontsize,
        )

        plt.yticks(
            fontsize=tick_fontsize
        )

        plt.xlabel(
            "Model",
            fontsize=axis_label_fontsize,
            labelpad=12,
        )

        plt.ylabel(
            metric_label,
            fontsize=axis_label_fontsize,
        )

        plt.grid(
            axis="y",
            linestyle="--",
            alpha=0.4,
        )

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--model-2d-pattern",
        required=True,
    )

    parser.add_argument(
        "--model-3d-pattern",
        required=True,
    )

    parser.add_argument(
        "--model-lpips-pattern",
        required=True,
    )

    parser.add_argument(
        "--baseline-2d-pattern",
        required=True,
    )

    parser.add_argument(
        "--baseline-3d-csv",
        required=True,
    )

    parser.add_argument(
        "--baseline-lpips-pattern",
        required=True,
    )

    parser.add_argument(
        "--model-dice-pattern",
        required=True,
    )

    parser.add_argument(
        "--baseline-dice-csv",
        required=True,
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = [
        (
            "psnr",
            "psnr_brain",
            False,
        ),
        (
            "ssim",
            "ssim_brain",
            False,
        ),
        (
            "lpips",
            "lpips_brain_2d",
            True,
        ),
        (
            "dice",
            "dice_mean_labels",
            False,
        ),
    ]

    for (
        metric_name,
        column_name,
        is_lpips,
    ) in metrics:
        mean_table, ci_table, all_values = build_metric_tables(
            args=args,
            metric_name=metric_name,
            column_name=column_name,
            is_lpips=is_lpips,
        )

        mean_table.to_csv(
            output_dir
            / f"summary_mean_{metric_name}.csv"
        )

        ci_table.to_csv(
            output_dir
            / f"summary_ci_{metric_name}.csv"
        )

        if metric_name == "psnr":
            plot_boxplot_grid(
                all_values=all_values,
                metric_label="PSNR 2D",
                output_path=(
                    output_dir
                    / "boxplot_2d_psnr.png"
                ),
                mode="2D",
            )

            plot_boxplot_grid(
                all_values=all_values,
                metric_label="PSNR 3D",
                output_path=(
                    output_dir
                    / "boxplot_3d_psnr.png"
                ),
                mode="3D",
            )

        elif metric_name == "ssim":
            plot_boxplot_grid(
                all_values=all_values,
                metric_label="SSIM 2D",
                output_path=(
                    output_dir
                    / "boxplot_2d_ssim.png"
                ),
                mode="2D",
            )

            plot_boxplot_grid(
                all_values=all_values,
                metric_label="SSIM 3D",
                output_path=(
                    output_dir
                    / "boxplot_3d_ssim.png"
                ),
                mode="3D",
            )

        elif metric_name == "lpips":
            plot_boxplot_grid(
                all_values=all_values,
                metric_label="LPIPS",
                output_path=(
                    output_dir
                    / "boxplot_lpips.png"
                ),
                mode="2D",
            )

        elif metric_name == "dice":
            plot_boxplot_grid(
                all_values=all_values,
                metric_label="Dice",
                output_path=(
                    output_dir
                    / "boxplot_dice.png"
                ),
                mode="3D",
            )

    print(
        f"Done. Results saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()

