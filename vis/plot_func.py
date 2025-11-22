import os
import pandas as pd
import scanpy as sc
from seaborn import heatmap
import numpy as np
import matplotlib.pyplot as plt
from eva.eval import *
import seaborn as sns
import math
from sklearn.metrics import confusion_matrix


def scatter_plot(eva_metrics, method_colors, vis_dir):
    categories = eva_metrics.index.tolist()

    method_colors = {key: value for key, value in method_colors.items() if key in categories}
    # eva_metrics_overall = eva_metrics[['FOSCTTM', 'LTA']]

    eva_metrics_overall = eva_metrics[['OMI', 'BCI']]
    # eva_metrics_overall = eva_metrics.iloc[:, :2]
    columns = eva_metrics_overall.columns
    fig, ax = plt.subplots(figsize=(8, 5))

    for category, x, y in zip(categories, eva_metrics_overall[columns[0]], eva_metrics_overall[columns[1]]):
        ax.scatter(x, y, label=category, color=method_colors[category], s=90)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.autoscale(True)

    handles = [plt.Line2D([0], [0],
                          color=method_colors[cat], marker='o', lw=0) for cat in categories]

    legend = ax.legend(handles=handles,
                       labels=categories,
                       loc='center left',
                       bbox_to_anchor=(1.05, 0.5),
                       title="Methods",
                       frameon=False)
    fig.text(0.36, 0.02, ' Omics Mixing Index ', ha='center', fontsize=16)
    fig.text(0.02, 0.5, 'Bio Conservation Index ', va='center', rotation='vertical', fontsize=16)

    # fig.text(0.36, 0.02, 'FOSCTTM', ha='center', fontsize=20)
    # fig.text(0.02, 0.5, 'LTA', va='center', rotation='vertical', fontsize=20)
    plt.tight_layout(rect=[0.05, 0.05, 0.85, 1])
    plt.show()
    fig.savefig(os.path.join(vis_dir, "OMI VS BCI.png"),
                dpi=600, bbox_inches='tight')
    print(f"scatter_plot finished")


def metrics_plot(eva_metrics, method_colors, vis_dir):
    vis_dir_metrics = vis_dir + '/metrics_Bar_plot'
    if not os.path.exists(vis_dir_metrics):
        os.makedirs(vis_dir_metrics)

    categories = eva_metrics.index.tolist()

    eva_metrics.columns = ['FOSCTTM', 'LTA', 'ARI', 'NMI', 'AMI', 'Omics_mixing', 'Bio_var_conser', 'Accuracy']

    method_colors = {key: value for key, value in method_colors.items() if key in categories}

    # eva_metrics_add = eva_metrics.iloc[:, 2:]
    # columns = eva_metrics_add.columns

    for column in eva_metrics.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        values = eva_metrics[column]
        if np.any(np.isnan(values)):
            ylim_max = 1
        else:
            max_value = values.max()
            ylim_max = math.ceil(max_value / 0.1) * 0.1
        bars = ax.bar(categories, values, color=[method_colors[cat] for cat in categories], alpha=0.8)
        ax.set_title(column)  # 添加标题为列名
        ax.set_ylim(0, ylim_max)
        ax.set_xticks([])  # Remove x-axis labels

        # 为每个柱子添加数值标签
        for bar, value in zip(bars, values):
            if not np.isnan(value):  # 仅为非 NaN 值添加标签
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.3f}',
                        ha='center', va='bottom', fontsize=8)

        handles = [plt.Line2D([0], [0],
                              color=method_colors[cat], lw=4) for cat in categories]

        legend = ax.legend(handles=handles,
                           labels=categories,
                           loc='center left',
                           bbox_to_anchor=(1.05, 0.5),
                           title="Methods",
                           frameon=False)

        fig.text(0.02, 0.5, '', va='center', rotation='vertical', fontsize=14)
        plt.tight_layout(rect=[0.05, 0.05, 0.85, 1])

        # plt.show()
        fig.savefig(os.path.join(vis_dir_metrics, f"{column}_Bar.png"),
                    dpi=600, bbox_inches='tight')
        plt.close(fig)
    print(f"metrics_plot finished")


def umap_plot(adata, omic_colors, cell_type_colors, vis_dir):
    vis_dir_umap = vis_dir + '/UMAP_truth'
    if not os.path.exists(vis_dir_umap):
        os.makedirs(vis_dir_umap)

    obsm_names = adata.obsm_keys()

    for i, method in enumerate(obsm_names):
        sc.pp.neighbors(adata, use_rep=method)
        sc.tl.umap(adata)

        fig, ax = plt.subplots(figsize=(16, 7))
        ax.axis('off')
        ax.set_title(method, fontsize=25, x=0.5, y=0.9)
        inner_ax1 = ax.inset_axes([0.05, 0.1, 0.4, 0.8])
        inner_ax2 = ax.inset_axes([0.55, 0.1, 0.4, 0.8])
        inner_ax1.axis('off')
        inner_ax2.axis('off')

        fig_left = sc.pl.umap(adata,
                              size=80,
                              color='omic_id',
                              palette=omic_colors,
                              title='',
                              ax=inner_ax1,
                              show=False)

        legend = inner_ax1.legend(
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            title='Omic Types',
            frameon=False)
        fig_right = sc.pl.umap(adata,
                               size=100,
                               color="cell_type",
                               palette=cell_type_colors,
                               title='',
                               ax=inner_ax2,
                               show=False)
        legend1 = inner_ax2.legend(
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            title='Cell Types',
            frameon=False)
        plt.tight_layout(pad=3.0)
        plt.show()
        fig.savefig(os.path.join(vis_dir_umap, method + ".png"),
                    dpi=600, bbox_inches='tight')
        plt.close(fig)
    print(f"umap_plot finished")


def umap_clusters_plot(adata, cell_type_colors, eva_dir, vis_dir):
    """
    Generates UMAP visualization of predicted clusters from saved labels.

    Args:
        adata :  the co-embedding NumPy file (e.g., 'embedding.npy').
        eva_dir : Path to the predicted labels file (clu_lables_pred.txt).
        vis_dir : Path to save the plot.
        cell_type_colors
    The function computes UMAP on the embedding, colors by predicted clusters, and optionally by true labels.
    """

    ground_truth = adata.obs['cell_type']
    cell_types = sorted(np.unique(ground_truth))
    n_types = len(cell_types)

    vis_dir_clu_umap = os.path.join(vis_dir, 'Clu_umap')
    if not os.path.exists(vis_dir_clu_umap):
        os.makedirs(vis_dir_clu_umap)
    obsm_names = adata.obsm_keys()

    for i, method in enumerate(obsm_names):
        pred_labels_path = os.path.join(eva_dir, method + "/clu_lables_pred.txt")
        pred_clu_labels = np.loadtxt(pred_labels_path).astype(int)  # Load predicted labels

        pred_clu_labels = [cell_types[label] for label in pred_clu_labels]
        adata.obs['pred_cluster'] = pd.Categorical(pred_clu_labels)
        # Compute UMAP
        sc.pp.neighbors(adata, use_rep=method)
        sc.tl.umap(adata)

        # Plot predicted clusters
        fig, ax = plt.subplots(1, figsize=(8, 6))

        # UMAP colored by predicted clusters
        sc.pl.umap(adata,
                   color='pred_cluster',
                   ax=ax,
                   show=False,
                   frameon=False,
                   palette=cell_type_colors,
                   legend_fontsize=9,
                   legend_fontoutline=1,
                   size=30)
        ax.set_title(f' {method}', fontsize=20, pad=10)

        # Customize legend for cluster-color correspondence
        legend = ax.legend(
            loc='center left',
            bbox_to_anchor=(1.05, 0.5),
            title='Cell Types',
            frameon=False)
        plt.tight_layout()

        # Save
        save_path = os.path.join(vis_dir_clu_umap, method + "_umap_clu.png")
        plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    print(f"umap_clusters_plot finished")


def paga_plot(adata, obsm_names, cell_type_colors, vis_dir, threshold=0.05):
    vis_dir_paga_trajectory = vis_dir + '/PAGA_trajectory' + '_' + str(threshold)
    if not os.path.exists(vis_dir_paga_trajectory):
        os.makedirs(vis_dir_paga_trajectory)

    adata.obs['ground_truth'] = adata.obs['cell_type'].copy()

    if not adata.obs['ground_truth'].dtype.name == 'category':
        adata.obs['ground_truth'] = adata.obs['ground_truth'].astype('category')

    categories = adata.obs['ground_truth'].cat.categories.tolist()

    unique_cell_types_for_colors = np.unique(adata.obs['cell_type'].values)
    color_dict = dict(zip(unique_cell_types_for_colors, cell_type_colors))
    ground_truth_colors = [color_dict[cat] for cat in categories]

    adata.uns['ground_truth_colors'] = ground_truth_colors

    for method in obsm_names:

        sc.pp.neighbors(adata, use_rep=method)
        sc.tl.umap(adata)
        sc.tl.paga(adata, groups='ground_truth')

        fig, ax = plt.subplots(figsize=(22, 15))
        sc.pl.paga(adata,
                   threshold=threshold,
                   labels=None,
                   show=False,
                   ax=ax,
                   node_size_scale=10,
                   edge_width_scale=2,
                   node_size_power=0.5,
                   frameon=False)

        ax.axis('off')
        ax.set_title(f'{method}',
                     fontsize=60, x=0.5, y=0.95, ha='center')

        for artist in ax.get_children():
            if isinstance(artist, plt.Text):
                artist.set_visible(5)

        handles = [plt.Line2D([0], [0],
                              marker='o',
                              color='w',
                              label=cell_type,
                              markersize=10,
                              markerfacecolor=color)
                   for cell_type, color in zip(categories, cell_type_colors)]

        legend = ax.legend(handles=handles,
                           loc='center left',
                           bbox_to_anchor=(1, 0.5),
                           title='Cell Types')

        legend.get_frame().set_linewidth(0)
        plt.setp(legend.get_texts(), fontsize=13)
        legend.get_title().set_fontsize(20)

        plt.tight_layout()
        fig.subplots_adjust(right=0.85)
        # plt.show()
        fig.savefig(os.path.join(vis_dir_paga_trajectory, method + ".png"),
                    dpi=600, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    print(f"paga_plot finished")


def clu_heatmap_plot(adata, obsm_names, CM, accuracy, vis_dir, keys='clu_heatmap_plot'):
    vis_dir_confusion = os.path.join(vis_dir, keys)
    if not os.path.exists(vis_dir_confusion):
        os.makedirs(vis_dir_confusion)

    ground_truth = adata.obs['cell_type']
    labels = sorted(np.unique(ground_truth))
    n_classes = len(labels)

    fig_size = max(8, n_classes * 0.5)
    cmap = 'Reds'

    for method in obsm_names:
        cm = CM[method].astype(float)
        cm = cm / cm.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))

        ax = heatmap(
            cm,
            ax=ax,
            xticklabels=labels,
            yticklabels=labels,
            cmap=cmap,
            annot=True,
            fmt='.3f',
            cbar_kws={'shrink': 0.71, 'label': 'Normalized clustering Prediction Probability'},
            square=True
        )

        for text in ax.texts:
            if float(text.get_text()) == 0.0:
                text.set_text('')

        ax.set_title(
            f'{method}\n Clustering Accuracy: {accuracy[method]:.2f}',
            fontsize=24,
            pad=20,
        )

        ax.set_xticklabels(ax.get_xticklabels(),
                           rotation=45,
                           ha='right',
                           fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(),
                           rotation=0,
                           fontsize=10)

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=9)

        plt.tight_layout()

        save_path = os.path.join(vis_dir_confusion, f'{method}.png')
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close(fig)
    print(f"clu_heatmap_plot finished")


def classified_confusion_matrices(adata, eva_dir, vis_dir, clu_rna):
    """
    Reads true and predicted labels from eva_dir for each method, computes confusion matrices,
    and saves heatmaps to vis_dir.

    Assumes:
    - eva_dir contains subdirectories named after methods.
    - Each method subdirectory has {pred_file_pattern} (e.g., 'pred_labels.txt').
    - True labels are shared in eva_dir/{true_file} (e.g., 'true_labels.txt').

    Args:
        eva_dir (str): Path to evaluation directory.
        vis_dir (str): Path to visualization directory .
        adata:
        clu_rna:
    """
    pred_file_pattern = 'rna_Classifier_label_predict.txt'
    # Create vis_dir if not exists
    vis_dir_confusion = vis_dir + '/' + 'fenlei_confusion_matrix'
    os.makedirs(vis_dir_confusion, exist_ok=True)
    # Get methods (subdirectories in eva_dir)
    methods = [d for d in os.listdir(eva_dir) if os.path.isdir(os.path.join(eva_dir, d))]
    # Load shared true labels
    true_labels = clu_rna
    ground_truth = adata.obs['cell_type']
    labels = sorted(np.unique(ground_truth))
    for method in methods:

        pred_path = os.path.join(eva_dir, method, pred_file_pattern)
        if not os.path.exists(pred_path):
            print(f"Predicted labels not found for {method}: {pred_path}. Skipping.")
            continue

        # Load predicted labels
        pred_labels = np.loadtxt(pred_path)

        # Compute accuracy (LTA)
        correct = np.sum(pred_labels == true_labels)
        accuracy = correct / len(true_labels)

        # Compute confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)

        # Visualize and save
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm,
                    xticklabels=labels,
                    yticklabels=labels,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    cbar=False)

        plt.title(f'{method}\n LTA of True vs Predicted Labels ({accuracy:.3f})', fontsize=22, pad=20)

        save_path = os.path.join(vis_dir_confusion, method + ".png")
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
    print(f"classified_confusion_matrices finished")


def radar_plot(eva_metrics, vis_dir, colors):
    # Input validation
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    if eva_metrics.empty:
        raise ValueError("eva_metrics must be a non-empty DataFrame")

    # Get method names (index) and metric names (columns)
    methods = eva_metrics.index.tolist()
    metrics = eva_metrics.columns.tolist()

    # Normalize FOSCTTM: inverse, log scale, and standardize
    df_processed = eva_metrics.copy()
    # df_processed = df_processed.drop('raw_data', errors='ignore')

    if 'FOSCTTM' in metrics:
        df_processed['FOSCTTM'] = 1 - df_processed['FOSCTTM'].astype(float)

    # Radar chart setup
    N = len(metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Initialize figure with journal-quality settings
    fig, ax = plt.subplots(figsize=(12, 12),
                           subplot_kw=dict(polar=True), dpi=600)
    max_value = df_processed.max().max()
    min_value = df_processed.min().min()
    if max_value == min_value:
        max_value += 0.1
    ax.set_rlim(0.0, 1.0)  # Explicitly set rlim from 0 to 1 for center-to-edge scaling

    # Draw radar polygons without filling
    for idx, method in enumerate(methods):
        values = df_processed.loc[method].values.flatten().tolist()
        values += values[:1]

        ax.plot(angles, values,
                linewidth=2.5,
                linestyle='solid',
                label=method,
                color=colors[method],
                zorder=10)

        # Customize axes and labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([])  # Hide default tick labels to avoid overlap

        # Manually place metric labels outside the chart to avoid intersection, upright (no rotation)
        label_radius = 1.12  # Closer to the 1.0 rim
        for i, metric in enumerate(metrics):
            theta = angles[i]
            ax.text(theta, label_radius, metric,
                    ha='center', va='bottom',
                    fontsize=17,
                    color='black',
                    rotation=0,  # No rotation, upright text
                    rotation_mode='anchor')

    # Enhance radial grid with custom scale annotations
    r_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_rgrids([0.0] + r_levels, labels=None)  # Disable default labels
    ax.set_yticklabels([])  # Explicitly hide any remaining r tick labels

    # Small angle offset to place labels beside the radial lines
    delta_theta = 0.05  # Smaller offset in radians for closer placement beside lines

    for i, metric in enumerate(metrics):
        theta = angles[i]
        is_reverse = (metric == 'FOSCTTM')
        # Alternate offset direction for better spacing (left/right of line)
        sign = 1 if i % 2 == 0 else -1
        for r_level in r_levels:
            offset_r = 0.015 if r_level < 1.0 else -0.02
            display_r = r_level + offset_r
            if is_reverse:
                label_val = 1.0 - r_level
            else:
                label_val = r_level
            ax.text(theta + sign * delta_theta, display_r, f'{label_val:.1f}',
                    ha='center', va='center',
                    fontsize=15,
                    color='black',
                    alpha=0.9,
                    rotation=0,
                    rotation_mode='anchor')

    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.9, color='black')

    # Improve legend and background
    ax.spines['polar'].set_visible(False)
    ax.set_facecolor('#f5f5f5')
    fig.patch.set_facecolor('white')
    plt.legend(loc='upper right',
               bbox_to_anchor=(1.2, 1.1),
               fontsize=12,
               facecolor='white',
               edgecolor='black')

    # Save with high resolution
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'radar_plot.png'),
                dpi=600, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"radar_plot finished")


def purity_box_plot(adata, obsm_names, vis_dir, cluster_labels_np, colors):
    vis_dir_purity = os.path.join(vis_dir, 'Purity_box')
    if not os.path.exists(vis_dir_purity):
        os.makedirs(vis_dir_purity)
    # obsm_names = obsm_names.drop('raw_data', errors='ignore')
   
    k_values = list(range(0, 21, 5))
    k_values.append(1)
   
    purity_data = []
    for method in obsm_names:
        purities = [knn_purity_score(adata.obsm[method], cluster_labels_np, k) for k in k_values]
        for k, purity in zip(k_values, purities):
            purity_data.append({'Method': method, 'k': k, 'Purity': purity})

    df = pd.DataFrame(purity_data)

    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))

    palette = dict(zip(obsm_names, colors[:len(obsm_names)]))
    # sns.boxplot(data=df,
    #             x='Method',
    #             y='Purity',
    #             palette=palette,
    #             width=0.4,
    #             fliersize=4,
    #             linewidth=1)
    sns.lineplot(data=df,
                 x='k',
                 y='Purity',
                 hue='Method',
                 palette=palette,
                 linewidth=2.0)

    # 设置标题和标签
    plt.title('Purity Score ',
              fontsize=20, pad=20)
    plt.ylabel('kNN Purity Score', fontsize=18)
    plt.xlabel('Number of Nearest Neighbors', fontsize=18)

    # X 轴设置
    plt.xlim(0, 20)
    plt.xticks([0, 5, 10, 15, 20], fontsize=12)

    # Y 轴设置
    plt.ylim(0, 1.1)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=12)

    # 图例设置
    plt.legend(title='Methods',
               title_fontsize=13,
               loc='upper right',
               bbox_to_anchor=(1.2, 1.1),
               fontsize=12,
               facecolor='white',
               edgecolor='black')

    ax = plt.gca()
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.tight_layout()
    save_path = os.path.join(vis_dir_purity, "purity_lineplot.png")
    plt.savefig(save_path,
                dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"purity_box_plot finished")


def total_performance_plot(eva_metrics, method_colors, vis_dir):
    """
    Computes the overall performance score for each method by averaging all metrics except the first one
    (without normalization), then generates a bar plot.

    Args:
        eva_metrics (pd.DataFrame): DataFrame with methods as index, metrics as columns.
        method_colors (dict): Dict mapping method names to colors.
        vis_dir (str): Directory to save the plot.
    """
    # Skip 'raw_data' if present
    # eva_metrics = eva_metrics.drop('raw_data', errors='ignore')

    # Compute overall performance as row-wise mean, excluding the first metric (raw values)
    first_metric = eva_metrics.columns[0]
    eva_metrics['total_performance'] = eva_metrics.drop(columns=[first_metric]).mean(axis=1)

    # Prepare data for plotting
    methods = eva_metrics.index.tolist()
    total_scores = eva_metrics.drop(columns=[first_metric]).mean(axis=1).values

    # Create bar plot
    fig, ax = plt.subplots(figsize=(7, 5), dpi=600)
    bars = ax.bar(range(len(methods)), total_scores, color=[method_colors[method] for method in methods], alpha=0.8)

    # Customize plot

    ax.set_title(f'\nTotal Performance', fontsize=20, pad=20)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([])

    # Add legend with methods and colors
    handles = [plt.Line2D([0], [0],
                          color=method_colors[method], lw=4) for method in methods]
    legend = ax.legend(handles,
                       methods,
                       loc='center left',
                       bbox_to_anchor=(1, 0.5),
                       title="Methods",
                       frameon=False)

    # Add value labels on bars
    for bar, score in zip(bars, total_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=9)

    # Save plot
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    save_path = os.path.join(vis_dir, f'total_performance.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"total_performance_plot finished")


def mix_and_bio_plot(mix_and_bio_metric, method_colors, vis_dir):
    """
    Generates grouped bar plots for omics mixing and bio var conservation metrics.

    Args:
        mix_and_bio_metric (pd.DataFrame): DataFrame with metrics as columns and methods as index.
        method_colors (dict): Dict of method names to colors for bars.
        vis_dir (str): Directory to save the plot.
    """
    # Create directory if not exists
    mix_bio_path = os.path.join(vis_dir, 'mix_and_bio_Bar_plot')
    os.makedirs(mix_bio_path, exist_ok=True)

    # Select only the 6 specific metrics (columns)
    metric_names = ['NOC', 'GC', 'SAS', 'ASW', 'PS', 'MAP']
    mix_and_bio_metric = mix_and_bio_metric[metric_names]  # Slice to the exact 6 columns

    # Methods (index)
    methods = mix_and_bio_metric.index.tolist()

    # Colors for methods
    colors = [method_colors[method] for method in methods]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=600)

    # Left subplot: Omics Mixing Metrics (first 3 columns: NOC, GC, SAS)
    mixing_metrics = mix_and_bio_metric[['NOC', 'GC', 'SAS']]  # Select first 3 columns, rows methods
    x_pos = np.arange(len(mixing_metrics.columns))  # Positions for metrics (3,)
    width = 0.8 / len(methods)  # Adjusted width for number of methods

    for i, method in enumerate(methods):
        values = mixing_metrics.loc[method].values  # Row for method, shape (3,)
        bars = ax1.bar(x_pos + i * width, values, width, label=method, color=colors[i], alpha=0.9)

    ax1.set_xlabel('', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Omics Mixing Index', fontsize=18, pad=20)
    ax1.set_xticks(x_pos + width * (len(methods) - 1) / 2)
    ax1.set_xticklabels(['NOC', 'GC', 'SAS'], fontsize=10)

    # Set y-limit slightly higher to fit labels
    max_y1 = max(mixing_metrics.max().max(), 0) * 1.05
    ax1.set_ylim(0, max_y1)

    # Right subplot: Bio Var Conservation Metrics (last 3 columns: ASW, PS, MAP)
    bio_metrics = mix_and_bio_metric[['ASW', 'PS', 'MAP']]  # Select last 3 columns, rows methods
    for i, method in enumerate(methods):
        values = bio_metrics.loc[method].values  # Row for method, shape (3,)
        bars = ax2.bar(x_pos + i * width, values, width, label=method, color=colors[i], alpha=0.9)

    ax2.set_xlabel('', fontsize=12, fontweight='bold')
    ax2.set_ylabel('', fontsize=12)
    ax2.set_title('Bio Conservation Index', fontsize=18, pad=20)
    ax2.set_xticks(x_pos + width * (len(methods) - 1) / 2)
    ax2.set_xticklabels(['ASW', 'PS', 'MAP'], fontsize=10)
    handles = [plt.Line2D([0], [0],
                          color=method_colors[cat], lw=3) for cat in methods]

    ax2.legend(handles=handles,
               labels=methods,
               bbox_to_anchor=(1.05, 1),
               loc='upper left',
               title="Methods",
               frameon=False,
               fontsize=9)

    # Set y-limit slightly higher to fit labels
    max_y2 = max(bio_metrics.max().max(), 0) * 1.05
    ax2.set_ylim(0, max_y2)

    # Overall layout and save
    plt.tight_layout()
    save_path = os.path.join(mix_bio_path, 'mix_bio_bar.png')
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"mix_and_bio_abr_plot finished")




