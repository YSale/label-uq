from matplotlib import pyplot as plt, gridspec
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sn
from scipy import ndimage
from scipy.stats import t
from skimage.measure import regionprops


def plot_accuracy_rejection_curve(true_labels, pred_labels, uncertainties, save_path, rand_order,
                                  unc_labels=['aleatoric', 'epistemic', 'total']):
    """
    Plot the accuracy rejection curve.

    Parameters
    ----------
    true_labels : list
        List of the ground truth labels
    pred_labels : list
        Array of the predicted labels
    uncertainties : list
        Array of uncertainties for each label
    save_path : str
        Path where the plot will be saved

    Returns
    -------

    """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Serif",
        "font.size": 12})

    # x-values: rejection rates
    portion_vals = np.linspace(0, 1, 50, endpoint=False)
    n = pred_labels.shape[1]
    t_crit = t.ppf(0.975, n - 1)

    # check correctness
    corrects = np.equal(true_labels.reshape((true_labels.shape[0], 1)), pred_labels).astype(int)
    corrects_random = corrects[rand_order]

    # colors = ['lightblue', 'orange', 'black']
    for idx, lbl in enumerate(unc_labels):
        sort_index = np.argsort(uncertainties[:, idx, :], axis=0)[::-1, :]
        sorted_corrects = corrects[sort_index, :]

        acc_mean_list = []
        acc_std_list = []
        acc_mean_rand_list = []
        acc_std_rand_list = []

        # iterate through every step and calculate accuracy
        for portion in portion_vals:
            step_index = int(portion * sorted_corrects.shape[0])
            acc_mean_list.append(
                np.mean(np.sum(sorted_corrects[step_index:, :], axis=0) / sorted_corrects[step_index:, :].shape[0]))
            acc_std_list.append(
                np.std(np.sum(sorted_corrects[step_index:, :], axis=0) / sorted_corrects[step_index:, :].shape[
                    0]))
            acc_mean_rand_list.append(
                np.mean(np.sum(corrects_random[step_index:, :], axis=0) / corrects_random[step_index:, :].shape[0]))
            acc_std_rand_list.append(
                np.std(np.sum(corrects_random[step_index:, :], axis=0) / corrects_random[step_index:, :].shape[
                    0]))
        plt.plot(portion_vals * 100, acc_mean_list, label=unc_labels[idx], linestyle='-')
        plt.fill_between(portion_vals * 100, np.array(acc_mean_list) - np.array(acc_std_list) * t_crit / np.sqrt(n),
                         np.array(acc_mean_list) + np.array(acc_std_list) * t_crit / np.sqrt(n), alpha=0.2)

    plt.plot(portion_vals * 100, acc_mean_rand_list, linestyle='--', color='black', label='random')
    plt.fill_between(portion_vals * 100, np.array(acc_mean_rand_list) - np.array(acc_std_rand_list) * t_crit / np.sqrt(n),
                     np.array(acc_mean_rand_list) + np.array(acc_std_rand_list) * t_crit / np.sqrt(n), color='black', alpha=0.2)
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.3f}'))
    plt.xticks(list(range(0, 101, 20)), fontsize=12)
    plt.yticks(fontsize=12)
    # plt.ylim(0.825, 1.050)
    plt.ylabel('Accuracy in \%')
    plt.xlabel('Rejection in \%')
    plt.margins(0.05)
    plt.legend(loc='upper left')
    # plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def hist_uncertainty(df, save_path, bins=11, score='TU_var', unc_label='TU_{var}'):
    p = sn.histplot(data=df, x=score, hue='correct', palette=['maroon', 'darkolivegreen'], bins=bins, stat='proportion', common_norm=False, alpha=0.5)
    p.set(xlabel=f"${unc_label}$", ylabel="Proportion")
    plt.legend(labels=['Correct', 'Incorrect'])
    plt.savefig(save_path)
    plt.close()


def plot_images_uncertainty(images, labels, au_data, eu_data, save_path, y_up=0.25):
    plt.figure(figsize=(22, 9))
    ax = gridspec.GridSpec(1, 3)
    ax.update(wspace=0, hspace=0)
    ax1 = plt.subplot(ax[0, 0])
    ax2 = plt.subplot(ax[0, 1])
    ax3 = plt.subplot(ax[0, 2])

    # Plot the images
    ax1.imshow(images[0])
    ax1.axis('off')  # turns off the axis
    ax1.margins(x=0)
    ax2.imshow(images[1])
    ax2.axis('off')  # turns off the axis
    ax2.margins(x=0)

    # Compute bounding box around tumor
    # Label connected components
    labeled_image, num_features = ndimage.label(images[2])
    # Find bounding boxes for each labeled component
    props = regionprops(labeled_image)
    zoom_factor = 1.4
    margin = 10
    if len(props) == 1:
        y0, x0, y1, x1 = props[0].bbox
        y0 = int(y0 * zoom_factor) - margin
        x0 -= margin
        y1 = int(y1 * zoom_factor) + margin
        x1 += margin
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor='r', linewidth=1, facecolor='none')
        rect1 = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor='r', linewidth=1, facecolor='none')
        ax1.add_patch(rect)
        ax2.add_patch(rect1)
    elif len(props) > 1:
        num_pixels = []
        bboxes = []
        for prop in props:
            num_pixels.append(prop.num_pixels)
            bboxes.append(prop.bbox)

        num_pixels = np.array(num_pixels)
        sorted_indices = np.argsort(num_pixels)[::-1]
        for j, idx in enumerate(sorted_indices):
            skip = False
            for i in sorted_indices[:j]:
                y0, x0, y1, x1 = bboxes[i]
                y0 -= margin
                x0 -= margin
                y1 += margin
                x1 += margin
                y2, x2, y3, x3 = bboxes[idx]
                if x0 < x2 < x1 and y0 < y2 < y1:
                    skip = True
                if x0 < x3 < x1 and y0 < y3 < y1:
                    skip = True
                if x0 < x2 < x1 and y0 < y3 < y1:
                    skip = True
                if x0 < x3 < x1 and y0 < y2 < y1:
                    skip = True

            if not skip:
                y0, x0, y1, x1 = bboxes[idx]
                y0 = int(y0 * zoom_factor) - margin
                x0 -= margin
                y1 = int(y1 * zoom_factor) + margin
                x1 += margin
                rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor='r', linewidth=1, facecolor='none')
                rect1 = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor='r', linewidth=1, facecolor='none')
                ax1.add_patch(rect)
                ax2.add_patch(rect1)

    # Plot the barplot
    ax3.bar(labels, eu_data[0], label='EU')
    ax3.bar(labels, au_data[0], color='red', bottom=eu_data[0], label='AU')
    ax3.set_ylim(0, y_up)  # set the y-axis limit
    ax3.set_xticks(np.arange(len(labels)))
    ax3.set_xticklabels(labels, fontsize=15)
    ax3.set_yticks([0, 0.1, 0.2, 0.25])  # set the y-axis ticks
    plt.yticks(fontsize=15)
    ax3.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()