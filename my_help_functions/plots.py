import base64
import io

import matplotlib.pyplot as plt
import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
import torch
from IPython.display import HTML, display
from tabulate import tabulate

from my_help_functions.cosine_matrix import (
    get_positions_of_classes_on_flattened_image_for_collage,
)
from my_help_functions.vis_model_arch import (
    draw_bn_dependency_graph_inline,
    do_the_magic,
)

from my_help_functions.tools import (
    cos_sim,
)


def scrollable_plot(fig, height=400):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    
    html = f"""
    <div class="scrollable-output" style="max-height: {height}px;">
        <img src="data:image/png;base64,{img_base64}">
    </div>
    """
    plt.close(fig)
    display(HTML(html))


def show_feature_maps(
    conv,
    savefig
):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    img0 = axes[0].imshow(torch.mean(conv['before conv' ], dim=0).detach().cpu())
    img1 = axes[1].imshow(torch.mean(conv['after conv'  ], dim=0).detach().cpu())
    img2 = axes[2].imshow(torch.mean(conv['after center'], dim=0).detach().cpu())

    axes[0].set_title("Before conv")
    axes[1].set_title("After conv")
    axes[2].set_title("After centering")

    fig.colorbar(img0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(img1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(img2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.suptitle("Channelwise-meaned feature maps")
    if savefig: plt.savefig("01. Карты признаков.png", bbox_inches="tight")
    plt.show();


def show_cos_sim_matrices(
    matrices,
    savefig,
    type        # diff | abs
):
    if type not in ("diff", "abs"):
        raise ValueError("type must be 'diff' or 'abs'")

    if type == "abs":
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        for name, ax in zip(matrices.keys(), axes.flat):
            img = ax.imshow(cos_sim(matrices[name]))
            ax.set_title(name)
            fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.suptitle("Cosine similarity matrices")
        if savefig: plt.savefig("02. Взаимные углы.png")
        plt.show();

    else:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
        img = axes[0].imshow(cos_sim(matrices['after conv']) - cos_sim(matrices['before conv']))
        axes[0].set_title("while conv")
        fig.colorbar(img, ax=axes[0], fraction=0.046, pad=0.04)

        img = axes[1].imshow(cos_sim(matrices['after center']) - cos_sim(matrices['after conv']))
        axes[1].set_title("while centering")
        fig.colorbar(img, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.suptitle("Angles change")
        if savefig: plt.savefig("03. Изменение углов.png")
        plt.show();


def per_class_angles(
    matrices,
    conv,
    idx,
    gray,
    type,          # diff | abs
    operation,     # conv | center
    savefig,
    visualize=True,
):
    if type not in ("diff", "abs"):
        raise ValueError("type must be 'diff' or 'abs'")

    if operation not in ("conv", "center"):
        raise ValueError("operation must be 'conv' or 'center'")

    positions, class_names = get_positions_of_classes_on_flattened_image_for_collage(
        idx,
        conv['after center'].shape[-1],
        f"{'gray_' if gray else ''}"
    )

    if type == "abs":
        second = f"after {operation}"
        cos_sim_matr = cos_sim(matrices[second])
    else:
        if operation == "conv":
            first = "before conv"
            second = "after conv"
        else:
            first = "after conv"
            second = "after center"

        cos_sim_matr = cos_sim(matrices[second]) - cos_sim(matrices[first])

    rows = cols = len(positions)

    if visualize:
        fig, axes = plt.subplots(rows, cols, figsize=(16, 16))

    diff = []

    for i in range(rows):
        mean_angle_change_w_others = 0
        var_angle_change_w_others = 0

        angle_change_w_self = 0
        var_angle_change_w_self = 0

        angle_change_w_back = 0
        var_angle_change_w_back = 0

        for j in range(cols):
            pair_matrix = np.array(np.meshgrid(positions[i + 1], positions[j + 1]))
            submatrix = cos_sim_matr[pair_matrix[0], pair_matrix[1]]

            mean = np.mean(submatrix)
            var = np.var(submatrix)

            if visualize:
                img = axes[i][j].imshow(submatrix)

            if i == j:
                angle_change_w_self = mean
                var_angle_change_w_self = var

            elif j == cols - 1:
                angle_change_w_back = mean
                var_angle_change_w_back = var

            else:
                mean_angle_change_w_others += mean
                var_angle_change_w_others += var

            if visualize:
                axes[i][j].set_title(
                    (
                        f"между классами {i+1} и {j+1}"
                        if i != j
                        else f"внутри класса {i+1}"
                    )
                    + f"\n{mean:.3f} ± {var:.3f}"
                )

                fig.colorbar(img, ax=axes[i][j], fraction=0.05, pad=0.25)

        mean_angle_change_w_others /= (cols - 2)
        var_angle_change_w_others /= (cols - 2)

        diff.append([
            i + 1,
            class_names[i + 1],
            angle_change_w_self,
            var_angle_change_w_self,
            mean_angle_change_w_others,
            var_angle_change_w_others,
            angle_change_w_back,
            var_angle_change_w_back,
            angle_change_w_self - mean_angle_change_w_others,
            angle_change_w_self - angle_change_w_back,
        ])

    if visualize:
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if type == "diff":
            title = (
                f"Изменение углов при {'свёртке' if operation == 'conv' else 'центрировании'}.\n"
                "последний класс - патч фона"
            )
            save_name = (
                f"04. angles_diff_{operation}.png"
            )
        else:
            title = (
                f"Углы после {'свёртки' if operation == 'conv' else 'центрирования'}.\n"
                "последний класс - патч фона"
            )
            save_name = (
                f"04. angles_after_{operation}.png"
            )

        plt.suptitle(title)

        if savefig:
            plt.savefig(save_name)

        scrollable_plot(fig)

    headers = (
        [
            "Класс",
            "Лэйбл",
            "Изменение угла внутри",
            "Среднее изменение угла\n с остальными",
            "Изменение угла с фоном",
            "Разница\nс другими классами",
            "Разница\nс фоном",
        ]
        if type == "diff"
        else
        [
            "Класс",
            "Лэйбл",
            "Угол внутри",
            "Средний угол\n с остальными",
            "Угол с фоном",
            "Разница\nс другими классами",
            "Разница\nс фоном",
        ]
    )
    display_diff = [
        [
            row[0],
            row[1],
            f"{row[2]:.3f} ± {row[3]:.3f}",
            f"{row[4]:.3f} ± {row[5]:.3f}",
            f"{row[6]:.3f} ± {row[7]:.3f}",
            f"{row[8]:.3f}",
            f"{row[9]:.3f}",
        ]
        for row in diff
    ]

    display_diff[-1][0] = "Фон"
    display_diff[-1][-1] = "0"

    print(tabulate(display_diff, headers=headers, tablefmt="grid"))

    mean_row = [
        "Среднее",
        "",
        f"{np.mean([x[2] for x in diff[:-1]]):.3f}",
        f"{np.mean([x[4] for x in diff[:-1]]):.3f}",
        f"{np.mean([x[6] for x in diff[:-1]]):.3f}",
        f"{np.mean([x[8] for x in diff[:-1]]):.3f}",
        f"{np.mean([x[9] for x in diff[:-1]]):.3f}",
    ]

    print()
    print(tabulate([mean_row], headers=headers, tablefmt="grid"))


def plot_angles_all_model(
    inside,
    outside,
    back,
    type,          # while | after
    operation,     # conv | center
    bns_fwd_names,
    savefig
):

    file_path = './architecture_all_deploy.txt'
    connections = do_the_magic(file_path)

    x = np.arange(0, len(inside))

    fig, ax = plt.subplots(figsize=(16, 8))

    ax.plot(x, inside, marker='o', label='inside', linestyle='-')
    ax.plot(x, outside, marker='o', label='outside', linestyle='-')
    ax.plot(x, back, marker='o', label='back', linestyle='-')

    draw_bn_dependency_graph_inline(ax, bns_fwd_names[2:], connections, -0.5)
    plt.xticks(x, [str(i + 3) for i in x], rotation=90)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.ylim(-1, 1)
    # plt.ylim(-0.7, 0.7)
    plt.legend()
    plt.grid(True)
    if type == "after":
        plt.suptitle(f"Angles after {operation}")
        if savefig: plt.savefig(f"05. Angles after {operation}.png")
    else:
        plt.suptitle(f"Angle change while {operation}")
        if savefig: plt.savefig(f"05. Angle change while {operation}.png")
    plt.show();


def plot_class_specific_angles_all_model(
    inside_np,
    outside_np,
    back_np,
    type,          # while | after
    operation,     # conv | center
    class_names,
    savefig
):
    # ==== Параметры ====
    x = np.arange(3, 48)
    width = 1600
    height = 800
    ylim = [-1, 1]


    # ==== Цветовые палитры ====
    def get_adjusted_colorscale(name, n_colors, min_val=0.3, max_val=1.0):
        return pc.sample_colorscale(name, np.linspace(min_val, max_val, n_colors))

    palette_inside = get_adjusted_colorscale('Blues', 5)
    palette_outside = get_adjusted_colorscale('Reds', 5)
    palette_back = get_adjusted_colorscale('Greens', 5)


    # ==== Построение ====
    fig = go.Figure()

    for i, class_name in enumerate(list(class_names.values())[:-1]):
        fig.add_trace(go.Scatter(
            x=x,
            y=inside_np[:, i],
            mode='lines+markers',
            name=f'inside: {class_name}',
            line=dict(color=palette_inside[i]),
            legendgroup='inside',
            hovertemplate=f'inside: {class_name}<br>x=%{{x}}<br>y=%{{y}}'
        ))
        fig.add_trace(go.Scatter(
            x=x,
            y=outside_np[:, i],
            mode='lines+markers',
            name=f'outside: {class_name}',
            line=dict(color=palette_outside[i]),
            legendgroup='outside',
            hovertemplate=f'outside: {class_name}<br>x=%{{x}}<br>y=%{{y}}'
        ))
        fig.add_trace(go.Scatter(
            x=x,
            y=back_np[:, i],
            mode='lines+markers',
            name=f'back: {class_name}',
            line=dict(color=palette_back[i]),
            legendgroup='back',
            hovertemplate=f'back: {class_name}<br>x=%{{x}}<br>y=%{{y}}'
        ))

    title = f"Angles after {operation}" if type == "after" else f"Angle change while {operation}"
    savename = f"06. Class specific angles after {operation}.html" if type == "after" else f"06. Class specific angle change while {operation}.html"

    # ==== Оформление ====
    fig.update_layout(
        title=title,
        hovermode='x',
        legend=dict(
            title='Классы и источники',
            itemclick='toggleothers',
            itemdoubleclick='toggle'
        ),
        xaxis=dict(
            title='X',
            tickangle=90
        ),
        yaxis=dict(
            title='Значение',
            range=ylim
        ),
        width=width,
        height=height,
        template='plotly_white'
    )

    fig.show()


    # ==== Сохранение ====
    if savefig: fig.write_html(savename)