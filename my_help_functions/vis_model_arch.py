import re
from collections import defaultdict, deque


def clean_label(label):
    # # Удаляем всё после скобок (если есть)
    # label = re.sub(r'\s*\(.*?\)', '', label)
    # return label.strip()
    return label.split()[0].strip('"')


def parse_graphviz_file(file_path):
    edges = []  # Список ребер (id_from, id_to)
    labels = {}  # id -> label

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            edge_match = re.match(r'(\d+)\s*->\s*(\d+)', line)
            if edge_match:
                src, dst = edge_match.groups()
                edges.append((src, dst))
                continue

            label_match = re.match(r'(\d+)\s*\[label=(.+?)\]', line)
            if label_match:
                node_id, raw_label = label_match.groups()
                clean = clean_label(raw_label)
                labels[node_id] = clean

    return edges, labels


def build_graph(edges):
    forward = defaultdict(list)
    backward = defaultdict(list)
    for src, dst in edges:
        forward[src].append(dst)
        backward[dst].append(src)
    return forward, backward


def find_batchnorm_names(backward, labels):
    batchnorm_map = {}

    for node_id, label in labels.items():
        if label != 'NativeBatchNormBackward0':
            continue

        # Ищем соседей назад (в AccumulateGrad)
        acc_nodes = backward.get(node_id, [])
        for acc in acc_nodes:
            if labels.get(acc) != 'AccumulateGrad':
                continue

            # Назад от AccumulateGrad, ищем узел с названием
            for candidate in backward.get(acc, []):
                name = labels.get(candidate)
                if name and 'bn' in name:
                    batchnorm_map[node_id] = '.'.join(name.split('.')[:-1])
                    break

            if node_id in batchnorm_map:
                break  # Достаточно одного имени

    return batchnorm_map


def bfs_batchnorm_connections(forward, batchnorm_map):
    result = defaultdict(list)
    id_to_name = {v: k for k, v in batchnorm_map.items()}

    for src_id, src_name in batchnorm_map.items():
        visited = set()
        queue = deque([src_id])

        while queue:
            current = queue.popleft()
            for neighbor in forward.get(current, []):
                if neighbor in visited:
                    continue
                visited.add(neighbor)

                if neighbor in batchnorm_map and neighbor != src_id:
                    result[src_name].append(batchnorm_map[neighbor])
                else:
                    queue.append(neighbor)

    return result


def do_the_magic(file_path):
    edges, labels = parse_graphviz_file(file_path)
    forward, backward = build_graph(edges)
    batchnorm_map = find_batchnorm_names(backward, labels)
    connections = bfs_batchnorm_connections(forward, batchnorm_map)
    connections['neck.merge_6.conv3.bn'] = []
    return connections


def draw_bn_dependency_graph_inline(ax, bns_fwd_names, bn_connections, base_y=-0.5):
    x_coords = {name: i for i, name in enumerate(bns_fwd_names)}
    # base_y = -0.5  # Гарантированно в пределах ylim(-1, 1)

    allowed = set(bns_fwd_names)
    edges = [
        (src, dst) for src, dsts in bn_connections.items() if src in allowed
        for dst in dsts if dst in allowed
    ]
    pos = {name: (x_coords[name], base_y) for name in bns_fwd_names}

    for src, dst in edges:
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        dx = abs(x2 - x1)

        if dx == 1:
            style = "arc3,rad=0.01"
            width = 0.5
            color = "gray"
        else:
            style = f"arc3,rad={0.3 if dx < 5 else 0.5}"
            width = 1.0
            color = "darkblue"

        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", lw=width, color=color, connectionstyle=style)
        )

    highlight_nodes = {
        "neck.merge_7.conv3.bn",
        "neck.merge_6.conv3.bn",
        "neck.merge_5.conv3.bn",
    }

    for name, (x, y) in pos.items():
        from matplotlib.patches import Ellipse
        ax.plot(x, y, 'o', color='black', markersize=4)

        if name in highlight_nodes:
            ax.add_patch(Ellipse((x, y), 0.7, 0.07, fill=False, color='red'))

    # Убираем set_ylim – он есть снаружи
