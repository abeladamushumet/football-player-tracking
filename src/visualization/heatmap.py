import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.ndimage import gaussian_filter

JSON_PATH = "outputs/tracking_field_coords.json"
OUTPUT_DIR = "outputs/heatmaps"

FIELD_LENGTH = 105
FIELD_WIDTH = 68

MIN_FRAMES = 200     # filter short noisy tracks
TOP_N = 5            # top stable players


def draw_pitch(ax):
    ax.set_facecolor("#3f995b")

    ax.plot([0, FIELD_LENGTH], [0, 0], color="white")
    ax.plot([0, FIELD_LENGTH], [FIELD_WIDTH, FIELD_WIDTH], color="white")
    ax.plot([0, 0], [0, FIELD_WIDTH], color="white")
    ax.plot([FIELD_LENGTH, FIELD_LENGTH], [0, FIELD_WIDTH], color="white")

    ax.plot([FIELD_LENGTH / 2, FIELD_LENGTH / 2],
            [0, FIELD_WIDTH],
            color="white")

    center_circle = plt.Circle(
        (FIELD_LENGTH / 2, FIELD_WIDTH / 2),
        9.15,
        fill=False,
        color="white"
    )
    ax.add_patch(center_circle)

    ax.set_xlim(0, FIELD_LENGTH)
    ax.set_ylim(0, FIELD_WIDTH)
    ax.set_aspect("equal")


def generate_heatmaps():

    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    tracks = defaultdict(list)

    for frame in data:
        for obj in frame["field_tracks"]:
            tid = obj["track_id"]
            x, y = obj["field_pos"]
            tracks[tid].append((x, y))

    # filter stable tracks
    tracks = {
        tid: coords
        for tid, coords in tracks.items()
        if len(coords) >= MIN_FRAMES
    }

    # sort by track length
    sorted_tracks = sorted(
        tracks.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:TOP_N]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for tid, coords in sorted_tracks:

        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]

        heatmap, xedges, yedges = np.histogram2d(
            xs,
            ys,
            bins=[80, 50],
            range=[[0, FIELD_LENGTH], [0, FIELD_WIDTH]]
        )

        heatmap = gaussian_filter(heatmap, sigma=3)

        # normalize
        heatmap = heatmap / np.max(heatmap)

        fig, ax = plt.subplots(figsize=(12, 8))
        draw_pitch(ax)

        ax.imshow(
            heatmap.T,
            extent=[0, FIELD_LENGTH, 0, FIELD_WIDTH],
            origin="lower",
            alpha=0.8
        )

        plt.title(f"Heatmap - Player ID {tid}")

        save_path = os.path.join(OUTPUT_DIR, f"heatmap_ID_{tid}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"🔥 Saved heatmap for ID {tid}")


if __name__ == "__main__":
    generate_heatmaps()
