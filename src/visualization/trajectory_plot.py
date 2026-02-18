import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.cm as cm
import matplotlib.lines as mlines


JSON_PATH = "outputs/tracking_field_coords.json"
OUTPUT_PATH = "outputs/trajectory_plot.png"

FIELD_LENGTH = 105
FIELD_WIDTH = 68

MIN_FRAMES = 120
TOP_N_TRACKS = 10
SMOOTHING_WINDOW = 5


def smooth(coords, window=5):
    if len(coords) < window:
        return coords
    coords = np.array(coords)
    smoothed = []
    for i in range(len(coords)):
        start = max(0, i - window)
        end = min(len(coords), i + window)
        smoothed.append(np.mean(coords[start:end], axis=0))
    return np.array(smoothed)


def calculate_distance(coords):
    coords = np.array(coords)
    diffs = np.diff(coords, axis=0)
    return np.sum(np.sqrt((diffs ** 2).sum(axis=1)))


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


def plot_trajectories():

    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    tracks = defaultdict(list)

    for frame in data:
        for obj in frame["field_tracks"]:
            track_id = obj["track_id"]
            x, y = obj["field_pos"]
            tracks[track_id].append((x, y))

    tracks = {
        tid: coords
        for tid, coords in tracks.items()
        if len(coords) >= MIN_FRAMES
    }

    sorted_tracks = sorted(
        tracks.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:TOP_N_TRACKS]

    fig, ax = plt.subplots(figsize=(12, 8))
    draw_pitch(ax)

    cmap = cm.get_cmap("tab10", len(sorted_tracks))

    legend_handles = []

    for idx, (track_id, coords) in enumerate(sorted_tracks):
        coords = smooth(coords, SMOOTHING_WINDOW)
        coords = np.array(coords)

        color = cmap(idx)
        distance = calculate_distance(coords)

        ax.plot(coords[:, 0],
                coords[:, 1],
                linewidth=2,
                color=color)

        # MATCHED COLOR LEGEND
        legend_handles.append(
            mlines.Line2D([], [],
                          color=color,
                          label=f"ID {track_id} ({distance:.1f}m)")
        )

    ax.legend(handles=legend_handles,
              loc="upper right",
              fontsize=9)

    os.makedirs("outputs", exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300)
    plt.close()

    print("✅ Trajectory plot saved:", OUTPUT_PATH)


if __name__ == "__main__":
    plot_trajectories()
