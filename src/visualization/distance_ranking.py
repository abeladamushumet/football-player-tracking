import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

JSON_PATH = "outputs/tracking_field_coords.json"
OUTPUT_PATH = "outputs/distance_ranking.png"


def calculate_distance(coords):
    coords = np.array(coords)
    diffs = np.diff(coords, axis=0)
    return np.sum(np.sqrt((diffs ** 2).sum(axis=1)))


def generate_ranking():

    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    tracks = defaultdict(list)

    for frame in data:
        for obj in frame["field_tracks"]:
            track_id = obj["track_id"]
            x, y = obj["field_pos"]
            tracks[track_id].append((x, y))

    distances = {}

    for tid, coords in tracks.items():
        if len(coords) > 300:
            distances[tid] = calculate_distance(coords)

    sorted_dist = sorted(
        distances.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    ids = [str(x[0]) for x in sorted_dist]
    values = [x[1] for x in sorted_dist]

    plt.figure(figsize=(10, 6))
    plt.bar(ids, values)
    plt.xlabel("Track ID")
    plt.ylabel("Distance (meters)")
    plt.title("Top 10 Distance Covered")

    plt.savefig(OUTPUT_PATH, dpi=300)
    plt.close()

    print("📊 Distance ranking saved:", OUTPUT_PATH)


if __name__ == "__main__":
    generate_ranking()
