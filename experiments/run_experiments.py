import os
import csv
import itertools
import random
from train import train_model


# --------------------------------
# SEARCH SPACE
# --------------------------------

patch_sizes = [8, 16]

embedding_dims = [128, 256]

heads_options = [4,8]

depth_options = [4, 8]

mlp_dims = [256, 512]


# --------------------------------
# RESULTS FILE
# --------------------------------

BASE_DIR = "/content/drive/MyDrive/vit_experiments"

RESULTS_FILE = os.path.join(BASE_DIR, "results.csv")

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

BEST_MODEL_DIR = os.path.join(BASE_DIR, "best_models")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)


# --------------------------------
# CREATE RESULTS FILE IF MISSING
# --------------------------------

if not os.path.exists(RESULTS_FILE):

    with open(RESULTS_FILE, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(
            ["patch", "emb_dim", "heads", "depth", "mlp_dim", "valaccuracy", "valloss"]
        )


# --------------------------------
# LOAD COMPLETED EXPERIMENTS
# --------------------------------

completed = set()

with open(RESULTS_FILE, "r") as f:

    reader = csv.DictReader(f)

    for row in reader:

        key = (
            int(row["patch"]),
            int(row["emb_dim"]),
            int(row["heads"]),
            int(row["depth"]),
            int(row["mlp_dim"]),

        )

        completed.add(key)


# --------------------------------
# GENERATE GRID
# --------------------------------

all_configs = list(
    itertools.product(
        patch_sizes,
        embedding_dims,
        heads_options,
        depth_options,
        mlp_dims
    )
)

configs = random.sample(all_configs,12)
# --------------------------------
# RUN EXPERIMENTS
# --------------------------------

for config in configs:

    patch, emb, heads, depth, mlp = config

    key = (patch, emb, heads, depth, mlp)

    if key in completed:

        print(f"Skipping completed experiment {key}")

        continue

    print("\n================================")
    print(f"Running experiment {key}")
    print("================================\n")

    accuracy,loss = train_model(
        patch_size=patch,
        emb_dim=emb,
        heads=heads,
        depth=depth,
        mlp_dim=mlp,
        CHECKPOINT_DIR=CHECKPOINT_DIR,
        BEST_MODEL_DIR=BEST_MODEL_DIR,
        config=key
    )

    with open(RESULTS_FILE, "a", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(
            [patch, emb, heads, depth, mlp, accuracy, loss]
        )

    print(f"Experiment {key} finished with val accuracy {accuracy} and val loss {loss}")
