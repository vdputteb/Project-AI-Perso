import json
import os

brains = []

if __name__ == "__main__":
    directories = os.listdir("./brains/")

    for dir in directories:
        filepath = f"./brains/{dir}"
        with open(filepath, "r", encoding="utf-8") as file:

            data = json.load(file)
            brains.append(data)  # garder le dict complet

    # Moyenne des poids
    middle = []
    num_brains = len(brains)
    num_weights = len(brains[0]["weights"])

    for i in range(num_weights):
        middle_weight = 0
        for brain in brains:
            middle_weight += brain["weights"][i]
        middle.append(middle_weight / num_brains)

    # Chemin de sortie
    path = "./brains/average.json"

    result = {
        "layer_sizes": brains[0]["layer_sizes"],
        "weights": middle,
        "lap_frames": 0,
        "lap_seconds": None,
    }

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f)
    os.replace(tmp, path)