from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from minisom import MiniSom
from sklearn.cluster import KMeans


project_dir = Path(__file__).resolve().parents[1]
scalogram_dir = project_dir / "Scalogram Images"
result_dir = project_dir / "Results"
result_dir.mkdir(parents=True, exist_ok=True)

files = sorted(scalogram_dir.glob("*.png"))
if not files:
    raise FileNotFoundError(f"No scalogram images found in {scalogram_dir}")

images = []
for file in files:
    image = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read scalogram image: {file}")
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    images.append(resized.astype(np.float32).reshape(-1) / 255.0)

images_norm = np.asarray(images, dtype=np.float32)

som_shape = (32, 32)
som = MiniSom(som_shape[0], som_shape[1], images_norm.shape[1], sigma=1.0, learning_rate=0.5)
som.train_random(images_norm, 64)

neuron_clusters = np.asarray([som.winner(image) for image in images_norm])
kmeans = KMeans(n_clusters=6, n_init=10, random_state=42)
image_clusters = kmeans.fit_predict(neuron_clusters)

value_counts = np.bincount(image_clusters, minlength=6)
x = np.arange(len(value_counts))
plt.bar(x, value_counts)
plt.xticks(x)
plt.xlabel("Cluster")
plt.ylabel("No. of Samples")
plt.title("Severity using Clustering")
plt.savefig(result_dir / "severity_kmeans_run.png", dpi=120, bbox_inches="tight")
plt.close()

print(f"Clustered {len(files)} scalograms; results saved to {result_dir / 'severity_kmeans_run.png'}")
