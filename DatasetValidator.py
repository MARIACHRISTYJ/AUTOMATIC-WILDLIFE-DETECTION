from collections import Counter
from Train import train_data

labels = [label for _, label in train_data.samples]
class_counts = Counter(labels)

for class_idx, count in class_counts.items():
    print(f"{train_data.classes[class_idx]}: {count} images")
