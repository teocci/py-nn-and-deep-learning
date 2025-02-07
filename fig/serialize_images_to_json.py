"""
serialize_images_to_json
~~~~~~~~~~~~~~~~~~~~~~~~

Utility to serialize parts of the training and validation data to JSON, 
for use with Javascript.
"""

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

# My library
import src.mnist_loader as mnist_loader

# Number of training and validation data images to serialize
NTD = 1000
NVD = 100

JSON_STRUCTURE = {
    "training": [],
    "validation": []
}


def process_mnist_data(
        training_data: List[Tuple[np.ndarray, np.ndarray]],
        validation_data: List[Tuple[np.ndarray, np.ndarray]],
        num_training: int = NTD,
        num_validation: int = NVD,
        output_path: str = f"data-{NTD}-{NVD}.json",
) -> None:
    """
    Process MNIST data and save it to a JSON file with optimized memory usage and performance.

    Args:
        training_data: List of tuples containing training images and labels
        validation_data: List of tuples containing validation images and labels
        num_training: Number of training samples to process
        num_validation: Number of validation samples to process
        output_path: Path to save the output JSON file
    """
    # Pre-allocate lists for better memory efficiency
    processed_data = JSON_STRUCTURE.copy()

    tlen = len(training_data)
    vlen = len(validation_data)

    # Process training data
    processed_data["training"] = [
        {
            "x": training_data[i][0].ravel().tolist(),  # Flatten array directly
            "y": training_data[i][1].ravel().tolist()  # Flatten labels
        }
        for i in range(min(num_training, tlen))
    ]

    # Process validation data
    processed_data["validation"] = [
        {
            "x": validation_data[i][0].ravel().tolist(),
            "y": int(validation_data[i][1])
        }
        for i in range(min(num_validation, vlen))
    ]

    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save with optimized JSON settings
    with output_file.open('w') as f:
        json.dump(processed_data, f)


def main(output_path: str = "data_1000.json"):
    training_data, validation_data, _ = mnist_loader.load_data_wrapper()

    # Process and save data
    process_mnist_data(
        training_data=training_data,
        validation_data=validation_data,
        output_path=output_path
    )


if __name__ == "__main__":
    main()
