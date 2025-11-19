import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

# Paths
input_folder = "images"
output_folder = "processed_images"

# Create output root if not exists
os.makedirs(output_folder, exist_ok=True)

# Desired image size for CNN
IMAGE_SIZE = (270, 360)

def process_image(paths):
    input_path, output_path = paths
    try:
        with Image.open(input_path) as img:
            img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
            img.save(output_path, "JPEG")
    except Exception as e:
        print(f"Skipping {input_path}: {e}")

def parallel_process(input_folder, output_folder, workers=4):
    files = [
        (os.path.join(input_folder, f), os.path.join(output_folder, f))
        for f in os.listdir(input_folder)
    ]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        list(executor.map(process_image, files))

parallel_process(input_folder, output_folder, workers=14)
