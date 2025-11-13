import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from run_train_model import main

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_noise_data(num_images=100, image_size = (20, 20)):
    num_labels = 14
    label_columns = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
        'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',
        'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
    ]

    os.makedirs('train2_subset', exist_ok=True)

    data = []

    for i in range(num_images):
        img_array = np.random.randint(0, 256, size=(image_size[0], image_size[1]), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')  # grayscale
        filename = f'img_{i:03d}.jpg'
        img.save(os.path.join('train2_subset', filename))

        labels = np.random.randint(0, 2, size=(num_labels,)).tolist()

        data.append([filename] + labels)

    columns = ['Image_name'] + label_columns
    df = pd.DataFrame(data, columns=columns)
    df.to_csv('train2_subset/train2.csv', index=False)


def test_model():

    generate_noise_data()

    try:
        main(is_test=True)
    except Exception as e:
        print(f"Error while running model: {e}")
        raise

    print("Model ran successfully.")

if __name__ == "__main__":
    test_model()


