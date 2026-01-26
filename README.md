This project presents a convolutional neural network (CNN) model for lung disease classification from medical images.

## Model

The classification model is located in the `src/` directory.
It outputs probabilities for 14 different lung diseases.

The model is based on **EfficientNetV2**.
All layers were fine-tuned during training.

On the test dataset, the model achieves a ROC-AUC score of 91%.

## Architecture Experiments

To select the best architecture, multiple experiments were conducted.
Training was performed on smaller datasets using different model architectures.

These experiments are documented in the notebook:

```
notebooks/experiments.ipynb
```

## Model Explainability

To interpret the model’s predictions, Grad-CAM and DeepLIFT methods were studied and compared.

DeepLIFT was modified for comparison purposes.

The interpretation methods produce different attribution maps:

* **DeepLIFT** generates more sparse explanations
* **Grad-CAM** highlights larger regions of the image

