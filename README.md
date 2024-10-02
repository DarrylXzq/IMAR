# Wheat Disease Detection with Improved MobileNet and Inception Model
<div align="center">
  <img src="https://github.com/user-attachments/assets/2f92a2ce-a2d6-444b-80d6-d1552af88ff3" width="200">
</div>
<br/>
<div align="center">
  <img src="https://img.shields.io/badge/-Python-blue.svg">
  <img src="https://img.shields.io/badge/-Conda-orange.svg">
  <img src="https://img.shields.io/badge/-TensorFlow-orange.svg">
  <img src="https://img.shields.io/badge/-Numpy-blue.svg">
  <img src="https://img.shields.io/badge/-Pandas-blue.svg">
  <img src="https://img.shields.io/badge/-Matplotlib-blue.svg">
  <img src="https://img.shields.io/badge/-Seaborn-lightblue.svg">
  <img src="https://img.shields.io/badge/-Jupyter-lightgrey.svg">
  <img src="https://img.shields.io/badge/-CV2-lightgrey.svg">
</div>

> [!IMPORTANT]
> This project uses the [`Kaggle`](https://www.kaggle.com/datasets/olyadgetch/wheat-leaf-dataset) wheat disease dataset. Through `stratified sampling`, `data augmentation`, `ensemble modeling`, `performance evaluation`, and `visual analysis`, I have designed and optimized a wheat disease detection model, which is then compared with `common pre-trained models`.

## Dataset Split
1. **Download the dataset**:
    - Download the wheat disease dataset from [`Kaggle`](https://www.kaggle.com/datasets/olyadgetch/wheat-leaf-dataset).
    - Observe the dataset, noting `significant differences in the count of each wheat disease category`.
2. **Stratified Sampling**:
    - Use the `stratified sampling` approach to sample the dataset.
    - Split the dataset into **train**, **validation**, and **test** sets in a ratio of **70%: 15%: 15%**.
    - Ensure that the data split maintains the same category distribution across the three sets.

## Data Augmentation
1. After splitting, ensure the training set contains an equal number of samples for each category to avoid model bias.
2. Apply data augmentation techniques, including:
    - CLAHE (***Contrast Limited Adaptive Histogram Equalization***)
    - Bilateral Filter operation
    - Histogram Equalization operation
    - RGB transformations
3. Ensure an equal number of samples for each wheat category in the training set.
<div align="left">
  <img src="https://github.com/user-attachments/assets/14df6e3a-718d-4540-8de2-7355a0a3b565" width="600">
</div>

## Custom Loss and Accuracy Functions
- Created `custom loss` and `accuracy functions` to improve model performance on the imbalanced dataset.

>[!TIP]
>- `Focal loss`
<div align="left">
  <img src="https://github.com/user-attachments/assets/e46ff2c6-33c7-49f1-a5ad-5d520606ebd0" width="600">
</div>

>[!TIP]
>- `Accuracy functions`
<div align="left">
  <img src="https://github.com/user-attachments/assets/ea36cd94-67b6-4bf3-ac7e-f4cae60cdeb0" width="350">
</div>

## Model Design
> [!IMPORTANT]
> - Use improved ***MobileNet*** and improved ***Inception*** models for ensemble.
> - Introduce **residual blocks** and **SE-block attention mechanisms** in these models to enhance performance.
> - `Figure 1`shows the improved Inception architecture, and `Figure 2` shows the integration model that includes improved Inception and MobileNet
<div align="left">
  <img src="https://github.com/user-attachments/assets/bcf3bed7-4023-45ea-bf9b-efceed661337" width="500">
  <img src="https://github.com/user-attachments/assets/97082b3c-a35e-4bf9-856c-aa6f4837079a" height="400">
</div>

## Performance Evaluation
> [!NOTE]
> - Use ***focal loss*** as the loss function and custom weighted accuracy functions.
> - Validate and test the models using multiple metrics:
>   - **Loss**
>   - **Accuracy**
>   - **Recall**
>   - **Precision**
>   - **F1-Score**
>   - **Specificity**
>   - **Sensitivity**
>   - **MCC (Matthew's correlation coefficient)**
>   - **Confusion Matrix**
>   - **PR Curve**
>   - **ROC Curve**
> - Select the best performing model.

## Grad-CAM Visualization
>[!NOTE]
>- Use ***Grad-CAM*** for visual analysis of the best performing model.
>- Display model attention regions to verify if the model identifies correct classification features.
<div align="left">
  <img src="https://github.com/user-attachments/assets/38a600e5-9754-44e5-b2bc-57d4e2f57901" width="600">
</div>


## Model Comparison
>[!NOTE]
>- Compare the designed model against common pre-trained models:
>   - **VGG16**
>   - **InceptionV3**
>   - **MobileNet**
>   - **ResNet50**
>- Also compare against models proposed by other authors to evaluate if the **IMAR model** (our model) outperforms them on this dataset.

<div align="left">
   <img src="https://github.com/user-attachments/assets/94650256-f3c3-448d-a742-4c4d1e47a518" width="600">
</div>

## Project Structure

| File | Description |
|------|-------------|
| `code/` | Jupyter Notebooks |
| `code/Code/completed_model_ensemble.ipynb` | Ensemble model |
| `Model_architecture/` | Model Architecture diagram |
| `README.md` | Project information and instructions |

## Results
The **IMAR model** achieved superior performance on the wheat disease detection task compared to traditional pre-trained models. Please refer to the experiment report in this project for detailed results.
| Model        | Loss  | Accuracy | Recall | Precision | F1    | Specificity | Sensitivity | MCC   |
|--------------|-------|----------|--------|-----------|-------|-------------|-------------|-------|
| VGG16        | 0.3127 | 82.54%   | 82.54% | 82.96%    | 82.62% | 87.51%     | 82.54%      | 72.81%|
| InceptionV3  | 0.4664 | 92.06%   | 92.06% | 92.71%    | 92.09% | 95.10%     | 92.06%      | 88.06%|
| MobileNet    | 0.3279 | 85.71%   | 85.71% | 87.49%    | 86.00% | 91.84%     | 85.71%      | 79.26%|
| ResNet50     | 0.8198 | 69.84%   | 69.84% | 69.84%    | 69.84% | 82.31%     | 69.84%      | 50.93%|
| IMAR         | 0.0997 | 98.41%   | 98.41% | 98.51%    | 98.42% | 99.46%     | 98.41%      | 97.62%|

