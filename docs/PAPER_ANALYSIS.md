# Comprehensive Analysis for Top-Tier Publication\n\nThis analysis evaluates multiple architectural parameters and regularization strategies (e.g., prototype representation, classification loss functions, distance penalties) to identify optimal combinations for domain-generalized disease classification.\n\n## 0. Exploratory Data Analysis\nSee `0_dataset_eda.png` for a breakdown of class distributions and dataset representation.\n\n## 1. Top 20 Best Performing Configurations\n| model_name   | classif_loss        | prototypes       | dloss          | fgsm   |      mcc |   accuracy |
|:-------------|:--------------------|:-----------------|:---------------|:-------|---------:|-----------:|
| resnet18     | inverseTriplet      | prototypes_no    | dist_euclidean | fgsm0  | 0.866025 |   0.933333 |
| resnet18     | arcface             | prototypes_class | dist_none      | fgsm0  | 0.707107 |   0.866667 |
| resnet18     | softmax_contrastive | prototypes_no    | dist_cosine    | fgsm1  | 0.707107 |   0.866667 |
| resnet18     | triplet             | prototypes_class | dist_cosine    | fgsm1  | 0.7      |   0.866667 |
| resnet18     | softmax_contrastive | prototypes_no    | dist_cosine    | fgsm0  | 0.7      |   0.866667 |
| resnet18     | triplet             | prototypes_no    | dist_cosine    | fgsm1  | 0.57735  |   0.8      |
| resnet18     | inverseTriplet      | prototypes_no    | dist_cosine    | fgsm1  | 0.57735  |   0.8      |
| resnet18     | inverseTriplet      | prototypes_no    | dist_cosine    | fgsm1  | 0.57735  |   0.733333 |
| resnet18     | inverseTriplet      | prototypes_no    | dist_euclidean | fgsm0  | 0.57735  |   0.733333 |
| resnet18     | arcface             | prototypes_class | dist_none      | fgsm0  | 0.57735  |   0.733333 |
| resnet18     | inverseTriplet      | prototypes_no    | dist_cosine    | fgsm0  | 0.57735  |   0.8      |
| resnet18     | inverseTriplet      | prototypes_class | dist_cosine    | fgsm0  | 0.57735  |   0.733333 |
| resnet18     | arcface             | prototypes_class | dist_none      | fgsm0  | 0.57735  |   0.8      |
| resnet18     | inverseTriplet      | prototypes_no    | dist_cosine    | fgsm1  | 0.57735  |   0.733333 |
| resnet18     | arcface             | prototypes_class | dist_none      | fgsm0  | 0.57735  |   0.733333 |
| resnet18     | inverseTriplet      | prototypes_class | dist_euclidean | fgsm0  | 0.57735  |   0.733333 |
| resnet18     | arcface             | prototypes_class | dist_none      | fgsm0  | 0.57735  |   0.8      |
| resnet18     | arcface             | prototypes_class | dist_none      | fgsm0  | 0.57735  |   0.8      |
| resnet18     | inverseTriplet      | prototypes_no    | dist_cosine    | fgsm0  | 0.5547   |   0.8      |
| resnet18     | inverseTriplet      | prototypes_no    | dist_cosine    | fgsm0  | 0.533002 |   0.8      |\n\n## 2. Key Ablations (Refer to attached figures)\n- **Architecture Comparison**: Demonstrates the raw capability of backbone feature extractors. See `1_architecture_mcc.png`.\n- **Loss Formulation**: ArcFace and Triplet margin strategies heavily dominate standard cross-entropy due to superior representation space constraint. See `2_loss_ablation.png`.\n- **Prototype Strategy**: Evaluates sub-center cluster mapping. 'Class' based prototypes provide global semantic alignment. See `3_prototype_ablation.png`.\n- **Domain Generalization Mapping**: Maximizing predictive power (MCC) without heavily biasing toward batch attributes (Batch Entropy). Optimal algorithms occupy the top-right quadrant. See `4_mcc_vs_entropy.png`.\n- **Interpretability**: Demonstrates clinical utility by comparing Grad-CAM activations alongside SHAP value gradients (see `6_interpretability.png`). Includes both correct and misclassified case studies.\n