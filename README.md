# Otitenet

Use python3.11

python -m pip install -r requirements.txt

python setup.py

make_dataset2.py

train_triplet_new.py

# MLFLOW

ssh -L 8502:localhost:8502 simon@198.168.189.19

streamlit run app.py --server.address 0.0.0.0 --server.port 8502

# make new db
python init_db.py

sudo mysql -u root -p
-- Create the database
CREATE DATABASE results_db;

-- Grant privileges to the user
GRANT ALL PRIVILEGES ON results_db.* TO 'y_user'@'%';

-- Set the user's password
ALTER USER 'y_user'@'%' IDENTIFIED BY 'password';

-- Apply the changes
FLUSH PRIVILEGES;

## Dataset Preprocessing & Transformations (`make_dataset2.py`)

This script processes raw image datasets and creates a unified dataset folder with resized images and a CSV metadata file.

**Transformations applied:**
- **Resize:** All images are resized to a square of the specified size (e.g., 224×224 pixels) using `torchvision.transforms.Resize`.
- **Format conversion:** Images are loaded and saved in their original format (JPG or PNG), but all are resized and copied to the new dataset folder.
- **Filtering:** Only images with the correct extension (`.jpg` or `.png`) and not containing 'Off'/'off' in the filename are included.
- **Stratified split:** The script uses `StratifiedKFold` to assign each image to either the train or test group, ensuring balanced label distribution.

**Output:**
- Resized images in `data/otite_ds_<size>/`
- Metadata CSV: `data/otite_ds_<size>/infos.csv` with columns: `dataset`, `name`, `label`, `group`

## Streamlit App Logic (`app.py`)

```mermaid
flowchart TD
    A[Start: User opens app.py] --> B{User logged in?}
    B -- No --> C[Show login form]
    C -->|Login| B
    B -- Yes --> D[Show sidebar: select person, add/remove person, delete account, remove result]
    D --> E{Person selected?}
    E -- No --> F[Show warning: select a family member]
    E -- Yes --> G[Show results table for person]
    G --> H{File uploaded or result selected?}
    H -- No --> I[Wait for user action]
    H -- Yes --> J[Show analysis UI]
    J --> K{Run Analysis button pressed?}
    K -- No --> I
    K -- Yes --> L[Get model args from sidebar]
    L --> M[Check if result exists in DB]
    M -- Exists --> N[Show previous results, allow re-analysis]
    M -- Not exists --> O[Load model, prototypes, data]
    O --> P[Save uploaded file]
    P --> Q[Run model inference, get predictions]
    Q --> R[Log SHAP/gradient images]
    R --> S[Insert results into DB]
    S --> T[Show results, images, metrics]
    T --> I
```

## Training Logic (`train_triplet_new.py`)

```mermaid
flowchart TD
    A[Start: Script/Module Entry]
    A --> B[Parse args, set up paths]
    B --> C[Initialize TrainAE class]
    C --> D[Load and preprocess data]
    D --> E[Set up model, loss, optimizer]
    E --> F[Repeat for n_repeats]
    F --> G[Split data for train/valid/test]
    G --> H[Epoch loop]
    H --> I[Train model on train set]
    I --> J[Validate on valid set]
    J --> K{Early stopping?}
    K -- Yes --> L[Break epoch loop]
    K -- No --> H
    J --> M{Best MCC?}
    M -- Yes --> N[Save model, prototypes, weights]
    M -- No --> H
    N --> O[Update best metrics]
    O --> H
    F --> P[After training: clustering, visualization, logging]
    P --> Q[Return best MCC]
```
