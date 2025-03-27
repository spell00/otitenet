import streamlit as st
import debugpy
if 'debugger_attached' not in st.session_state:
    try:
        debugpy.listen(5679)
        print("⏳ Waiting for debugger attach on port 5679...")
        debugpy.wait_for_client()
        debugpy.breakpoint()
        st.session_state.debugger_attached = True
    except Exception as e:
        print("❌ debugpy.listen failed:", e)

import os
os.environ["STREAMLIT_SECRETS_LOAD_MODE"] = "read_only"
os.environ["STREAMLIT_FILE_WATCHER_TYPE"] = "none"  # Disable file watching
import torch
import pickle
import mysql.connector
from mysql.connector import Error
import numpy as np
from PIL import Image
from otitenet.train.train_triplet_new import TrainAE  # If needed for params
from otitenet.data.data_getters import GetData, get_images_loaders, get_images  # Update import path
from otitenet.models.cnn import Net, Net_shap  # Update path if needed
from otitenet.utils.utils import get_empty_traces
from otitenet.logging.shap import log_shap_images_gradients
from otitenet.logging.metrics import MCC
from torchvision import transforms
from matplotlib import pyplot as plt

# Your model imports
import argparse

# ---- Load datasets from /data ---- #
data_dir = './data'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- MySQL Database Setup ---- #
def create_db():
    try:
        conn = mysql.connector.connect(
            host="localhost",  # Update with your host
            user="y_user",  # Update with your MySQL username
            password="password",  # Update with your MySQL password
            database="results_db"  # Update with your MySQL database name
        )
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM users LIMIT 1")
        cursor.fetchall()  # Consume the result to avoid "Unread result found"
        return conn, cursor
    except Error as e:
        st.error(f"❌ Create Database error: {e}")
        st.stop()

def get_image(path, size=-1):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.13854676, 0.10721603, 0.09241733], [0.07892648, 0.07227526, 0.06690206]),
    ])
    original = Image.open(path).convert('RGB')
    if size != -1:
        png = transforms.Resize((size, size))(original)
    else:
        png = original
    print(size, png)
    
    return transform(original).unsqueeze(0), transform(png).unsqueeze(0)


def choose_dataset(label, datasets, default=None):
    if len(datasets) == 0:
        st.warning(f"No datasets found in {data_dir}.")
        return None
    elif len(datasets) <= 3:
        return st.radio(label, datasets, index=datasets.index(default) if default in datasets else 0)
    else:
        return st.selectbox(label, datasets, index=datasets.index(default) if default in datasets else 0)


# ---- Database Lookup ---- #
def check_ds_exists(cursor, filename, args):
    cursor.execute(''' 
        SELECT * FROM results
        WHERE filename=%s AND task=%s AND model_name=%s AND path=%s AND nsize=%s AND fgsm=%s AND
              n_calibration=%s AND classif_loss=%s AND dloss=%s AND prototypes=%s AND npos=%s AND nneg=%s
    ''', (filename, args.task, args.model_name, args.path, str(args.new_size), str(args.fgsm),
          str(args.n_calibration), args.classif_loss, args.dloss, args.prototypes_to_use,
          str(args.n_positives), str(args.n_negatives)))
    
    row = cursor.fetchone()  # Consume the result here
    if row:
        st.info("This image was already analyzed with the current model settings.")
        if st.button("Load Previous Results"):
            pred_label = row[14]
            confidence = row[15]
            log_path = row[16]
            st.write(f"**Previous Prediction:** {pred_label} ({confidence:.2f} confidence)")
            fig1 = plt.imread(f"{log_path}/gradients_shap/queries_{filename}_layer5.png")
            st.image(fig1, caption="SHAP Gradient Explanation (layer 5)", use_column_width=True)
            fig2 = plt.imread(f"{log_path}/knn_shap/queries_{filename}_layer5.png")
            st.image(fig2, caption="KNN SHAP Gradient Explanation (layer 5)", use_column_width=True)
            st.stop()
        elif st.button("Re-analyze Image"):
            st.info("Re-analyzing image...")
            return None
        return row
    else:
        st.info("New analysis will be performed.")
        return None

def get_args():
    # ---- User selects paths ---- #
    with st.sidebar:
        st.header("Model Parameters")
        available_datasets = sorted([
            name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))
        ])
        selected_path = choose_dataset(
            "Select --path dataset", available_datasets, default='otite_ds_64'
        )
        # selected_path_original = choose_dataset(
        #     "Select --path_original dataset", available_datasets, default='otite_ds_-1'
        # )

        # ---- Other args with defaults ---- #
        model_name = st.selectbox("Model Name (--model_name)", ["resnet18", "resnet34", "resnet50"], index=0)
        fgsm = st.number_input("fgsm", min_value=0, max_value=1, value=0, step=1)
        n_calibration = st.number_input("n_calibration", value=0, step=1)
        dloss = st.selectbox("dloss", ['no', 'inverseTriplet', 'arcface'], index=0)
        prototypes_to_use = st.selectbox("prototypes_to_use", ['class', 'batch', 'no'], index=0)
        n_positives = st.number_input("n_positives", value=1, step=1)
        n_negatives = st.number_input("n_negatives", value=1, step=1)
        # new_size = st.number_input("new_size", value=224, step=1)
        device = st.selectbox("device", ['cpu', 'cuda'], index=1)
        task = st.selectbox("task", ['notNormal', 'binary'], index=0)
        classif_loss = st.selectbox("classif_loss", ['arcface', 'triplet', 'softmax_contrastive', 'cosine'], index=0)
        # seed = st.number_input("seed", value=42, step=1)
        # bs = st.number_input("bs", value=32, step=1)
        # groupkfold = st.number_input("groupkfold", value=1, step=1)
        # random_recs = st.number_input("random_recs", value=0, step=1)
        valid_dataset = st.text_input("valid_dataset", value='Banque_Viscaino_Chili_2020')
        dist_fct = st.selectbox("dist_fct", ['euclidean', 'cosine'], index=0)

    # ---- Build args Namespace ---- #
    args = argparse.Namespace(
        model_name=model_name,
        fgsm=fgsm,
        n_calibration=n_calibration,
        dloss=dloss,
        prototypes_to_use=prototypes_to_use,
        n_positives=n_positives,
        n_negatives=n_negatives,
        # new_size=new_size,
        device=device,
        task=task,
        classif_loss=classif_loss,
        # seed=seed,
        path=os.path.join(data_dir, selected_path) if selected_path else None,
        # path_original=os.path.join(data_dir, selected_path_original) if selected_path_original else None,
        # bs=bs,
        # groupkfold=groupkfold,
        # random_recs=random_recs,
        valid_dataset=valid_dataset,
        dist_fct=dist_fct
    )

    # TODO REMOVE THESE PARAMETERS
    args.new_size = 224
    args.bs = 32
    args.groupkfold = 1
    args.random_recs = 0
    args.seed = 42

    return args


# ---- Load Model and Prototypes ---- #
@st.cache_resource
def load_model_and_prototypes(_args):
    data_getter = GetData(_args.path, _args.valid_dataset, _args)
    data, unique_labels, unique_batches = data_getter.get_variables()
    n_cats = len(unique_labels)
    n_batches = len(unique_batches)

    params = f'{_args.path.split("/")[-1]}/nsize{_args.new_size}/fgsm{_args.fgsm}/ncal{_args.n_calibration}/{_args.classif_loss}/{_args.dloss}/prototypes_{_args.prototypes_to_use}/npos{_args.n_positives}/nneg{_args.n_negatives}'
    model_path = f'logs/best_models/{_args.task}/{_args.model_name}/{params}/model.pth'
    proto_path = f'logs/best_models/{_args.task}/{_args.model_name}/{params}/prototypes.pkl'

    model = Net(_args.device, n_cats=n_cats, n_batches=n_batches, model_name=_args.model_name, is_stn=0, n_subcenters=n_batches)
    model.load_state_dict(torch.load(model_path, map_location=_args.device))
    model.to(_args.device)
    model.eval()

    shap_model = Net_shap(_args.device, n_cats=n_cats, n_batches=n_batches, model_name=_args.model_name, is_stn=0, n_subcenters=n_batches)
    shap_model.load_state_dict(torch.load(model_path, map_location=_args.device))
    shap_model.to(_args.device)
    shap_model.eval()

    with open(proto_path, 'rb') as f:
        proto_obj = pickle.load(f)
    prototypes = {'combined': proto_obj.prototypes, 'class': proto_obj.class_prototypes, 'batch': proto_obj.batch_prototypes}

    return model, shap_model, prototypes, _args.new_size, _args.device, data, unique_labels, unique_batches, data_getter

# ---- Image Preprocessing ---- #
def preprocess_image(img: Image.Image, size: int):
    img = img.resize((size, size))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)  # HWC → CHW
    tensor = torch.tensor(img_array).unsqueeze(0).to(device)
    return tensor

def log_shap_images_gradients_streamlit(nets, i, inputs, group, name, log_path):
    """
    Logs SHAP values and gradients and returns figures for Streamlit display.
    """
    log_shap_images_gradients(nets, i, inputs, group, log_path, name, device=device)


# ---- Streamlit UI ---- #

conn, cursor = create_db()

if 'user_email' not in st.session_state:
    st.session_state.user_email = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'person_id' not in st.session_state:
    st.session_state.person_id = None

if st.session_state.user_email is None:
    st.title("🔒 Login Required")
    email = st.text_input("Enter your email to log in or sign up:")
    if st.button("Continue") and email:
        try:
            cursor.execute("SELECT id FROM users WHERE email=%s", (email,))
            row = cursor.fetchone()
            if row:  # earlier I had a row here with just 1 in 
                st.session_state.user_id = row[0]
            else:
                cursor.execute("INSERT INTO users (email) VALUES (%s)", (email,))
                conn.commit()
                st.session_state.user_id = cursor.lastrowid
            st.session_state.user_email = email
            st.rerun()
        except Error as e:
            st.error(f"❌ Database error: {e}")
            st.stop()
    st.stop()

# ---- From here onward, user is guaranteed to be logged in ---- #
st.title("Ear Health Classifier with SHAP 👂")

# ---- Require person selection before using the app ---- #
st.sidebar.header("👤 Select a Family Member")
cursor.execute("SELECT id, name FROM people WHERE user_id=%s", (st.session_state.user_id,))
people = cursor.fetchall()
person_options = [p[1] for p in people]
person_ids = {p[1]: p[0] for p in people}

selected_person = st.sidebar.selectbox("Choose a person", person_options)

if selected_person:
    st.session_state.person_id = person_ids[selected_person]

with st.sidebar.expander("➕ Add a new person"):
    new_name = st.text_input("Person's Name", key="new_person_name")
    if st.button("Add Person") and new_name:
        try:
            cursor.execute("INSERT INTO people (user_id, name) VALUES (%s, %s)", (st.session_state.user_id, new_name))
            conn.commit()
            st.rerun()
        except Error as e:
            st.error(f"❌ Could not add person: {e}")

if st.session_state.person_id is None:
    st.warning("👤 Please select a family member to proceed.")
    st.stop()



if st.session_state.person_id:
    st.subheader("📄 Results for Family Member")

    query = '''
        SELECT filename, model_name, task, confidence, timestamp
        FROM results
        WHERE person_id=%s
        ORDER BY timestamp DESC
    '''
    cursor.execute(query, (st.session_state.person_id,))
    rows = cursor.fetchall()

    import pandas as pd
    df = pd.DataFrame(rows, columns=["Filename", "Model", "Task", "Confidence", "Timestamp"])

    selected_row = st.dataframe(df, use_container_width=True)

# ---- Database Insertion ---- #
def insert_score(cursor, conn, filename, args, pred_label, confidence, log_path):
    query = '''
        INSERT INTO results (
            filename, model_name, task, path, nsize, fgsm, n_calibration, classif_loss,
            dloss, prototypes, npos, nneg, pred_label, confidence, log_path, person_id
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    '''
    values = (
        filename, args.model_name, args.task, args.path, str(args.new_size), str(args.fgsm),
        str(args.n_calibration), args.classif_loss, args.dloss, args.prototypes_to_use,
        str(args.n_positives), str(args.n_negatives), pred_label, confidence, log_path,
        st.session_state.person_id
    )
    cursor.execute(query, values)
    conn.commit()

uploaded_file = st.file_uploader("Upload an ear image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    args = get_args()
    exists = check_ds_exists(cursor, uploaded_file.name, args)

    if exists is None:
        model, shap_model, prototypes, image_size, device, data, unique_labels, unique_batches, data_getter = \
            load_model_and_prototypes(args)
        
        save_path = os.path.join('data/queries', uploaded_file.name)

        # Save the uploaded file
        with open(save_path, 'wb') as f:
            f.write(uploaded_file.read())

        st.success(f"File saved to {save_path}")
        
        #  dist_fct = get_distance_fct(params['dist_fct'])
        train = TrainAE(args, args.path, load_tb=False, log_metrics=True, keep_models=True,
                        log_inputs=False, log_plots=True, log_tb=False, log_neptune=True,
                        log_mlflow=False, groupkfold=args.groupkfold)
        # train.make_triplet_loss(dist_fct, params)
        train.n_batches = len(unique_batches)
        train.n_cats = len(unique_labels)
        train.unique_batches = unique_batches
        train.unique_labels = unique_labels
        train.epoch = 1  # Otherwise it will throw an error in TrainAE.get_losses
        train.params = {
            'n_neighbors': 10,  # TODO NOT HARDCODE THIS
            'lr': 0,
            'wd': 0,
            'smoothing': 0,
            'is_transform': 0,
            'valid_dataset': args.valid_dataset
        }

        train.set_arcloss()

        lists, traces = get_empty_traces()
        # values, _, _, _ = get_empty_dicts()  # Pas élégant
        # Loading best model that was saved during training

        params = f'{args.path.split("/")[-1]}/nsize{args.new_size}/fgsm{args.fgsm}/ncal{args.n_calibration}/{args.classif_loss}/' \
                    f'{args.dloss}/prototypes_{args.prototypes_to_use}/' \
                    f'npos{args.n_positives}/nneg{args.n_negatives}'
        model_path = f'logs/best_models/{args.task}/{args.model_name}/{params}/model.pth'
        train.complete_log_path = f'logs/best_models/{args.task}/{args.model_name}/{params}'

        model = Net(args.device, train.n_cats, train.n_batches,
                            model_name=args.model_name, is_stn=0,
                            n_subcenters=train.n_batches)
        train.model = model
        model.load_state_dict(torch.load(model_path))
        model.to(args.device)
        model.eval()
        shap_model = Net_shap(args.device, train.n_cats, train.n_batches,
                            model_name=args.model_name, is_stn=0,
                            n_subcenters=train.n_batches)
        shap_model.load_state_dict(torch.load(model_path))
        shap_model.eval()
        prototypes = pickle.load(open(f'logs/best_models/{args.task}/{args.model_name}/{params}/prototypes.pkl', 'rb'))

        prototypes = {
            'combined': prototypes.prototypes,
            'class': prototypes.class_prototypes,
            'batch': prototypes.batch_prototypes
        }

        loaders = get_images_loaders(data=data,
                                        random_recs=args.random_recs,
                                        weighted_sampler=0,
                                        is_transform=0,
                                        samples_weights=None,
                                        epoch=1,
                                        unique_labels=unique_labels,
                                        triplet_dloss=args.dloss, bs=args.bs,
                                        prototypes_to_use=args.prototypes_to_use,
                                        prototypes=prototypes,
                                        size=args.new_size,
                                        )
        with torch.no_grad():
            _, best_lists1, _ = train.loop('train', None, 0, loaders['train'], lists, traces)
            for group in ["train", "valid", "test"]:
                _, best_lists2, traces, knn = train.predict(group, loaders[group], lists, traces)
        best_lists = {**best_lists1, **best_lists2}

        nets = {'cnn': shap_model, 'knn': knn}
        original, image = get_image(f'data/queries/{uploaded_file.name}', size=image_size)
        inputs = {
            'queries': {"inputs": [image]},
            'train': {"inputs": best_lists['train']['inputs']},
        }
        
        # TODO Get valid scores
        
        correct = best_lists['valid']['preds'] == best_lists['valid']['targets']
        valid_mcc = best_lists['valid']['mcc']

        # Get prediction
        with torch.no_grad():
            embedding = nets['cnn'](image.to(device))
            embedding = embedding.detach().cpu().numpy()
            pred_probs = nets['knn'].predict_proba(embedding)
            pred_class = np.argmax(pred_probs, axis=1)[0]  # Get class index
            pred_confidence = np.max(pred_probs)  # Optional: confidence
            pred_label = unique_labels[pred_class]

        st.write(f"**Predicted Label:** {pred_label} ({pred_confidence:.2f} confidence)")

        complete_log_path = f"logs/best_models/{args.task}/{args.model_name}/{args.path.split('/')[-1]}/" \
                            f"nsize{args.new_size}/fgsm{args.fgsm}/ncal{args.n_calibration}/" \
                            f"{args.classif_loss}/{args.dloss}/prototypes_{args.prototypes_to_use}" \
                            f"/npos{args.n_positives}/nneg{args.n_negatives}/queries"

        os.makedirs(f'{complete_log_path}/gradients_shap', exist_ok=True)
        os.makedirs(f'{complete_log_path}/knn_shap', exist_ok=True)

        log_shap_images_gradients_streamlit(nets, i=0, inputs=inputs, group='queries', 
                                                    name=uploaded_file.name.split("/")[-1], 
                                                    log_path=complete_log_path
                                        )
        
        print('logging shap done')

        # Insert results into the database
        insert_score(cursor, conn, uploaded_file.name, args, pred_label, pred_confidence, complete_log_path)
        st.success("Results saved to database.")

    # Import figures at log_path/gradients_shap/queries_{name}.png
    fig = plt.imread(f'logs/queries/gradients_shap/queries_{uploaded_file.name.split("/")[-1]}_layer5.png')
    st.image(fig, caption="SHAP Gradient Explanation (layer 5)", use_column_width=True)

    fig = plt.imread(f'logs/queries/knn_shap/queries_{uploaded_file.name.split("/")[-1]}_layer5.png')
    st.image(fig, caption="KNN SHAP Gradient Explanation (layer 5)", use_column_width=True)

    fig = plt.imread(f'logs/queries/knn_shap/queries_{uploaded_file.name.split("/")[-1]}.png')
    st.image(fig, caption="KNN SHAP Gradient Explanation", use_column_width=True)

conn.close()
