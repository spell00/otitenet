import os

# Disable Streamlit file watcher early to avoid importing/inspecting heavy packages
# (prevents Streamlit from touching packages like torch._classes which can raise)
os.environ.setdefault("STREAMLIT_FILE_WATCHER_TYPE", "none")

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

# os.environ["STREAMLIT_SECRETS_LOAD_MODE"] = "read_only"
import torch
import pickle
import mysql.connector
from mysql.connector import Error
import numpy as np
from PIL import Image
from otitenet.train.train_triplet_new import TrainAE  # If needed for params
from otitenet.data.data_getters import GetData, get_images_loaders, get_images, PerImageNormalize  # Update import path
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

def get_image(path, size=-1, normalize='no'):
    ops = [
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ]
    if str(normalize).lower() in ['yes', 'true', '1']:
        ops.append(PerImageNormalize())
    
    transform = transforms.Compose(ops)
    original = Image.open(path).convert('RGB')
    if size != -1:
        png = transforms.Resize((size, size))(original)
    else:
        png = original
    print(size, png)
    
    return transform(original).unsqueeze(0), transform(png).unsqueeze(0)


def choose_dataset(label, datasets, default=None, key=None):
    if len(datasets) == 0:
        st.warning(f"No datasets found in {data_dir}.")
        return None
    elif len(datasets) <= 3:
        return st.radio(label, datasets, index=datasets.index(default) if default in datasets else 0, key=key)
    else:
        return st.selectbox(label, datasets, index=datasets.index(default) if default in datasets else 0, key=key)


# ---- Database Lookup ---- #
def check_ds_exists(cursor, filename, args):
    cursor.execute(''' 
        SELECT * FROM results
        WHERE filename=%s AND model_name=%s AND nsize=%s AND fgsm=%s AND normalize=%s AND
              n_calibration=%s AND classif_loss=%s AND dloss=%s
    ''', (filename, args.model_name, str(args.new_size), str(args.fgsm),
          args.normalize, str(args.n_calibration), args.classif_loss, args.dloss))
    
    row = cursor.fetchone()  # Consume the result here
    return row

def build_params_from_args(_args, keys):
    parts = []
    for key in keys:
        value = getattr(_args, key, None)
        if value is not None:
            parts.append(f"{key}{value}")
    return "/".join(parts)


def extract_params_from_log_path(log_path: str):
    """Derive leaderboard defaults from the stored best-model log path."""
    params = {}
    if not log_path:
        return params
    parts = log_path.strip("/").split("/")
    try:
        base_idx = parts.index("best_models")
    except ValueError:
        return params

    if len(parts) > base_idx + 1:
        params["Task"] = parts[base_idx + 1]
    if len(parts) > base_idx + 3:
        params["Dataset"] = parts[base_idx + 3]
    if len(parts) > base_idx + 4:
        nsize_part = parts[base_idx + 4]
        if nsize_part.startswith("nsize"):
            params["new_size"] = nsize_part[len("nsize"):]
    if len(parts) > base_idx + 5:
        params.setdefault("FGSM", parts[base_idx + 5])
    if len(parts) > base_idx + 6:
        params.setdefault("N_Calibration", parts[base_idx + 6])
    if len(parts) > base_idx + 7:
        params.setdefault("classif_loss", parts[base_idx + 7])
    if len(parts) > base_idx + 8:
        params.setdefault("DLoss", parts[base_idx + 8])
    if len(parts) > base_idx + 9:
        params.setdefault("Prototypes", parts[base_idx + 9])
    if len(parts) > base_idx + 10:
        params.setdefault("NPos", parts[base_idx + 10])
    if len(parts) > base_idx + 11:
        params.setdefault("NNeg", parts[base_idx + 11])
    if len(parts) > base_idx + 12:
        params.setdefault("Normalize", parts[base_idx + 12])
    return params

def get_args(selected_params=None):
    if selected_params is None:
        selected_params = st.session_state.get('selected_model_params', {})
    selected_params_version = st.session_state.get('selected_params_version')
    last_synced_version = st.session_state.get('selected_params_last_sync')
    should_sync = bool(selected_params) and selected_params_version != last_synced_version
    if should_sync:
        st.session_state['selected_params_last_sync'] = selected_params_version

    def sync_value(key, value):
        if should_sync and value is not None:
            st.session_state[key] = value
    # ---- User selects paths ---- #
    with st.sidebar:
        st.header("Model Parameters")
        available_datasets = sorted([
            name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))
        ])
        dataset_default = selected_params.get('Dataset')
        if dataset_default not in available_datasets:
            dataset_default = 'otite_ds_64' if 'otite_ds_64' in available_datasets else (available_datasets[0] if available_datasets else None)
        sync_value('dataset_selectbox', dataset_default)
        selected_path = choose_dataset(
            "Select --path dataset", available_datasets, default=dataset_default, key="dataset_selectbox"
        )
        # selected_path = choose_dataset(
        #     "Select --path dataset", available_datasets, default='otite_ds_64'
        # )  # No key needed, handled in choose_dataset
        # selected_path_original = choose_dataset(
        #     "Select --path_original dataset", available_datasets, default='otite_ds_-1'
        # )

        # ---- Other args with defaults ---- #
        new_size_raw = selected_params.get('new_size')
        try:
            new_size = int(new_size_raw) if new_size_raw is not None else 224
        except ValueError:
            new_size = 224

        task_root = 'logs/best_models'
        task_list = sorted(os.listdir(task_root)) if os.path.isdir(task_root) else []
        if not task_list:
            st.error("No tasks found under logs/best_models.")
            st.stop()
        task_default = selected_params.get("Task")
        if task_default not in task_list:
            task_default = task_list[0]
        sync_value('task_selectbox', task_default)
        task = st.selectbox("task", task_list, key="task_selectbox")

        model_dir = os.path.join(task_root, task)
        model_name_list = [name for name in sorted(os.listdir(model_dir)) if not name.endswith('.csv')] if os.path.isdir(model_dir) else []
        if not model_name_list:
            st.error(f"No models found for task {task}.")
            st.stop()
        model_name_default = selected_params.get("Model Name")
        if model_name_default not in model_name_list:
            model_name_default = model_name_list[0]
        sync_value('model_name_selectbox', model_name_default)
        model_name = st.selectbox("Model Name", model_name_list, key="model_name_selectbox")

        fgsm_dir = os.path.join(model_dir, model_name, 'otite_ds_64', f'nsize{new_size}')
        fgsm_list = sorted(os.listdir(fgsm_dir)) if os.path.isdir(fgsm_dir) else []
        if not fgsm_list:
            st.error(f"No FGSM entries found in {fgsm_dir}.")
            st.stop()
        fgsm_default = selected_params.get("FGSM")
        if fgsm_default not in fgsm_list:
            fgsm_default = fgsm_list[0]
        sync_value('fgsm_selectbox', fgsm_default)
        fgsm = st.selectbox("fgsm", fgsm_list, key="fgsm_selectbox")

        n_cal_dir = os.path.join(fgsm_dir, fgsm)
        n_calibration_list = sorted(os.listdir(n_cal_dir)) if os.path.isdir(n_cal_dir) else []
        if not n_calibration_list:
            st.error(f"No calibration folders found in {n_cal_dir}.")
            st.stop()
        n_calibration_default = selected_params.get("N_Calibration")
        if n_calibration_default not in n_calibration_list:
            n_calibration_default = n_calibration_list[0]
        sync_value('n_calibration_selectbox', n_calibration_default)
        n_calibration = st.selectbox("n_calibration", n_calibration_list, key="n_calibration_selectbox")

        classif_dir = os.path.join(n_cal_dir, n_calibration)
        classif_loss_list = sorted(os.listdir(classif_dir)) if os.path.isdir(classif_dir) else []
        if not classif_loss_list:
            st.error(f"No classif_loss folders found in {classif_dir}.")
            st.stop()
        classif_loss_default = selected_params.get("classif_loss")
        if classif_loss_default not in classif_loss_list:
            classif_loss_default = classif_loss_list[0]
        sync_value('classif_loss_selectbox', classif_loss_default)
        classif_loss = st.selectbox("classif_loss", classif_loss_list, key="classif_loss_selectbox")

        dloss_dir = os.path.join(classif_dir, classif_loss)
        dloss_list = sorted(os.listdir(dloss_dir)) if os.path.isdir(dloss_dir) else []
        if not dloss_list:
            st.error(f"No dloss folders found in {dloss_dir}.")
            st.stop()
        dloss_default = selected_params.get("DLoss")
        if dloss_default not in dloss_list:
            dloss_default = dloss_list[0]
        sync_value('dloss_selectbox', dloss_default)
        dloss = st.selectbox("dloss", dloss_list, key="dloss_selectbox")

        proto_dir = os.path.join(dloss_dir, dloss)
        prototypes_list = sorted(os.listdir(proto_dir)) if os.path.isdir(proto_dir) else []
        if not prototypes_list:
            st.error(f"No prototype folders found in {proto_dir}.")
            st.stop()
        prototypes_default = selected_params.get("Prototypes")
        if prototypes_default not in prototypes_list:
            prototypes_default = prototypes_list[0]
        sync_value('prototypes_selectbox', prototypes_default)
        prototypes_to_use = st.selectbox("prototypes_to_use", prototypes_list, key="prototypes_selectbox")

        npos_dir = os.path.join(proto_dir, prototypes_to_use)
        n_positives_list = sorted(os.listdir(npos_dir)) if os.path.isdir(npos_dir) else []
        if not n_positives_list:
            st.error(f"No npos folders found in {npos_dir}.")
            st.stop()
        npos_default = str(selected_params.get("NPos")) if selected_params.get("NPos") is not None else None
        if npos_default not in n_positives_list:
            npos_default = n_positives_list[0]
        sync_value('npos_selectbox', npos_default)
        n_positives = st.selectbox('npos', n_positives_list, key="npos_selectbox")

        nneg_dir = os.path.join(npos_dir, n_positives)
        n_negatives_list = sorted(os.listdir(nneg_dir)) if os.path.isdir(nneg_dir) else []
        if not n_negatives_list:
            st.error(f"No nneg folders found in {nneg_dir}.")
            st.stop()
        nneg_default = str(selected_params.get("NNeg")) if selected_params.get("NNeg") is not None else None
        if nneg_default not in n_negatives_list:
            nneg_default = n_negatives_list[0]
        sync_value('nneg_selectbox', nneg_default)
        n_negatives = st.selectbox('nneg', n_negatives_list, key="nneg_selectbox")

        n_neighbors_default = selected_params.get("n_neighbors")
        if should_sync and n_neighbors_default is not None:
            try:
                st.session_state['n_neighbors_input'] = int(n_neighbors_default)
            except (TypeError, ValueError):
                st.session_state['n_neighbors_input'] = 1
        n_neighbors_default_value = st.session_state.get('n_neighbors_input', 1)
        n_neighbors = st.number_input("n_neighbors", value=int(n_neighbors_default_value), step=1, key="n_neighbors_input")

        normalize_default = selected_params.get("Normalize", "no")
        if should_sync and normalize_default is not None:
            st.session_state['normalize_input'] = normalize_default
        normalize = st.selectbox("normalize", ['yes', 'no'], index=1 if normalize_default == "no" else 0, key="normalize_input")

        device = st.selectbox("device", ['cpu', 'cuda'], index=1, key="device_selectbox")
        valid_dataset_default = selected_params.get('valid_dataset', 'Banque_Viscaino_Chili_2020')
        if should_sync and valid_dataset_default is not None:
            st.session_state['valid_dataset_input'] = valid_dataset_default
        valid_dataset = st.text_input("valid_dataset", value=st.session_state.get('valid_dataset_input', valid_dataset_default), key="valid_dataset_input")
        dist_fct = st.selectbox("dist_fct", ['euclidean', 'cosine'], index=0, key="dist_fct_selectbox")
        # new_size = st.number_input("new_size", value=224, step=1)
        # seed = st.number_input("seed", value=42, step=1)
        # bs = st.number_input("bs", value=32, step=1)
        # groupkfold = st.number_input("groupkfold", value=1, step=1)
        # random_recs = st.number_input("random_recs", value=0, step=1)

    # ---- Build args Namespace ---- #
    args = argparse.Namespace(
        model_name=model_name,
        fgsm=fgsm,
        n_calibration=n_calibration,
        n_neighbors=n_neighbors,
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
    args.new_size = new_size
    args.bs = 32
    args.groupkfold = 1
    args.random_recs = 0
    args.seed = 42
    args.normalize = normalize

    return args


# ---- Load Model and Prototypes ---- #
@st.cache_resource
def load_model_and_prototypes(_args):
    data_getter = GetData(_args.path, _args.valid_dataset, _args)
    data, unique_labels, unique_batches = data_getter.get_variables()
    n_cats = len(unique_labels)
    n_batches = len(unique_batches)

    params = f'{_args.path.split("/")[-1]}/nsize{_args.new_size}/' \
             f'{_args.fgsm}/{_args.n_calibration}/{_args.classif_loss}/' \
             f'{_args.dloss}/{_args.prototypes_to_use}/{_args.n_positives}' \
             f'/{_args.n_negatives}'
    model_path = f'logs/best_models/{_args.task}/{_args.model_name}/{params}/model.pth'
    proto_path = f'logs/best_models/{_args.task}/{_args.model_name}/{params}/prototypes.pkl'

    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.stop()
    if not os.path.exists(proto_path):
        st.error(f"❌ Prototype file not found: {proto_path}")
        st.stop()

    model = Net(_args.device, n_cats=n_cats, n_batches=n_batches,
                model_name=_args.model_name, is_stn=0,
                n_subcenters=n_batches)
    model.load_state_dict(torch.load(model_path, map_location=_args.device))
    model.to(_args.device)
    model.eval()

    shap_model = Net_shap(_args.device, n_cats=n_cats, n_batches=n_batches,
                          model_name=_args.model_name, is_stn=0,
                          n_subcenters=n_batches)
    shap_model.load_state_dict(torch.load(model_path, map_location=_args.device))
    shap_model.to(_args.device)
    shap_model.eval()

    with open(proto_path, 'rb') as f:
        proto_obj = pickle.load(f)
    prototypes = {'combined': proto_obj.prototypes, 'class': proto_obj.class_prototypes, 'batch': proto_obj.batch_prototypes}

    return model, shap_model, prototypes, _args.new_size, _args.device, data, unique_labels, unique_batches, data_getter

# ---- Image Preprocessing ---- #
def preprocess_image(img: Image.Image, size: int, normalize='no'):
    img = img.resize((size, size))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)  # HWC → CHW
    tensor = torch.tensor(img_array)
    
    if str(normalize).lower() in ['yes', 'true', '1']:
        per_img_norm = PerImageNormalize()
        tensor = per_img_norm(tensor)
    
    tensor = tensor.unsqueeze(0).to(device)
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

with st.sidebar.expander("❌ Remove person"):
    if len(person_options) > 0:
        person_to_remove = st.selectbox("Select person to delete", person_options, key="remove_person")
        if st.button("Delete Person"):
            try:
                person_id = person_ids[person_to_remove]
                cursor.execute("DELETE FROM results WHERE person_id = %s", (person_id,))
                cursor.execute("DELETE FROM people WHERE id = %s", (person_id,))
                conn.commit()
                if st.session_state.person_id == person_id:
                    st.session_state.person_id = None
                st.success(f"Deleted {person_to_remove}")
                st.rerun()
            except Error as e:
                st.error(f"❌ Could not delete person: {e}")

with st.sidebar.expander("🚫 Delete your account"):
    if st.button("Delete My Account"):
        try:
            uid = st.session_state.user_id
            cursor.execute("DELETE FROM results WHERE person_id IN (SELECT id FROM people WHERE user_id = %s)", (uid,))
            cursor.execute("DELETE FROM people WHERE user_id = %s", (uid,))
            cursor.execute("DELETE FROM users WHERE id = %s", (uid,))
            conn.commit()
            st.success("Your account and all associated data has been deleted.")
            st.session_state.clear()
            st.rerun()
        except Error as e:
            st.error(f"❌ Could not delete account: {e}")

# ---- Remove a result ---- #
with st.sidebar.expander("🗑️ Remove a result"):
    cursor.execute("""
        SELECT id, filename FROM results
        WHERE person_id = %s
        ORDER BY timestamp DESC
    """, (st.session_state.person_id,))
    results = cursor.fetchall()
    if results:
        result_names = [r[1] for r in results]
        result_map = {r[1]: r[0] for r in results}
        result_to_delete = st.selectbox("Select result to delete", result_names, key="delete_result")
        if st.button("Delete Result"):
            try:
                cursor.execute("DELETE FROM results WHERE id = %s", (result_map[result_to_delete],))
                conn.commit()
                st.success(f"Deleted result: {result_to_delete}")
                st.rerun()
            except Error as e:
                st.error(f"❌ Could not delete result: {e}")

    with st.sidebar:
        if st.button("🚪 Log Out"):
            st.session_state.user_email = None
            st.session_state.user_id = None
            st.session_state.person_id = None
            st.success("Logged out successfully.")
            st.rerun()

if st.session_state.person_id is None:
    st.warning("👤 Please select a family member to proceed.")
    st.stop()



selected_filename = None
if st.session_state.person_id:
    st.subheader("📄 Results for Family Member")
    # Query more parameters if needed
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

    for i, row in df.iterrows():
        if st.button(f"View SHAP: {row['Filename']}", key=f"view_shap_{row['Filename']}_{i}"):
            selected_filename = row['Filename']
            break
    selected_row = st.dataframe(df, use_container_width=True)


# ---- Database Insertion ---- #
def insert_score(cursor, conn, filename, args, pred_label, confidence, log_path):
    query = '''
        INSERT INTO results (
            filename, model_name, task, path, n_neighbors, nsize, fgsm, normalize, n_calibration, classif_loss,
            dloss, prototypes, npos, nneg, pred_label, confidence, log_path, person_id
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    '''
    values = (
        filename, args.model_name, args.task, args.path, str(args.n_neighbors), str(args.new_size), str(args.fgsm),
        args.normalize, str(args.n_calibration), args.classif_loss, args.dloss, args.prototypes_to_use,
        str(args.n_positives), str(args.n_negatives), pred_label, confidence, log_path,
        st.session_state.person_id
    )
    cursor.execute(query, values)
    
    # Update model_usage_summary
    summary_query = '''
        INSERT INTO model_usage_summary (
            model_name, task, path, nsize, fgsm, normalize, n_calibration, classif_loss,
            dloss, prototypes, npos, nneg, n_neighbors, num_samples_analyzed
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            num_samples_analyzed = num_samples_analyzed + 1,
            last_used = CURRENT_TIMESTAMP
    '''
    summary_values = (
        args.model_name, args.task, args.path, str(args.new_size), str(args.fgsm),
        args.normalize, str(args.n_calibration), args.classif_loss, args.dloss, args.prototypes_to_use,
        str(args.n_positives), str(args.n_negatives), str(args.n_neighbors), 1
    )
    cursor.execute(summary_query, summary_values)
    
    conn.commit()

uploaded_file = st.file_uploader("Upload an ear image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None or selected_filename is not None:
    if uploaded_file:
        st.success("✅ File uploaded. Ready to run analysis.")
        # Display non-normalized image before inference
        from PIL import Image
        import io
        uploaded_bytes = uploaded_file.read()
        img = Image.open(io.BytesIO(uploaded_bytes)).convert('RGB')
        st.image(img, caption="Original (non-normalized) image", use_column_width=True)
        # Reset file pointer for later use
        uploaded_file.seek(0)
    else:
        st.success("✅ Ready to run new analysis.")
    args = get_args()
    if st.button("Run Analysis"):
        # Only include keys that may exist
        param_keys = [
            "n_neighbors", "nsize", "fgsm", "n_calibration", "classif_loss",
            "dloss", "prototypes_to_use", "npos", "nneg"
        ]

        params = f'{args.path.split("/")[-1]}/' + build_params_from_args(args, param_keys)
        complete_log_path = f"logs/best_models/{args.task}/{args.model_name}/{args.path.split('/')[-1]}/" \
                            f"nsize{args.new_size}/{args.fgsm}/{args.n_calibration}/" \
                            f"{args.classif_loss}/{args.dloss}/{args.prototypes_to_use}" \
                            f"/{args.n_positives}/{args.n_negatives}/queries"

        # If force_reanalyze is set, delete old results for THIS EXACT model and parameters, then run fresh analysis
        force_reanalyze = st.session_state.get('force_reanalyze', False)
        if force_reanalyze:
            st.session_state['force_reanalyze'] = False
            exists = None
            # Delete old SHAP results ONLY if they exist for this exact model/parameters combination
            import shutil
            if os.path.exists(complete_log_path):
                try:
                    shutil.rmtree(complete_log_path)
                    st.info("🔄 Deleted previous analysis results for this model. Running fresh analysis...")
                except Exception as e:
                    st.warning(f"Could not delete old results: {e}")
        else:
            exists = check_ds_exists(cursor, uploaded_file.name, args)

        if exists is None or force_reanalyze:
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
                'n_neighbors': 1,  # TODO NOT HARDCODE THIS. info available in models.csv
                'lr': 0,
                'wd': 0,
                'smoothing': 0,
                'is_transform': 1,
                'valid_dataset': args.valid_dataset
            }

            train.set_arcloss()

            lists, traces = get_empty_traces()
            # values, _, _, _ = get_empty_dicts()  # Pas élégant
            # Loading best model that was saved during training

            params = f'{args.path.split("/")[-1]}/nsize{args.new_size}/{args.fgsm}/' \
                    f'{args.n_calibration}/{args.classif_loss}/' \
                        f'{args.dloss}/{args.prototypes_to_use}/' \
                        f'{args.n_positives}/{args.n_negatives}'
            model_path = f'logs/best_models/{args.task}/{args.model_name}/{params}/model.pth'
            train.complete_log_path = f'logs/best_models/{args.task}/{args.model_name}/{params}'

            model = Net(args.device, train.n_cats, train.n_batches,
                                model_name=args.model_name, is_stn=0,
                                n_subcenters=train.n_batches)
            train.model = model
            model.load_state_dict(torch.load(model_path, map_location=args.device))
            model.to(args.device)
            model.eval()
            shap_model = Net_shap(args.device, train.n_cats, train.n_batches,
                                model_name=args.model_name, is_stn=0,
                                n_subcenters=train.n_batches)
            shap_model.load_state_dict(torch.load(model_path, map_location=args.device))
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
                                            normalize=args.normalize,
                                            )
            with torch.no_grad():
                _, best_lists1, _ = train.loop('train', None, 0, loaders['train'], lists, traces)
                for group in ["train", "valid", "test"]:
                    try:
                        _, best_lists2, traces, knn = train.predict(group, loaders[group], lists, traces)
                    except Exception as e:
                        st.error(f"❌ Error during prediction on group '{group}': {e}")
                        st.stop()
            best_lists = {**best_lists1, **best_lists2}

            nets = {'cnn': shap_model, 'knn': knn}
            original, image = get_image(f'data/queries/{uploaded_file.name}', size=image_size, normalize=args.normalize)
            inputs = {
                'queries': {"inputs": [image]},
                'train': {
                    "inputs": [
                        torch.concatenate(best_lists['train']['inputs']),
                        torch.concatenate(best_lists['valid']['inputs'])
                    ],
                },
            }
            
            # TODO Get valid scores
            
            correct = np.argmax(np.concatenate(best_lists['valid']['preds']), 1) == np.concatenate(best_lists['valid']['cats'])
            valid_acc = np.sum(correct) / len(correct)
            valid_mcc = MCC(np.concatenate(best_lists['valid']['cats']), np.argmax(np.concatenate(best_lists['valid']['preds']), 1))

            # Get prediction
            with torch.no_grad():
                embedding = nets['cnn'](image.to(device))
                embedding = embedding.detach().cpu().numpy()
                pred_probs = nets['knn'].predict_proba(embedding)
                pred_class = np.argmax(pred_probs, axis=1)[0]  # Get class index
                pred_confidence = np.max(pred_probs)  # Optional: confidence
                pred_label = unique_labels[pred_class]

            st.write(f"**Validation Accuracy:** {valid_acc:.2f}")
            st.write(f"**Validation MCC:** {valid_mcc:.2f}")
            st.write(f"**Predicted Label:** {pred_label} ({pred_confidence:.2f} confidence)")

            os.makedirs(f'{complete_log_path}/gradients_shap', exist_ok=True)
            os.makedirs(f'{complete_log_path}/knn_shap', exist_ok=True)

            # Clean filename to avoid double extensions
            base_filename = uploaded_file.name.split("/")[-1]
            if base_filename.endswith('.png'):
                base_filename = base_filename[:-4]  # Remove .png extension
            
            print(f"Processing file: {uploaded_file.name}")
            print(f"Base filename: {base_filename}")
            print(f"Log path: {complete_log_path}")

            log_shap_images_gradients_streamlit(nets, i=0, inputs=inputs, group='queries', 
                                                        name=base_filename, 
                                                        log_path=complete_log_path
                                            )
            
            print('logging shap done')
            
            # Debug: List files in SHAP directories
            print("Files in gradients_shap:")
            if os.path.exists(f'{complete_log_path}/gradients_shap'):
                print(os.listdir(f'{complete_log_path}/gradients_shap'))
            
            print("Files in knn_shap:")
            if os.path.exists(f'{complete_log_path}/knn_shap'):
                print(os.listdir(f'{complete_log_path}/knn_shap'))

            # Insert results into the database
            insert_score(cursor, conn, uploaded_file.name, args, pred_label, pred_confidence, complete_log_path)
            st.success("Results saved to database.")

        else:
            # Load previous results
            pred_label = exists[14]
            pred_confidence = exists[15]
            log_path = exists[16]
            st.info("✅ This image was already analyzed with these exact model settings. Loading previous results...")
            st.write(f"**Previous Prediction:** {pred_label} ({pred_confidence:.2f} confidence)")

        # Import figures at log_path/gradients_shap/queries_{name}.png
        # Check if files exist before trying to display them
        base_name = uploaded_file.name.split("/")[-1]
        if base_name.endswith('.png'):
            base_name = base_name[:-4]  # Remove .png extension to avoid double extension
        
        # Try to display SHAP gradient explanation (layer 5)
        grad_shap_layer5_path = f'{complete_log_path}/gradients_shap/queries_{base_name}_layer5.png'
        if os.path.exists(grad_shap_layer5_path):
            fig = plt.imread(grad_shap_layer5_path)
            st.image(fig, caption="SHAP Gradient Explanation (layer 5)", use_column_width=True)
        else:
            st.warning(f"SHAP gradient explanation (layer 5) not found at: {grad_shap_layer5_path}")

        # Try to display KNN SHAP explanation (layer 5)
        knn_shap_layer5_path = f'{complete_log_path}/knn_shap/queries_{base_name}_layer5.png'
        if os.path.exists(knn_shap_layer5_path):
            fig = plt.imread(knn_shap_layer5_path)
            st.image(fig, caption="KNN SHAP Gradient Explanation (layer 5)", use_column_width=True)
        else:
            st.warning(f"KNN SHAP explanation (layer 5) not found at: {knn_shap_layer5_path}")

        # Try to display KNN SHAP explanation (main)
        knn_shap_path = f'{complete_log_path}/knn_shap/queries_{base_name}.png'
        if os.path.exists(knn_shap_path):
            fig = plt.imread(knn_shap_path)
            st.image(fig, caption="KNN SHAP Gradient Explanation", use_column_width=True)
        else:
            st.warning(f"KNN SHAP explanation not found at: {knn_shap_path}")
        
        # Add button to re-run analysis for this image
        if st.button("🔄 Re-run Analysis for This Image"):
            st.session_state['force_reanalyze'] = True
            st.rerun()

    # ---- Model Usage Summary ---- #
    st.subheader("📊 Models Used for Analysis")
    try:
        cursor.execute("""
            SELECT model_name, task, nsize, fgsm, normalize, n_calibration, classif_loss, dloss, prototypes, 
                   npos, nneg, n_neighbors, num_samples_analyzed, last_used
            FROM model_usage_summary
            ORDER BY last_used DESC
        """)
        usage_rows = cursor.fetchall()
        if usage_rows:
            import pandas as pd
            usage_columns = [
                "Model", "Task", "Size", "FGSM", "Normalize", "N_Cal", "Loss", "DLoss", "Prototypes", 
                "NPos", "NNeg", "N_Neighbors", "Samples", "Last Used"
            ]
            usage_df = pd.DataFrame(usage_rows, columns=usage_columns)
            st.dataframe(usage_df, use_container_width=True)
            
            # Add dropdown to load previous results for a specific model configuration
            st.subheader("📂 Load Previous Analysis Results")
            
            # Build dropdown options
            dropdown_options = []
            for row in usage_rows:
                model_name, task, nsize, fgsm, normalize, n_cal, loss, dloss, prototypes, npos, nneg, n_neighbors, samples, last_used = row
                option_label = f"{model_name} | Size:{nsize} FGSM:{fgsm} Norm:{normalize} | {samples} samples"
                dropdown_options.append({
                    'label': option_label,
                    'model_name': model_name,
                    'task': task,
                    'nsize': nsize,
                    'fgsm': fgsm,
                    'normalize': normalize,
                    'n_calibration': n_cal,
                    'classif_loss': loss,
                    'dloss': dloss,
                    'prototypes': prototypes,
                    'npos': npos,
                    'nneg': nneg,
                    'n_neighbors': n_neighbors
                })
            
            selected_model_idx = st.selectbox(
                "Select a model configuration to load previous results:",
                options=range(len(dropdown_options)),
                format_func=lambda i: dropdown_options[i]['label'],
                key="model_config_dropdown"
            )
            
            if st.button("📥 Load Results for Selected Model", key="load_model_results_btn"):
                selected_config = dropdown_options[selected_model_idx]
                
                # Get current filename from uploaded_file or selected_filename
                current_filename = None
                if uploaded_file is not None:
                    current_filename = uploaded_file.name
                elif selected_filename is not None:
                    current_filename = selected_filename
                
                if current_filename:
                    # Query for result with this model configuration AND the current filename
                    cursor.execute("""
                        SELECT filename, pred_label, confidence, log_path
                        FROM results
                        WHERE filename=%s AND model_name=%s AND nsize=%s AND fgsm=%s AND normalize=%s AND 
                              n_calibration=%s AND classif_loss=%s AND dloss=%s
                        ORDER BY timestamp DESC
                        LIMIT 1
                    """, (
                        current_filename,
                        selected_config['model_name'],
                        str(selected_config['nsize']),
                        str(selected_config['fgsm']),
                        selected_config['normalize'],
                        str(selected_config['n_calibration']),
                        selected_config['classif_loss'],
                        selected_config['dloss']
                    ))
                    
                    result = cursor.fetchone()
                    if result:
                        filename, pred_label, confidence, log_path = result
                        
                        st.success(f"✅ Found previous analysis for {filename}")
                        st.write(f"**Filename:** {filename}")
                        st.write(f"**Prediction:** {pred_label} ({confidence:.2f} confidence)")
                        st.write(f"**Model:** {selected_config['model_name']}")
                        
                        # Display SHAP explanations
                        base_name = filename.split("/")[-1]
                        if base_name.endswith('.png'):
                            base_name = base_name[:-4]
                        elif base_name.endswith('.jpg'):
                            base_name = base_name[:-4]
                        
                        # Display SHAP gradient explanation (layer 5)
                        grad_shap_layer5_path = f'{log_path}/gradients_shap/queries_{base_name}_layer5.png'
                        if os.path.exists(grad_shap_layer5_path):
                            fig = plt.imread(grad_shap_layer5_path)
                            st.image(fig, caption="SHAP Gradient Explanation (layer 5)", use_column_width=True)
                        else:
                            st.warning(f"SHAP gradient explanation (layer 5) not found at: {grad_shap_layer5_path}")
                        
                        # Display KNN SHAP explanation (layer 5)
                        knn_shap_layer5_path = f'{log_path}/knn_shap/queries_{base_name}_layer5.png'
                        if os.path.exists(knn_shap_layer5_path):
                            fig = plt.imread(knn_shap_layer5_path)
                            st.image(fig, caption="KNN SHAP Explanation (layer 5)", use_column_width=True)
                        else:
                            st.warning(f"KNN SHAP explanation (layer 5) not found at: {knn_shap_layer5_path}")
                    else:
                        st.info(f"No previous analysis found for '{current_filename}' with this model configuration.")
                else:
                    st.warning("Please upload or select a file first.")
        else:
            st.info("No models have been used for analysis yet.")
    except Exception as e:
        st.warning(f"Could not load model usage summary: {e}")

    # ---- Best Models Leaderboard ---- #
    st.subheader("🏆 Best Models Leaderboard")
    try:
        cursor.execute("""
            SELECT model_name, fgsm, prototypes, npos, nneg, dloss, n_calibration, accuracy, mcc, normalize, log_path
            FROM best_models_registry
            ORDER BY mcc DESC
        """)
        models = cursor.fetchall()
        import pandas as pd
        columns = [
            "Model Name", "FGSM", "Prototypes", "NPos", "NNeg", "DLoss", "N_Calibration", "Accuracy", "MCC", "Normalize", "Log Path"
        ]
        models_df = pd.DataFrame(models, columns=columns)

        # Keep only the best (highest MCC) for each unique combination of the first 7 columns
        group_cols = columns[:8]  # include Normalize in deduplication
        # Sort by MCC descending, then drop duplicates keeping the first (best MCC)
        models_df = models_df.sort_values("MCC", ascending=False).drop_duplicates(subset=group_cols, keep="first")

        st.write("**Top Models (best per parameter combination):**")
        display_columns = [col for col in models_df.columns if col != "Log Path"]
        display_df = models_df[display_columns].copy()
        st.dataframe(display_df, use_container_width=True)

        # Dropdown menu to select model
        model_options = [str(i) for i in range(len(models_df))]
        selected_idx = st.selectbox(
            "Select a model to use:",
            options=list(range(len(model_options))),
            format_func=lambda i: f"Model #{i} (normalize={models_df.iloc[i]['Normalize']})",
            index=0,
            key="model_selectbox_dropdown"
        )
        if st.button("Use Selected Model", key="use_selected_model_btn"):
            row = models_df.iloc[selected_idx]
            row_dict = row.to_dict()
            row_dict.update(extract_params_from_log_path(row_dict.get("Log Path")))
            st.session_state.selected_model_params = row_dict
            st.session_state.selected_params_version = st.session_state.get('selected_params_version', 0) + 1
            st.success(f"Selected model: {row['Model Name']} (normalize={row['Normalize']})")
            st.rerun()
    except Exception as e:
        st.warning(f"Could not load best models leaderboard: {e}")

    if st.session_state.get('selected_model_params'):
        st.info("Leaderboard selection applied. Adjust any parameters from the sidebar if needed.")

    conn.close()
