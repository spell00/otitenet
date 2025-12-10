import os

import matplotlib

matplotlib.use('Agg')
CUDA_VISIBLE_DEVICES = ""

import neptune
import mlflow
import warnings
import torchvision
import torch.nn.functional as F
import uuid
import copy
import shutil
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import json
import torch
from torch import nn
from PIL import Image
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from umap.umap_ import UMAP
from sklearn.manifold import Isomap
from sklearn.cross_decomposition import CCA
from sklearn.manifold import MDS
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn import metrics
from tensorboardX import SummaryWriter
from ax.service.managed_loop import optimize
from sklearn.metrics import matthews_corrcoef as MCC
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import mysql.connector


from ..logging.loggings import LogConfusionMatrix, add_to_neptune, add_to_mlflow, create_neptune_run, log_mlflow, \
    NEPTUNE_API_TOKEN, NEPTUNE_PROJECT_NAME, NEPTUNE_MODEL_NAME, add_to_logger
from ..data.data_getters import GetData, get_distance_fct, get_images_loaders, get_n_features
from ..utils.utils import get_optimizer, to_categorical, get_empty_dicts, get_empty_traces, \
    log_traces, save_tensor, get_best_params
from ..models.cnn import Net, Net_deep_shap, Net_shap, replace_bn_with_gn
from ..logging.losses import TupletLoss,ArcFaceLoss, ArcFaceLossWithHSM, \
        ArcFaceLossWithSubcenters, ArcFaceLossWithSubcentersHSM, SoftmaxContrastiveLoss
from ..logging.metrics import compute_batch_effect_metrics
from ..logging.plotting import save_roc_curve, plot_pca

warnings.filterwarnings("ignore")

random.seed(1)
torch.manual_seed(1)
np.random.seed(1)

from ..logging.shap import log_shap_images_gradients
from ..logging.grad_cam import log_grad_cam_similarity
from ..utils.prototypes import Prototypes

def update_best_model_registry(params, accuracy, mcc, log_path):
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="y_user",
            password="password",
            database="results_db"
        )
        cursor = conn.cursor()
        cursor.execute("""
            SELECT accuracy FROM best_models_registry
            WHERE model_name=%s AND fgsm=%s AND prototypes=%s AND npos=%s AND nneg=%s AND dloss=%s AND n_calibration=%s AND normalize=%s
        """, (params['model_name'], params['fgsm'], params['prototypes'], params['npos'], params['nneg'], params['dloss'], params['n_calibration'], params['normalize']))
        row = cursor.fetchone()
        if row is None or accuracy > row[0]:
            cursor.execute("""
                REPLACE INTO best_models_registry
                (model_name, fgsm, prototypes, npos, nneg, dloss, n_calibration, accuracy, mcc, normalize, log_path)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                params['model_name'],
                params['fgsm'],
                params['prototypes'],
                params['npos'],
                params['nneg'],
                params['dloss'],
                params['n_calibration'],
                float(accuracy),  # <-- cast to float
                float(mcc),       # <-- cast to float
                params['normalize'],
                log_path
            ))
            conn.commit()
        cursor.close()
        conn.close()
    except mysql.connector.Error as e:
        print(f"❌ Registry DB error: {e}")

class TrainAE:
    def __init__(self, args, path, load_tb=False, log_metrics=False, keep_models=True, log_inputs=True,
                 log_plots=False, log_tb=False, log_neptune=False, log_mlflow=True, groupkfold=True):
        """

        Args:
            args: contains multiple arguments passed in the command line
            log_path: Path where the tensorboard logs are saved
            path: Path to the data (in .csv format)
            load_tb: If True, loads previous runs already saved
        """
        self.hparams_names = None
        self.best_acc = 0
        self.best_best_mcc = -1
        self.best_closs_final = np.inf
        self.logged_inputs = False
        self.log_tb = log_tb
        self.log_neptune = log_neptune
        self.log_mlflow = log_mlflow
        self.args = args
        self.path = path
        self.log_metrics = log_metrics
        self.log_plots = log_plots
        self.log_inputs = log_inputs
        self.load_tb = load_tb
        self.groupkfold = groupkfold
        self.foldername = None

        self.verbose = 1
        self.epoch = 0

        self.n_cats = None
        self.data = None
        self.unique_labels = None
        self.unique_batches = None
        self.triplet_loss = None
        self.triplet_dloss = None
        
        self.arcloss = None

        # Store all samples (including those previously removed)
        self.all_samples = {
            'inputs': {'train': [], 'valid': [], 'test': [], 'calibration': [],
                       'train_calibration': [], 'valid_calibration': [], 'test_calibration': []},
            'labels': {'train': [], 'valid': [], 'test': [], 'calibration': [],
                       'train_calibration': [], 'valid_calibration': [], 'test_calibration': []},
            'old_labels': {'train': [], 'valid': [], 'test': [], 'calibration': [],
                           'train_calibration': [], 'valid_calibration': [], 'test_calibration': []},
            'names': {'train': [], 'valid': [], 'test': [], 'calibration': [],
                      'train_calibration': [], 'valid_calibration': [], 'test_calibration': []},
            'cats': {'train': [], 'valid': [], 'test': [], 'calibration': [],
                     'train_calibration': [], 'valid_calibration': [], 'test_calibration': []},
            'batches': {'train': [], 'valid': [], 'test': [], 'calibration': [],
                        'train_calibration': [], 'valid_calibration': [], 'test_calibration': []},
            'encoded_values': {'train': [], 'valid': [], 'test': [], 'calibration': [],
                               'train_calibration': [], 'valid_calibration': [], 'test_calibration': []},
            'subcenters': {'train': [], 'valid': [], 'test': [], 'calibration': [],
                           'train_calibration': [], 'valid_calibration': [], 'test_calibration': []},
        }
        self.initial_nsamples = None
        self.prototypes = {'train': [], 'valid': [], 'test': [], 'calibration': [],
                           'train_calibration': [], 'valid_calibration': [], 'test_calibration': [], 'all': []}
        self.class_prototypes = {'train': [], 'valid': [], 'test': [], 'calibration': [],
                                 'train_calibration': [], 'valid_calibration': [], 'test_calibration': [], 'all': []}
        self.batch_prototypes = {'train': [], 'valid': [], 'test': [], 'calibration': [],
                                 'train_calibration': [], 'valid_calibration': [], 'test_calibration': [], 'all': []}
        self.combined_prototypes = {'train': [], 'valid': [], 'test': [], 'calibration': [],
                                    'train_calibration': [], 'valid_calibration': [], 'test_calibration': [], 'all': []}
        self.n_prototypes = {'train': None, 'valid': None, 'test': None, 'calibration': None,
                             'train_calibration': None, 'valid_calibration': None, 'test_calibration': None, 'all': None}
        self.best_params = {}

    def save_wrong_classif_imgs(self, run, models, lists, preds, names, group):
        # Save Grad-CAM / SHAP artefacts for valid and test predictions
        for label in self.unique_labels:
            base_wrong = f'{self.complete_log_path}/wrong_classif/{label}/{group}'
            base_correct = f'{self.complete_log_path}/correct_classif/{label}/{group}'
            os.makedirs(f'{base_wrong}/gradients_shap', exist_ok=True)
            os.makedirs(f'{base_wrong}/imgs', exist_ok=True)
            os.makedirs(f'{base_wrong}/knn_shap', exist_ok=True)
            os.makedirs(f'{base_wrong}/grad_cam', exist_ok=True)
            os.makedirs(f'{base_correct}/gradients_shap', exist_ok=True)
            os.makedirs(f'{base_correct}/knn_shap', exist_ok=True)
            os.makedirs(f'{base_correct}/grad_cam', exist_ok=True)

        cats = np.concatenate(lists[group]['cats'])
        preds = np.concatenate(lists[group]['preds'])
        names = np.concatenate(lists[group]['names'])
        wrong = 0
        correct = 0
        prototype_store = self.class_prototypes.get('train', {})
        if not isinstance(prototype_store, dict):
            prototype_store = {}

        for i, (cat, pred_idx, name) in enumerate(zip(cats, np.argmax(preds, 1), names)):
            pred_label = self.unique_labels[pred_idx]
            proto = prototype_store.get(pred_label)
            if cat != pred_idx:
                wrong += 1
                try:
                    img = Image.open(f'{self.args.path_original}/{name}')
                    img.save(f'{self.complete_log_path}/wrong_classif/{self.unique_labels[cat]}/{group}/imgs/{wrong}_{name}.png')
                except Exception as e:
                    print(f'Failed to save image {name}: {e}')
                log_shap_images_gradients(
                    models,
                    i,
                    lists,
                    group,
                    f'{self.complete_log_path}/wrong_classif/{self.unique_labels[cat]}/{group}/',
                    f'{wrong}_{name}',
                )
                if proto is not None:
                    log_grad_cam_similarity(
                        models['cnn'],
                        i,
                        lists,
                        group,
                        f'{self.complete_log_path}/wrong_classif/{self.unique_labels[cat]}/{group}/grad_cam',
                        f'{wrong}_{name}',
                        proto,
                        device=self.args.device,
                    )
            else:
                log_shap_images_gradients(
                    models,
                    i,
                    lists,
                    group,
                    f'{self.complete_log_path}/correct_classif/{self.unique_labels[cat]}/{group}/',
                    f'{correct}_{name}',
                )
                if proto is not None:
                    log_grad_cam_similarity(
                        models['cnn'],
                        i,
                        lists,
                        group,
                        f'{self.complete_log_path}/correct_classif/{self.unique_labels[cat]}/{group}/grad_cam',
                        f'{correct}_{name}',
                        proto,
                        device=self.args.device,
                    )
                correct += 1

        for label in self.unique_labels:
            base_wrong = f'{self.complete_log_path}/wrong_classif/{label}/{group}'
            base_correct = f'{self.complete_log_path}/correct_classif/{label}/{group}'
            run[f'wrong_classif/{label}/{group}'].upload_files(f'{base_wrong}/imgs')
            run[f'wrong_classif/{label}/{group}'].upload_files(f'{base_wrong}/gradients_shap')
            run[f'wrong_classif/{label}/{group}'].upload_files(f'{base_wrong}/knn_shap')
            run[f'wrong_classif/{label}/{group}'].upload_files(f'{base_wrong}/grad_cam')
            run[f'correct_classif/{label}/{group}'].upload_files(f'{base_correct}/gradients_shap')
            run[f'correct_classif/{label}/{group}'].upload_files(f'{base_correct}/knn_shap')
            run[f'correct_classif/{label}/{group}'].upload_files(f'{base_correct}/grad_cam')

    def set_prototypes(self, group, list1):
        self.prototypes.set_prototypes(group, list1)
        self.combined_prototypes = self.prototypes.prototypes
        self.class_prototypes = self.prototypes.class_prototypes
        self.batch_prototypes = self.prototypes.batch_prototypes
        self.n_prototypes = self.prototypes.n_prototypes
        self.means = self.prototypes.means

    def make_samples_weights(self):
        self.n_batches = len(
            set(np.concatenate((
            self.data['batches']['train'],
            self.data['batches']['valid'],
            self.data['batches']['test']
            )))
        )
        self.class_weights = {
            label: 1 / (len(np.where(label == self.data['labels']['train'])[0]) /
                        self.data['labels']['train'].shape[0])
            for label in self.unique_labels if
            label in self.data['labels']['train']}
        self.unique_unique_labels = list(self.class_weights.keys())

        self.samples_weights = {
            group: [self.class_weights[label] for name, label in
                    zip(self.data['names'][group],
                        self.data['labels'][group])] if group == 'train' else [
                1 for name, label in
                zip(self.data['names'][group], self.data['labels'][group])] for group in
            ['train', 'valid', 'test']}
        self.n_cats = len(self.unique_labels)
        self.scaler = None
        if self.args.classif_loss == 'ce':
            self.celoss = nn.CrossEntropyLoss()
        elif self.args.classif_loss == 'hinge':
            self.celoss = nn.MultiMarginLoss(p=2, margin=self.params['margin'])
        if self.args.dloss == 'DANN':
            self.dceloss = nn.CrossEntropyLoss()

    def make_triplet_loss(self, dist_fct, params):
        if self.args.n_positives + self.args.n_negatives > 2:
            self.triplet_loss = TupletLoss(margin=params['margin'])
            self.triplet_dloss = TupletLoss(margin=params['dmargin'])
        elif self.args.classif_loss == 'softmax_contrastive':
            self.triplet_loss = SoftmaxContrastiveLoss(
                temperature=params['smoothing'], distance_metric=params['dist_fct']
            )
        elif self.args.classif_loss in ['triplet']:
            self.triplet_loss = nn.TripletMarginWithDistanceLoss(
                distance_function=dist_fct, margin=params['margin'], swap=False
            )
        if self.args.dloss in ['inverseTriplet']:
            self.triplet_dloss = nn.TripletMarginWithDistanceLoss(
                distance_function=dist_fct, margin=params['dmargin'], swap=False
            )
        elif self.args.dloss in ['inverse_softmax_contrastive']:
            self.triplet_dloss = SoftmaxContrastiveLoss(
                temperature=params['smoothing'], distance_metric=params['dist_fct']
            )
        elif self.args.dloss in ['none']:
            self.triplet_dloss = None


    def train(self, params):
        """

        Args:
            params: Contains the hyperparameters to be optimized

        Returns:
            best_closs: The best classification loss on the valid set

        """
        start_time = datetime.now()
        # Fixing the hyperparameters that are not optimized

        print(params)

        # Assigns the hyperparameters getting optimized
        self.params = params
        if not self.args.fgsm:
            params['epsilon'] = 0
        if 'margin' not in params:
            params['margin'] = 0
        if 'dmargin' not in params:
            params['dmargin'] = 0
        if 'dist_fct' not in params:
            params['dist_fct'] = 'none'
        if 'gamma' not in params:
            params['gamma'] = 0
        if 'n_neighbors' not in params:
            params['n_neighbors'] = 0
        if 'is_transform' not in params:
            params['is_transform'] = 1
        if 'normalize' not in params:
            params['normalize'] = 'no'
        if not hasattr(self.args, 'normalize'):
            self.args.normalize = params['normalize']
        params['dropout'] = 0
        params['optimizer_type'] = 'adam'
        dist_fct = get_distance_fct(params['dist_fct'])

        self.make_triplet_loss(dist_fct, params)

        self.foldername = str(uuid.uuid4())

        if self.log_mlflow:
            log_mlflow(self.args, params, self.foldername)

        self.complete_log_path = f'logs/{self.args.task}/{self.foldername}'
        loggers = {'cm_logger': LogConfusionMatrix(self.complete_log_path)}
        print(f'See results using: tensorboard --logdir={self.complete_log_path} --port=6006')

        hparams_filepath = self.complete_log_path + '/hp'
        os.makedirs(hparams_filepath, exist_ok=True)
        epoch = 0
        best_closses = []
        self.best_loss = np.inf
        self.best_closs = np.inf
        self.best_acc = 0
        data_getter = GetData(self.path, self.args.valid_dataset, self.args)
        self.unique_labels = data_getter.unique_labels
        self.unique_batches = data_getter.unique_batches
        self.prototypes = Prototypes(self.unique_labels, self.unique_batches)

        # Store all samples before clustering
        for group in ['train', 'valid', 'test', 'calibration']:
            self.all_samples['inputs'][group] = data_getter.data['inputs'][group]
            self.all_samples['labels'][group] = data_getter.data['labels'][group]
            self.all_samples['old_labels'][group] = data_getter.data['old_labels'][group]
            self.all_samples['names'][group] = data_getter.data['names'][group]
            self.all_samples['cats'][group] = data_getter.data['cats'][group]
            self.all_samples['batches'][group] = data_getter.data['batches'][group]
            # Means per channel

            # means = np.mean(data_getter.data['inputs'][group], axis=(0, 1, 2))
            # stds = np.std(data_getter.data['inputs'][group], axis=(0, 2, 3))

        self.initial_nsamples = len(data_getter.data['labels']['all'])
        event_acc = EventAccumulator(hparams_filepath)
        event_acc.Reload()
        NEPTUNE_MODEL_NAME = 'resnet50'
        if self.log_neptune:
            run = create_neptune_run(self.args, self.params, self.complete_log_path, self.foldername)
        else:
            run = None


        for h in range(self.args.n_repeats):
            print("Repeat:", h)
            self.best_mcc = -1
            if not self.args.groupkfold:
                self.data, self.unique_labels, self.unique_batches = data_getter.split('valid', self.args.groupkfold, seed=1+h)
            else:
                self.data, self.unique_labels, self.unique_batches = data_getter.get_variables()
            # self.unique_batches = np.unique(self.data['batches']['all'])

            self.make_samples_weights()
            self.set_arcloss()

            if self.args.n_calibration > 0:
                # take a few samples from train to be used for calibration. The indices
                self.data['inputs']['calibration'] = []
                self.data['labels']['calibration'] = []
                self.data['old_labels']['calibration'] = []
                self.data['names']['calibration'] = []
                self.data['cats']['calibration'] = []
                self.data['batches']['calibration'] = []

                for group in ['valid', 'test']:
                    indices = {label: np.argwhere(self.data['labels'][group] == label).flatten() for label in self.unique_labels}
                    calib_inds = np.concatenate([np.random.choice(indices[label], int(self.args.n_calibration/len(self.unique_labels)), replace=False) for label in self.unique_labels])

                    try:
                        self.data['inputs']['calibration'] += [np.stack(self.data['inputs'][group][calib_inds].tolist())]
                    except Exception as e:
                        print(f"Error stacking inputs for calibration: {e}")
                        continue
                    self.data['labels']['calibration'] += [self.data['labels'][group][calib_inds]]
                    self.data['old_labels']['calibration'] += [self.data['old_labels'][group][calib_inds]]
                    self.data['names']['calibration'] += [self.data['names'][group][calib_inds].tolist()]
                    self.data['cats']['calibration'] += [self.data['cats'][group][calib_inds].tolist()]
                    self.data['batches']['calibration'] += [self.data['batches'][group][calib_inds].tolist()]

                    # add them to train
                    self.data['inputs']['train'] = np.concatenate((self.data['inputs']['train'], self.data['inputs'][group][calib_inds]))
                    self.data['labels']['train'] = np.concatenate((self.data['labels']['train'], self.data['labels'][group][calib_inds]))
                    self.data['old_labels']['train'] = np.concatenate((self.data['old_labels']['train'], self.data['old_labels'][group][calib_inds]))
                    self.data['names']['train'] = np.concatenate((self.data['names']['train'], self.data['names'][group][calib_inds]))
                    self.data['cats']['train'] = np.concatenate((self.data['cats']['train'], self.data['cats'][group][calib_inds]))
                    self.data['batches']['train'] = np.concatenate((self.data['batches']['train'], self.data['batches'][group][calib_inds]))
                    # remove them from test
                    self.data['inputs'][group] = np.delete(self.data['inputs'][group], calib_inds, axis=0)
                    self.data['labels'][group] = np.delete(self.data['labels'][group], calib_inds, axis=0)
                    self.data['old_labels'][group] = np.delete(self.data['old_labels'][group], calib_inds, axis=0)
                    self.data['names'][group] = np.delete(self.data['names'][group], calib_inds, axis=0)
                    self.data['cats'][group] = np.delete(self.data['cats'][group], calib_inds, axis=0)
                    self.data['batches'][group] = np.delete(self.data['batches'][group], calib_inds, axis=0)
                self.data['inputs']['calibration'] = np.concatenate(self.data['inputs']['calibration'])
                self.data['labels']['calibration'] = np.concatenate(self.data['labels']['calibration'])
                self.data['old_labels']['calibration'] = np.concatenate(self.data['old_labels']['calibration'])
                self.data['names']['calibration'] = np.concatenate(self.data['names']['calibration'])
                self.data['cats']['calibration'] = np.concatenate(self.data['cats']['calibration'])
                self.data['batches']['calibration'] = np.concatenate(self.data['batches']['calibration'])

            print("Train Batches:", np.unique(self.data['batches']['train']))
            print("Valid Batches:", np.unique(self.data['batches']['valid']))
            print("Test Batches:", np.unique(self.data['batches']['test']))
            # event_acc is used to verify if the hparams have already been tested. If they were,
            # the best classification loss is retrieved and we go to the next trial
            # If thres > 0, features that are 0 for a proportion of samples smaller than thres are removed
            # data = self.keep_good_features(thres)

            # Transform the data with the chosen scaler
            # data = copy.deepcopy(self.data)
            # data, self.scaler = scale_data(scale, data, self.args.device)
            prototypes = {
                'combined': self.combined_prototypes,
                'class': self.class_prototypes,
                'batch': self.batch_prototypes
            }
            loaders = get_images_loaders(data=self.data,
                                         random_recs=self.args.random_recs,
                                         weighted_sampler=self.args.weighted_sampler,
                                         is_transform=params['is_transform'],
                                         samples_weights=self.samples_weights,
                                         epoch=0,
                                         unique_labels=self.unique_labels,
                                         triplet_dloss=self.args.dloss, bs=self.args.bs,
                                         prototypes_to_use=self.args.prototypes_to_use,
                                         prototypes=prototypes,
                                         size=self.args.new_size,
                                         normalize=self.args.normalize
                                         )
            # Initialize model
            self.model = Net(self.args.device, self.n_cats, self.n_batches,
                        model_name=self.args.model_name, is_stn=self.args.is_stn,
                        n_subcenters=self.n_batches)
            if self.args.classif_loss in ['ce', 'hinge']:
                self.shap_model = Net_deep_shap(self.args.device, self.n_cats, self.n_batches,
                                    model_name=self.args.model_name, is_stn=self.args.is_stn,
                                    n_subcenters=self.n_batches, shape=(3, self.args.new_size, self.args.new_size))
            else:
                self.shap_model = Net_shap(self.args.device, self.n_cats, self.n_batches,
                                    model_name=self.args.model_name, is_stn=self.args.is_stn,
                                    n_subcenters=self.n_batches)
            loggers['logger_cm'] = SummaryWriter(f'{self.complete_log_path}/cm')
            loggers['logger'] = SummaryWriter(f'{self.complete_log_path}/traces')
            # sceloss = nn.CrossEntropyLoss(label_smoothing=params['smoothing'])
            optimizer_model = get_optimizer(self.model, params['lr'], params['wd'], params['optimizer_type'])
            self.scheduler = ReduceLROnPlateau(optimizer_model, mode='max', factor=0.1, patience=10)  # Reduce on plateau

            values, best_values, _, best_traces = get_empty_dicts()
            early_stop_counter = 0
            best_vals = values
            warmup = 0
            for epoch in range(0, self.args.n_epochs):
                self.epoch = epoch
                if early_stop_counter >= self.args.early_stop:
                    if self.verbose > 0:
                        print('EARLY STOPPING.', epoch)
                    break
                lists, traces = get_empty_traces()
                self.model.train()
                if args.dloss == 'no':
                    train_groups = ['train']
                else:
                    train_groups = ['all', 'train']
                for group in train_groups:
                    if group == 'train' and epoch < warmup:
                        continue
                    _, best_lists1, _ = self.loop(group, optimizer_model, params['gamma'], loaders[group], lists, traces)
                if epoch < warmup:
                    values['all']['dloss'] += [np.mean(traces['all']['dloss'])]
                    add_to_neptune(values, run)
                    continue
                # Save images in run
                if self.log_neptune:
                    run['example_data'].upload(f'{self.complete_log_path}/{group}_data.png')
                    if self.args.fgsm:
                        run['example_adv_data'].upload(f'{self.complete_log_path}/{group}_adv_data.png')
                self.model.eval()
                self.set_prototypes('train', best_lists1)
                for group in ["valid", "test"]:
                    # if self.args.prototypes_to_use in ['combined', 'class']:
                    #     _, best_lists2, _, knn = self.predict_prototypes(group, loaders[group], lists, traces)
                    # else:
                    _, best_lists2, _, knn = self.predict(group, loaders[group], lists, traces)
                # put the best lists together. keys are all only in one dict
                best_lists = {**best_lists1, **best_lists2}
                self.set_prototypes('valid', best_lists)
                self.set_prototypes('test', best_lists)
                if 'all' in train_groups:
                    self.set_prototypes('all', best_lists)
                self.set_means_classes(best_lists)
                prototypes = {
                    'combined': self.combined_prototypes,
                    'class': self.class_prototypes,
                    'batch': self.batch_prototypes
                }
                loaders = get_images_loaders(data=self.data,
                                            random_recs=self.args.random_recs,
                                            weighted_sampler=self.args.weighted_sampler,
                                            is_transform=params['is_transform'],
                                            samples_weights=self.samples_weights,
                                            epoch=epoch+1,  # loaders for Next epoch
                                            unique_labels=self.unique_labels,
                                            triplet_dloss=self.args.dloss, bs=self.args.bs,
                                            prototypes_to_use=self.args.prototypes_to_use,
                                            prototypes=prototypes,
                                            size=self.args.new_size,
                                            normalize=self.args.normalize
                                            )
                values = log_traces(traces, values)
                self.scheduler.step(values['valid']['mcc'][-1])
                print('Current LR:', self.scheduler.get_last_lr())
                if self.log_tb:
                    try:
                        add_to_logger(values, loggers['logger'], epoch)
                    except:
                        print("Problem with add_to_logger!")
                if self.log_neptune:
                    add_to_neptune(values, run)
                if self.log_mlflow:
                    add_to_mlflow(values, epoch)
                if values['valid']['mcc'][-1] > self.best_mcc:
                    print(f"Best Classification Mcc Epoch {epoch}, "
                            f"Acc: test: {values['test']['acc'][-1]}, valid: {values['valid']['acc'][-1]}"
                            f"Mcc: test: {values['test']['mcc'][-1]}, valid: {values['valid']['mcc'][-1]}"
                            f"Classification "
                            f" valid loss: {values['valid']['closs'][-1]},"
                            f" test loss: {values['test']['closs'][-1]}, dloss: {values['all']['dloss'][-1]}")
                    self.best_mcc = values['valid']['mcc'][-1]
                    torch.save(self.model.state_dict(), f'{self.complete_log_path}/model.pth')
                    self.save_prototypes(self.complete_log_path)
                    self.save_samples_weights(self.complete_log_path)
                    self.log_predictions(best_lists, run, 0)

                    loggers['cm_logger'].add(best_lists)
                    best_vals = values.copy()
                    best_vals['rec_loss'] = self.best_loss
                    # best_vals['dom_loss'] = self.best_dom_loss
                    # best_vals['dom_acc'] = self.best_dom_acc
                    early_stop_counter = 0
                    print(values['valid']['mcc'][-1], self.best_mcc)
                    self.make_encoded_values()
                    self.cluster_and_visualize(run, best_lists, ['train', 'valid', 'test'],
                                                n_clusters=self.n_cats * self.n_batches)
                    if self.args.remove_noisy_samples:
                        data_getter.remove_noisy_samples(self.complete_log_path)
                    try:
                        assert len(data_getter.data['labels']['all']) == self.initial_nsamples
                    except:
                        print("Problem with removing noisy samples")
                        exit()
                else:
                    # if epoch > self.warmup:
                    early_stop_counter += 1

                if values['valid']['acc'][-1] > self.best_acc:
                    print(f"Best Classification Acc Epoch {epoch}, "
                            f"Acc: {values['test']['acc'][-1]}"
                            f"Mcc: {values['test']['mcc'][-1]}"
                            f"Classification "
                            f" valid loss: {values['valid']['closs'][-1]},"
                            f" test loss: {values['test']['closs'][-1]}, dloss: {values['all']['dloss'][-1]}")

                    self.best_acc = values['valid']['acc'][-1]

                if values['train']['closs'][-1] < self.best_closs:
                    print(f"Best Classification Loss Epoch {epoch}, "
                            f"Acc: {values['test']['acc'][-1]} "
                            f"Mcc: {values['test']['mcc'][-1]} "
                            f"Classification "
                            f"valid loss: {values['valid']['closs'][-1]}, "
                            f"test loss: {values['test']['closs'][-1]}, dloss: {values['all']['dloss'][-1]}")
                    self.best_closs = values['train']['closs'][-1]

            early_stop_counter = 0
            if self.args.n_calibration > 0:
                for epoch in range(0, self.args.n_epochs):
                    values, _, _, _ = get_empty_dicts()  # Pas élégant
                    self.epoch = epoch
                    if early_stop_counter >= self.args.early_stop:
                        if self.verbose > 0:
                            print('EARLY STOPPING.', epoch)
                        break
                    lists, traces = get_empty_traces()
                    lists['calibration'] = lists['all']
                    self.model.train()
                    self.model.to(self.args.device)
                    
                    for group in ['all', 'train', 'calibration']:
                        _, best_lists1, _ = self.loop_calibration(group, optimizer_model, params['gamma'], loaders[group], lists, traces)
                    self.model.eval()
                    for group in ["valid", "test"]:
                        _, best_lists2, _, _ = self.predict(group, loaders[group], lists, traces)
                    # put the best lists together. keys are all only in one dict
                    best_lists = {**best_lists1, **best_lists2}

                    self.set_prototypes('train', best_lists)
                    self.set_prototypes('valid', best_lists)
                    self.set_prototypes('test', best_lists)
                    self.set_prototypes('all', best_lists)
                    self.set_prototypes('calibration', best_lists)

                    self.model.eval()
                    for group in ["valid", "test"]:
                        _, best_lists, _, _ = self.predict(group, loaders[group], lists, traces)
                    prototypes = {
                        'combined': self.combined_prototypes,
                        'class': self.class_prototypes,
                        'batch': self.batch_prototypes
                    }
                    loaders = get_images_loaders(data=self.data,
                                                random_recs=self.args.random_recs,
                                                weighted_sampler=self.args.weighted_sampler,
                                                is_transform=params['is_transform'],
                                                samples_weights=self.samples_weights,
                                                epoch=epoch+1,
                                                unique_labels=self.unique_labels,
                                                triplet_dloss=self.args.dloss, bs=self.args.bs,
                                                prototypes_to_use=self.args.prototypes_to_use,
                                                prototypes=prototypes,
                                                size=self.args.new_size,
                                                normalize=self.args.normalize
                                                )

                    values = log_traces(traces, values)

                    # Add calib to the keys of values that are not calibration
                    keys = list(values.keys())
                    for key in keys:
                        if key in ['train', 'valid', 'test']:
                            values[key + '_calibration'] = values[key]
                            values.pop(key, None)
                            best_lists[key + '_calibration'] = best_lists[key]
                            best_lists.pop(key, None)
                      
                    if self.log_tb:
                        try:
                            add_to_logger(values, loggers['logger'], epoch)
                        except:
                            print("Problem with add_to_logger!")
                    if self.log_neptune:
                        add_to_neptune(values, run)
                    if self.log_mlflow:
                        add_to_mlflow(values, epoch)
                    if values['valid_calibration']['mcc'][-1] > self.best_mcc:
                        print(f"Best Classification Mcc Epoch {epoch}, "
                                f"Acc: test: {values['test_calibration']['acc'][-1]}, valid: {values['valid_calibration']['acc'][-1]}"
                                f"Mcc: test: {values['test_calibration']['mcc'][-1]}, valid: {values['valid_calibration']['mcc'][-1]}"
                                f"Classification "
                                f" valid loss: {values['valid_calibration']['closs'][-1]},"
                                f" test loss: {values['test_calibration']['closs'][-1]}, dloss: {values['all']['dloss'][-1]}")
                        self.best_mcc = values['valid_calibration']['mcc'][-1]
                        torch.save(self.model.state_dict(), f'{self.complete_log_path}/model.pth')
                        self.save_prototypes(self.complete_log_path)
                        self.save_samples_weights(self.complete_log_path)
                        best_vals = values.copy()
                        best_vals['rec_loss'] = self.best_loss
                        # After training, perform clustering and visualize
                        # Store all samples before clustering
                        self.make_encoded_values()
                        self.cluster_and_visualize(run, best_lists, 
                                                   ['train', 'valid', 'test'], 
                                                   n_clusters=self.n_cats * self.n_batches, update_lists=False)
                        early_stop_counter = 0
                    else:
                        print('missed', values['valid_calibration']['mcc'][-1], self.best_mcc)
                        early_stop_counter += 1
                    if values['valid_calibration']['acc'][-1] > self.best_acc:
                        print(f"Best Classification Acc Epoch {epoch}, "
                                f"Acc: {values['test_calibration']['acc'][-1]}"
                                f"Mcc: {values['test_calibration']['mcc'][-1]}"
                                f"Classification "
                                f" valid loss: {values['valid_calibration']['closs'][-1]},"
                                f" test loss: {values['test_calibration']['closs'][-1]}, dloss: {values['all']['dloss'][-1]}")
                        self.best_acc = values['valid_calibration']['acc'][-1]
                    if values['train_calibration']['closs'][-1] < self.best_closs:
                        print(f"Best Classification Loss Epoch {epoch}, "
                                f"Acc: {values['test_calibration']['acc'][-1]} "
                                f"Mcc: {values['test_calibration']['mcc'][-1]} "
                                f"Classification "
                                f"valid loss: {values['valid_calibration']['closs'][-1]}, "
                                f"test loss: {values['test_calibration']['closs'][-1]}, dloss: {values['all']['dloss'][-1]}")
                        self.best_closs = values['valid_calibration']['closs'][-1]

        lists, traces = get_empty_traces()
        # values, _, _, _ = get_empty_dicts()  # Pas élégant
        # Loading best model that was saved during training
        self.model.load_state_dict(torch.load(f'{self.complete_log_path}/model.pth', map_location=self.args.device))
        # Need another model because the other cant be use to get shap values
        self.shap_model.load_state_dict(torch.load(f'{self.complete_log_path}/model.pth', map_location=self.args.device))
        self.model.eval()
        self.shap_model.eval()
        with torch.no_grad():
            _, best_lists1, _ = self.loop('train', optimizer_model, params['gamma'], loaders['train'], lists, traces)
            for group in ["train", "valid", "test"]:
                _, best_lists2, traces, knn = self.predict(group, loaders[group], lists, traces)
        best_lists = {**best_lists1, **best_lists2}

        # Logging every model is taking too much resources and it makes it quite complicated to get information when
        # Too many runs have been made. This will make the notebook so much easier to work with
        if self.best_mcc > self.best_best_mcc:
            self.save_best_run(run, params, best_vals)
            # Save best model parameters and hyperparamters into a csv, including task

        run.stop()

        return self.best_mcc

    def save_best_run(self, run, params, best_vals):
        # Create csv file if not exists
        models_csv_path = f'logs/best_models/{self.args.task}/models.csv'
        log_dir = f'logs/best_models/{self.args.task}'

        if not os.path.exists(models_csv_path):
            os.makedirs(log_dir, exist_ok=True)
            with open(models_csv_path, 'w') as f:
                f.write('valid_mcc,model,path,n_neighbors,nsize,fgsm,n_calibration,loss,dloss,prototype,n_positives,n_negatives,normalize,task,complete_log_path\n')
                f.write(f'{self.best_mcc},{self.args.model_name},{self.path.split("/")[-1]},nn{params["n_neighbors"]},nsize{self.args.new_size},fsgm{self.args.fgsm},ncal{self.args.fgsm},' \
                        f'{self.args.classif_loss},{self.args.dloss},prototypes_{self.args.prototypes_to_use},' \
                        f'npos{self.args.n_positives},nneg{self.args.n_negatives},{self.args.normalize},{self.args.task},{self.complete_log_path}\n')
        else:
            with open(models_csv_path, 'r') as f:
                lines = f.readlines()

            line_found = False
            for i, line in enumerate(lines[1:]):  # skip header
                line_parts = line.strip().split(',')
                line_mcc = float(line_parts[0])
                if (line_parts[1:-1] == [self.args.model_name, f'{self.path.split("/")[-1]}', f'nn{params["n_neighbors"]}', f'nsize{self.args.new_size}', f'fsgm{self.args.fgsm}', f'ncal{self.args.fgsm}', 
                                       str(self.args.classif_loss), str(self.args.dloss), 
                                       f'prototypes_{self.args.prototypes_to_use}',
                                       f'npos{self.args.n_positives}', f'nneg{self.args.n_negatives}',
                                       self.args.task]):
                    if self.best_mcc >= line_mcc:
                        lines[i + 1] = f'{self.best_mcc},{self.args.model_name},{self.path.split("/")[-1]},nn{params["n_neighbors"]},nsize{self.args.new_size},fsgm{self.args.fgsm},' \
                                    f'ncal{self.args.fgsm},{self.args.classif_loss},{self.args.dloss},' \
                                    f'prototypes_{self.args.prototypes_to_use},npos{self.args.n_positives},' \
                                    f'nneg{self.args.n_negatives},{self.args.task},{self.complete_log_path}\n'
                    else:
                        lines[i + 1] = line
                    line_found = True
                    break

            if not line_found:
                lines.append(f'{self.best_mcc},{self.args.model_name},{self.path.split("/")[-1]},nn{params["n_neighbors"]}'\
                             f'nsize{self.args.new_size},fsgm{self.args.fgsm},'\
                             f'ncal{self.args.fgsm},{self.args.classif_loss},' \
                             f'{self.args.dloss},prototypes_{self.args.prototypes_to_use},' \
                             f'npos{self.args.n_positives},nneg{self.args.n_negatives},' \
                             f'{self.args.task},{self.complete_log_path}\n')

            with open(models_csv_path, 'w') as f:
                f.writelines(lines)

        params_str = f'{self.path.split("/")[-1]}/nsize{self.args.new_size}/fgsm{self.args.fgsm}/ncal{self.args.n_calibration}/{self.args.classif_loss}/' \
                 f'{self.args.dloss}/prototypes_{self.args.prototypes_to_use}/' \
                 f'npos{self.args.n_positives}/nneg{self.args.n_negatives}'
        model_dir = f'logs/best_models/{self.args.task}/{self.args.model_name}/{params_str}'

        try:
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir, ignore_errors=True)
            shutil.copytree(self.complete_log_path, model_dir)
        except Exception as e:
            print(f"Error copying model directory: {e}")

        self.best_best_mcc = self.best_mcc
        self.keep_best_run(run, best_vals)
        self.save_prototypes(model_dir)
        self.save_samples_weights(model_dir)

        # --- Update best model registry in MySQL ---
        registry_params = {
            'model_name': self.args.model_name,
            'fgsm': self.args.fgsm,
            'prototypes': self.args.prototypes_to_use,
            'npos': self.args.n_positives,
            'nneg': self.args.n_negatives,
            'dloss': self.args.dloss,
            'n_calibration': self.args.n_calibration,
            'normalize': self.args.normalize
        }
        update_best_model_registry(
            registry_params,
            accuracy=best_vals.get('acc', self.best_mcc),
            mcc=self.best_mcc,
            log_path=model_dir
        )

    # TODO Decide if valuable class method
    def save_prototypes(self, model_dir):
        with open(f'{model_dir}/prototypes.pkl', 'wb') as f:
            pickle.dump(self.prototypes, f)

    # TODO Decide if valuable class method
    def save_samples_weights(self, model_dir):
        with open(f'{model_dir}/sample_weights.pkl', 'wb') as f:
            pickle.dump(self.samples_weights, f)

    def keep_best_run(self, run, best_vals):
        self.best_params = get_best_params(run, best_vals)

    def set_arcloss(self):
        if self.args.classif_loss == 'arcface':
            self.arcloss = ArcFaceLoss(
                get_n_features(self.args.model_name),
                len(self.unique_labels),
                s=30,
                m=0.5,
                device=self.args.device
            )
        elif self.args.classif_loss == 'arcfacewithsubcenters':
            self.arcloss = ArcFaceLossWithSubcenters(
                get_n_features(self.args.model_name),
                len(self.unique_labels),
                s=30,
                m=0.5,
                num_subcenters=self.n_batches * 2,
                device=self.args.device
            )
        elif self.args.classif_loss == 'ArcFaceLossWithHSM':
            self.arcloss = ArcFaceLossWithHSM(
                get_n_features(self.args.model_name),
                len(self.unique_labels),
                s=30,
                m=0.5,
                num_subcenters=self.n_batches * 2,
                device=self.args.device
            )
        elif self.args.classif_loss == 'arcfacewithsubcentersHSM':
            self.arcloss = ArcFaceLossWithSubcentersHSM(
                get_n_features(self.args.model_name),
                len(self.unique_labels),
                s=30,
                m=0.5,
                num_subcenters=self.n_batches * 2,
                device=self.args.device
            )


    def make_encoded_values(self, groups=['train', 'valid', 'test']):
        # Get transform function from transform['test']
        # TODO should not have to redefine it again
        # Apply normalization only when explicitly requested via args.normalize
        transform_ops = [torchvision.transforms.ToTensor()]
        if str(self.args.normalize).lower() in ['yes', 'true', '1']:
            transform_ops.append(
                torchvision.transforms.Normalize(
                    [0.50818628, 0.40659687, 0.37182656],
                    [0.27270308, 0.25273249, 0.24185425]
                )
            )
        transform = torchvision.transforms.Compose(transform_ops)


        for group in groups:
            # Make cuda mini batches of batch size
            all_samples = []
            all_subcenters = []
            for i in range(0, len(self.all_samples['inputs'][group]), self.args.bs):
                transformed_samples = []
                batch = self.all_samples['inputs'][group][i:i + self.args.bs]
                for j in range(len(batch)):
                    transformed_samples.append(transform(batch[j]))
                transformed_samples = torch.stack(transformed_samples).to(self.args.device)
                # Use transformed_samples for encoding
                try:
                    all_samples += [self.model(transformed_samples)[0]]
                except Exception as e:
                    print(f"Error occurred: {e}")
                    exit()
                
                if 'arcfacewithsubcenters' in self.args.classif_loss:
                    _, subcenters = self.arcloss(
                        all_samples[-1],
                        torch.Tensor(
                            self.all_samples['cats'][group][i:i + self.args.bs]
                        ).long().to(self.args.device))
                    all_subcenters += [subcenters.detach().cpu().numpy()]
                all_samples[-1] = all_samples[-1].detach().cpu().numpy()
            self.all_samples['encoded_values'][group] = np.concatenate(all_samples)

            if 'arcfacewithsubcenters' in self.args.classif_loss:
                self.all_samples['subcenters'][group] = np.concatenate(all_subcenters)

    def log_predictions(self, best_lists, run, step):
        cats, labels, preds, scores, names = [{'train': [], 'valid': [], 'test': []} for _ in range(5)]
        for group in ['valid', 'test']:
            cats[group] = np.concatenate(best_lists[group]['cats'])
            labels[group] = np.concatenate(best_lists[group]['labels'])
            # Use KNN outputs directly (assume best_lists[group]['preds'] contains KNN probabilities or one-hot)
            scores[group] = np.concatenate(best_lists[group]['preds'])
            preds[group] = scores[group].argmax(1)
            names[group] = np.concatenate(best_lists[group]['names'])
            score_colnames = [f'probs_{l}' for l in self.unique_labels]
            colnames = ['label'] + score_colnames + ['pred', 'name']
            df = pd.DataFrame(
                np.concatenate((labels[group].reshape(-1, 1),
                               scores[group],
                               np.array([self.unique_labels[x] for x in preds[group]]).reshape(-1, 1),
                               names[group].reshape(-1, 1)), 1),
                columns=colnames
            )
            df.to_csv(f'{self.complete_log_path}/{group}_predictions.csv', index=False)
            if self.log_neptune:
                run[f"{group}_predictions"].track_files(f'{self.complete_log_path}/{group}_predictions.csv')
                run[f'{group}_AUC'] = metrics.roc_auc_score(y_true=to_categorical(cats[group], len(self.unique_labels)), y_score=scores[group], multi_class='ovr')
                save_roc_curve(
                    scores[group],
                    to_categorical(cats[group], len(self.unique_labels)),
                    self.unique_labels,
                    f'{self.complete_log_path}/{group}_roc_curve.png',
                    binary=False,
                    acc=metrics.accuracy_score(cats[group], np.array(scores[group].argmax(1))),
                    mlops='neptune',
                    epoch=step,
                    logger=None
                )
                run[f'{group}_ROC'].upload(f'{self.complete_log_path}/{group}_roc_curve.png')
            if self.log_mlflow:
                mlflow.log_metric(f'{group}_AUC',
                            metrics.roc_auc_score(y_true=to_categorical(cats[group], len(self.unique_labels)), y_score=scores[group], multi_class='ovr'),
                            step=step)

                # TODO ADD OTHER FINAL METRICS HERE
                mlflow.log_metric(f'{group}_acc',
                            metrics.accuracy_score(cats[group], scores[group].argmax(1)),
                            step=step)
                mlflow.log_metric(f'{group}_mcc',
                            MCC(cats[group], scores[group].argmax(1)),
                            step=step)

                save_roc_curve(
                    scores[group],
                    to_categorical(cats[group], len(self.unique_labels)),
                    self.unique_labels,
                    f'{self.complete_log_path}/{group}_roc_curve.png',
                    binary=False,
                    acc=metrics.accuracy_score(cats[group], np.array(scores[group].argmax(1))),
                    mlops='mlflow',
                    epoch=step,
                    logger=None
                )
    
    def get_losses(self, enc, to_rec, not_to_rec, pos_batch_sample, neg_batch_sample, dpreds, domains, labels, group):
        if self.args.classif_loss == 'cosine':
            classif_loss = 1 - torch.nn.functional.cosine_similarity(enc, self.model(to_rec)).mean()
        elif self.args.classif_loss == 'triplet' or self.args.classif_loss == 'softmax_contrastive':
            not_to_rec = not_to_rec.to(self.args.device).float()
            not_to_rec = not_to_rec.requires_grad_(True)
            if self.args.prototypes_to_use in ['no', 'batch'] or self.epoch == 0: # On the first epoch, we don't use prototypes but regular triplet loss	
                try:
                    pos_enc, _ = self.model(to_rec)
                except:
                    pass
                neg_enc, _ = self.model(not_to_rec)
            else:
                # Both are already encoded when prototypes are registered
                pos_enc = to_rec
                neg_enc = not_to_rec
            classif_loss = self.triplet_loss(enc, pos_enc, neg_enc)
        elif self.args.classif_loss in ['ce', 'hinge']:
            preds = self.model.linear(enc)
            labels = labels.to(self.args.device).long()
            classif_loss = self.celoss(preds, labels)

        elif 'arcfacewithsubcenters' in self.args.classif_loss:
            classif_loss, subcenters = self.arcloss(enc, labels)
            lists[group]['subcenters'] += [subcenters.detach().cpu().numpy()]

        elif 'arcface' in self.args.classif_loss:
            classif_loss = self.arcloss(enc, labels)
        else:
            exit(f"Classif loss {self.args.classif_loss} not implemented")

        if self.args.dloss == 'DANN':
            dloss = self.dceloss(dpreds, domains)

        elif self.args.dloss in ['inverseTriplet', 'inverse_softmax_contrastive'] and self.triplet_dloss is not None:
            pos_batch_sample, neg_batch_sample = pos_batch_sample.to(
                self.args.device).float(), neg_batch_sample.to(self.args.device).float()
            pos_batch_sample, neg_batch_sample = pos_batch_sample.requires_grad_(True), neg_batch_sample.requires_grad_(True)
            if self.args.prototypes_to_use in ['both', 'batch'] and self.epoch > 0:
                pos_enc = pos_batch_sample
                neg_enc = neg_batch_sample
            else:
                pos_enc, _ = self.model(pos_batch_sample)
                neg_enc, _ = self.model(neg_batch_sample)
            dloss = self.triplet_dloss(enc, neg_enc, pos_enc)
            # domain = domain.argmax(1)
        else:
            dloss = torch.Tensor([0]).to(self.args.device)[0]
        return classif_loss, dloss

    def loop(self, group, optimizer_model, gamma, loader, lists, traces):
        """

        Args:
            group: Which set? Train, valid or test
            optimizer_model: Object that contains the optimizer for the autoencoder
            ae: AutoEncoder (pytorch model, inherits nn.Module)
            sceloss: torch.nn.CrossEntropyLoss instance
            triplet_loss: torch.nn.TripletMarginLoss instance
            loader: torch.utils.data.DataLoader
            lists: List keeping informations on the current run
            traces: List keeping scores on the current run
            nu: hyperparameter controlling the importance of the classification loss

        Returns:

        """
        # If group is train and nu = 0, then it is not training. valid can also have sampling = True
        # model, classifier, dann = models['model'], models['classifier'], models['dann']
        # triplet_loss = nn.TripletMarginLoss(margin, p=2)
        classif_loss = torch.tensor([0]).to(self.args.device)[0]
        # with tqdm(total=len(loader), position=0, leave=True) as pbar:
        for i, batch in enumerate(loader):
            # if group in ['train']:
            if self.model.training:
                optimizer_model.zero_grad()
            data, names, labels, _, old_labels, domain, to_rec, not_to_rec, pos_batch_sample, neg_batch_sample = batch
            data = data.to(self.args.device).float()
            data = data.requires_grad_(True)
            to_rec = to_rec.to(self.args.device).float()
            to_rec = to_rec.requires_grad_(True)
            not_to_rec = not_to_rec.to(self.args.device).float()
            not_to_rec = not_to_rec.requires_grad_(True)
            pos_batch_sample = pos_batch_sample.to(self.args.device).float()
            pos_batch_sample = pos_batch_sample.requires_grad_(True)
            neg_batch_sample = neg_batch_sample.to(self.args.device).float()
            neg_batch_sample = neg_batch_sample.requires_grad_(True)

            domains = to_categorical(domain.long(), self.n_batches).squeeze().float().to(self.args.device)

            enc, dpreds = self.model(data)

            lists[group]['domains'] += [np.array([self.unique_batches[np.argmax(d)] for d in domains.detach().cpu().numpy()])]
            lists[group]['classes'] += [labels.detach().cpu().numpy()]
            if group == 'train':
                lists[group]['preds'] += [dpreds.detach().cpu().numpy()]
            lists[group]['encoded_values'] += [enc.view(enc.shape[0], -1).detach().cpu().numpy()]
            lists[group]['names'] += [names]
            # lists[group]['inputs'] += [data.view(data.shape[0], -1).detach().cpu().numpy()]
            lists[group]['labels'] += [np.array(
                [self.unique_labels[x] for x in labels.detach().cpu().numpy()])]
            lists[group]['old_labels'] += [np.array(old_labels)]
            traces[group]['dom_acc'] += [np.mean([0 if pred != dom else 1 for pred, dom in
                                                zip(dpreds.detach().cpu().numpy().argmax(1),
                                                    domains.argmax(1).detach().cpu().numpy())])]
            lists[group]['cats'] += [labels.detach().cpu().numpy()]
            if self.model.training:
                classif_loss, dloss = self.get_losses(enc, to_rec, not_to_rec, pos_batch_sample,
                                                    neg_batch_sample, dpreds, domains, labels, group)
                traces[group]['closs'] += [classif_loss.item()]
                traces[group]['dloss'] += [dloss.item()]
                if group in ['train']:
                    if i == 0:
                        save_tensor(data[0], f'{self.complete_log_path}/{group}_data.png')
                    # total_loss = classif_loss + gamma * dloss
                    classif_loss.backward()
                    # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                elif group in ['all']:
                    if gamma > 0:
                        dloss = gamma * dloss
                        dloss.backward()
                if self.args.fgsm:
                    adv_data = data + self.params['epsilon'] * data.grad.sign()
                    adv_data = torch.clamp(adv_data, -1, 1)
                    enc, dpreds = self.model(adv_data)
                    enc = enc.detach()
                    if group in ['train']:
                        if i == 0:
                            save_tensor(adv_data[0], f'{self.complete_log_path}/{group}_adv_data.png')
                        if self.args.classif_loss not in ['ce', 'hinge', 'arcface', 'arcfacewithsubcenters']:
                            adv_to_rec = to_rec + self.params['epsilon'] * to_rec.grad.sign()
                            adv_to_rec = torch.clamp(adv_to_rec, -1, 1)
                            adv_not_to_rec = not_to_rec + self.params['epsilon'] * not_to_rec.grad.sign()
                            adv_not_to_rec = torch.clamp(adv_not_to_rec, -1, 1)
                            adv_to_rec = adv_to_rec.detach()
                            adv_not_to_rec = adv_not_to_rec.detach()
                        else:
                            adv_to_rec = to_rec
                            adv_not_to_rec = not_to_rec
                        adv_classif_loss, _ = self.get_losses(enc, adv_to_rec, adv_not_to_rec, pos_batch_sample, neg_batch_sample, dpreds, domains, labels, group)
                        (0.1 * adv_classif_loss).backward()
                        del adv_data, adv_to_rec, adv_not_to_rec, adv_classif_loss
                    elif group in ['all']:
                        if self.args.classif_loss in ['inverseTriplet', 'inverse_softmax_contrastive']:
                            adv_pos_batch_sample = pos_batch_sample + self.params['epsilon'] * pos_batch_sample.grad.sign()
                            adv_pos_batch_sample = torch.clamp(adv_pos_batch_sample, -1, 1)
                            adv_neg_batch_sample = neg_batch_sample + self.params['epsilon'] * neg_batch_sample.grad.sign()
                            adv_neg_batch_sample = torch.clamp(adv_neg_batch_sample, -1, 1)
                            adv_pos_batch_sample = adv_pos_batch_sample.detach()
                            adv_neg_batch_sample = adv_neg_batch_sample.detach()
                        else:
                            adv_pos_batch_sample = pos_batch_sample
                            adv_neg_batch_sample = neg_batch_sample
                        _, adv_dloss = self.get_losses(enc, to_rec, not_to_rec, adv_pos_batch_sample, adv_neg_batch_sample, dpreds, domains, labels, group)
                        (0.1 * adv_dloss).backward()
                        del adv_data, adv_pos_batch_sample, adv_neg_batch_sample, adv_dloss
                del dloss

                optimizer_model.step()
            del data, to_rec, not_to_rec, pos_batch_sample, neg_batch_sample, enc, dpreds, domains
        return classif_loss.detach(), lists, traces

    def loop_calibration(self, group, optimizer_model, gamma, loader, lists, traces):
        """

        Args:
            group: Which set? Train, valid or test
            optimizer_model: Object that contains the optimizer for the autoencoder
            ae: AutoEncoder (pytorch model, inherits nn.Module)
            sceloss: torch.nn.CrossEntropyLoss instance
            triplet_loss: torch.nn.TripletMarginLoss instance
            loader: torch.utils.data.DataLoader
            lists: List keeping informations on the current run
            traces: List keeping scores on the current run
            nu: hyperparameter controlling the importance of the classification loss

        Returns:

        """
        # If group is train and nu = 0, then it is not training. valid can also have sampling = True
        # model, classifier, dann = models['model'], models['classifier'], models['dann']
        # triplet_loss = nn.TripletMarginLoss(margin, p=2)
        classif_loss = None
        # with tqdm(total=len(loader), position=0, leave=True) as pbar:
        for i, batch in enumerate(loader):
            # if group in ['train']:
            if self.model.training:
                optimizer_model.zero_grad()
            data, names, labels, _, old_labels, domain, to_rec, not_to_rec, pos_batch_sample, neg_batch_sample = batch
            data = data.to(self.args.device).float()
            to_rec = to_rec.to(self.args.device).float()
            domains = to_categorical(domain.long(), self.n_batches).squeeze().float().to(self.args.device)
            enc, dpreds = self.model(data)

            # if self.args.classif_loss == 'cosine':
            #     classif_loss = 1 - torch.nn.functional.cosine_similarity(enc, self.model(to_rec)).mean()
            # elif self.args.classif_loss == 'triplet' or self.args.classif_loss == 'softmax_contrastive':
            #     not_to_rec = not_to_rec.to(self.args.device).float()
            #     if not self.args.prototypes_to_use:  # or self.epoch == 0:
            #         pos_enc, _ = self.model(to_rec)
            #         neg_enc, _ = self.model(not_to_rec)
            #     else:
            #         # Both are already encoded when prototypes are registered
            #         pos_enc = to_rec
            #         neg_enc = not_to_rec
            #     classif_loss = self.triplet_loss(enc, pos_enc, neg_enc)
            # if self.args.dloss == 'DANN':
            #     dloss = self.celoss(dpreds, domains)
            # elif self.args.dloss == 'inverseTriplet':
            #     pos_batch_sample, neg_batch_sample = neg_batch_sample.to(
            #         self.args.device).float(), pos_batch_sample.to(self.args.device).float()
            #     pos_enc, _ = self.model(pos_batch_sample)
            #     neg_enc, _ = self.model(neg_batch_sample)
            #     dloss = self.triplet_dloss(enc, neg_enc, pos_enc)
            #     # domain = domain.argmax(1)
            # else:
            #     dloss = torch.Tensor([0]).to(self.args.device)[0]
            classif_loss, dloss = self.get_losses(enc, to_rec, not_to_rec, pos_batch_sample,
                                                  neg_batch_sample, dpreds, domains, labels, group)

            lists[group]['domains'] += [np.array([self.unique_batches[np.argmax(d)] for d in domains.detach().cpu().numpy()])]
            lists[group]['classes'] += [labels.detach().cpu().numpy()]
            if group == 'train':
                lists[group]['preds'] += [dpreds.detach().cpu().numpy()]
            lists[group]['encoded_values'] += [enc.view(enc.shape[0], -1).detach().cpu().numpy()]
            lists[group]['names'] += [names]
            # lists[group]['inputs'] += [data.view(data.shape[0], -1).detach().cpu().numpy()]
            lists[group]['labels'] += [np.array(
                [self.unique_labels[x] for x in labels.detach().cpu().numpy()])]
            lists[group]['old_labels'] += [np.array(old_labels)]
            traces[group]['dom_acc'] += [np.mean([0 if pred != dom else 1 for pred, dom in
                                                zip(dpreds.detach().cpu().numpy().argmax(1),
                                                    domains.argmax(1).detach().cpu().numpy())])]
            traces[group]['closs'] += [classif_loss.item()]
            traces[group]['dloss'] += [dloss.item()]
            lists[group]['cats'] += [labels.detach().cpu().numpy()]
            if self.model.training:
                if group in ['calibration', 'train']:
                    # total_loss = classif_loss + gamma * dloss
                    classif_loss.backward()
                    # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                elif group in ['all']:
                    if gamma > 0:
                        dloss = gamma * dloss
                        dloss.backward()
                optimizer_model.step()
            # pbar.update(1)
        return classif_loss, lists, traces

    def predict(self, group, loader, lists, traces):
        """

        Args:
            group: Which set? Train, valid or test
            loader: torch.utils.data.DataLoader
            lists: List keeping informations on the current run
            traces: List keeping scores on the current run
            nu: hyperparameter controlling the importance of the classification loss

        Returns:

        """
        # If group is train and nu = 0, then it is not training. valid can also have sampling = True
        # model, classifier, dann = models['model'], models['classifier'], models['dann']
        # import nearest_neighbors from sklearn
        classif_loss = None
        train_encs = np.concatenate(lists['train']['encoded_values'])
        train_cats = np.concatenate(lists['train']['cats'])
        if self.args.classif_loss not in ['ce', 'hinge']:
            if train_encs.shape[0] >= self.params['n_neighbors']:
                KNeighborsClassifier = KNN(n_neighbors=self.params['n_neighbors'], metric='minkowski')
            else:
                KNeighborsClassifier = KNN(n_neighbors=train_encs.shape[0], metric='minkowski')
            KNeighborsClassifier.fit(train_encs, train_cats)
        else:
            KNeighborsClassifier = self.model
        # plot_decision_boundary(KNeighborsClassifier, train_encs, train_cats)
        # plt.savefig(f'{self.complete_log_path}/decision_boundary.png')
        # with tqdm(total=len(loader), position=0, leave=True) as pbar:
        for i, batch in enumerate(loader):
            # if group in ['train']:
            data, names, labels, _, old_labels, domain, to_rec, not_to_rec, pos_batch_sample, neg_batch_sample = batch
            data = data.to(self.args.device).float()
            to_rec = to_rec.to(self.args.device).float()
            domains = to_categorical(domain.long(), self.n_batches).squeeze().float().to(self.args.device)
            enc, dpreds = self.model(data)
            if self.args.classif_loss not in ['ce', 'hinge']:
                preds = KNeighborsClassifier.predict(enc.view(enc.shape[0], -1).detach().cpu().numpy())
                proba = KNeighborsClassifier.predict_proba(enc.view(enc.shape[0], -1).detach().cpu().numpy())
            else:
                out = self.model.linear(enc)
                preds = out.argmax(1).detach().cpu().numpy()
                proba = torch.softmax(out, 1).detach().cpu().numpy()
            if self.args.dloss == 'DANN':
                dloss = self.dceloss(dpreds, domains)

            elif self.args.dloss in ['inverseTriplet', 'inverse_softmax_contrastive'] and self.args.prototypes_to_use not in ['both', 'batch'] and self.triplet_dloss is not None:
                pos_batch_sample, neg_batch_sample = neg_batch_sample.to(
                    self.args.device).float(), pos_batch_sample.to(self.args.device).float()
                pos_enc, _ = self.model(pos_batch_sample)
                neg_enc, _ = self.model(neg_batch_sample)
                dloss = self.triplet_dloss(enc, neg_enc, pos_enc)
                # domain = domain.argmax(1)
            else:
                dloss = torch.Tensor([0]).to(self.args.device)[0]

            lists[group]['domains'] += [np.array([self.unique_batches[np.argmax(d)] for d in domains.detach().cpu().numpy()])]
            lists[group]['classes'] += [labels.detach().cpu().numpy()]
            lists[group]['encoded_values'] += [enc.view(enc.shape[0], -1).detach().cpu().numpy()]

            lists[group]['names'] += [names]
            lists[group]['cats'] += [labels.detach().cpu().numpy()]
            # lists[group]['inputs'] += [data.view(data.shape[0], -1).detach().cpu().numpy()]
            lists[group]['labels'] += [np.array(
                [self.unique_labels[x] for x in labels.detach().cpu().numpy()])]
            lists[group]['old_labels'] += [np.array(old_labels)]
            traces[group]['acc'] += [np.mean([0 if pred != dom else 1 for pred, dom in
                                                zip(preds,
                                                    labels.detach().cpu().numpy())])]
            traces[group]['dom_acc'] += [np.mean([0 if pred != dom else 1 for pred, dom in
                                                zip(dpreds.detach().cpu().numpy().argmax(1),
                                                    domains.argmax(1).detach().cpu().numpy())])]
            traces[group]['closs'] += [0]
            traces[group]['dloss'] += [dloss.item()]
            lists[group]['preds'] += [proba]
            lists[group]['inputs'] += [data.detach().cpu()]

        try:
            traces[group]['mcc'] += [np.round(
                MCC(np.concatenate(lists[group]['cats']), np.concatenate(lists[group]['preds']).argmax(1))
                , 3)
            ]
        except:
            pass
        # Add other metrics here
        if len(np.unique(np.concatenate(lists[group]['cats']))) > 1:
            try:
                traces[group]['tpr'] += [np.round(
                    metrics.recall_score(np.concatenate(lists[group]['cats']), np.concatenate(lists[group]['preds']).argmax(1), average='macro', pos_label=1)
                    , 3)
                ]
                traces[group]['tnr'] += [np.round(
                    metrics.recall_score(np.concatenate(lists[group]['cats']), np.concatenate(lists[group]['preds']).argmax(1), average='macro', pos_label=0)
                    , 3)
                ]
                traces[group]['ppv'] += [np.round(
                    metrics.precision_score(np.concatenate(lists[group]['cats']), np.concatenate(lists[group]['preds']).argmax(1), average='macro', pos_label=1)
                    , 3)
                ]
                traces[group]['npv'] += [np.round(
                    metrics.precision_score(np.concatenate(lists[group]['cats']), np.concatenate(lists[group]['preds']).argmax(1), average='macro', pos_label=0)
                    , 3)
                ]
            except:
                pass

        if group == 'test':
            # plot the PCA of the encoded values
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            valid_encs = np.concatenate(lists['valid']['encoded_values'])
            valid_cats = np.concatenate(lists['valid']['cats'])
            try:
                test_encs = np.concatenate(lists['test']['encoded_values'])
            except:
                pass
            test_cats = np.concatenate(lists['test']['cats'])
            train_encs_pca = pca.fit_transform(train_encs)
            valid_encs_pca = pca.transform(valid_encs)
            test_encs_pca = pca.transform(test_encs)
            # plot the values and save img
            plt.figure(figsize=(10, 10))
            plt.scatter(train_encs_pca[:, 0], train_encs_pca[:, 1], c=train_cats, marker='o')
            plt.scatter(valid_encs_pca[:, 0], valid_encs_pca[:, 1], c=valid_cats, marker='x')
            plt.scatter(test_encs_pca[:, 0], test_encs_pca[:, 1], c=test_cats, marker='^')
            plt.colorbar()
            plt.title('PCA of encoded values')
            plt.savefig(f'{self.complete_log_path}/pca_batches.png')

        return None, lists, traces, KNeighborsClassifier

    def predict_prototypes(self, group, loader, lists, traces):
        """

        Args:
            group: Which set? Train, valid or test
            loader: torch.utils.data.DataLoader
            lists: List keeping informations on the current run
            traces: List keeping scores on the current run
            nu: hyperparameter controlling the importance of the classification loss

        Returns:

        """
        # If group is train and nu = 0, then it is not training. valid can also have sampling = True
        # model, classifier, dann = models['model'], models['classifier'], models['dann']
        # import nearest_neighbors from sklearn
        classif_loss = None
        train_prototypes = np.stack([self.class_prototypes['train'][x] for x in self.class_prototypes['train']])
        train_labels = list(self.class_prototypes['train'].keys())
        train_cats = [np.argwhere(x == self.unique_labels).flatten() for x in train_labels]
        KNeighborsClassifier = KNN(n_neighbors=1, metric='minkowski')
        KNeighborsClassifier.fit(train_prototypes, train_cats)
        # plot_decision_boundary(KNeighborsClassifier, train_encs, train_cats)
        # plt.savefig(f'{self.complete_log_path}/decision_boundary.png')
        # with tqdm(total=len(loader), position=0, leave=True) as pbar:
        for i, batch in enumerate(loader):
            # if group in ['train']:
            data, names, labels, _, old_labels, domain, to_rec, not_to_rec, pos_batch_sample, neg_batch_sample = batch
            data = data.to(self.args.device).float()
            to_rec = to_rec.to(self.args.device).float()
            domains = to_categorical(domain.long(), self.n_batches).squeeze().float().to(self.args.device)
            enc, dpreds = self.model(data)
            if self.args.classif_loss not in ['ce', 'hinge']:
                preds = KNeighborsClassifier.predict(enc.view(enc.shape[0], -1).detach().cpu().numpy())
                proba = KNeighborsClassifier.predict_proba(enc.view(enc.shape[0], -1).detach().cpu().numpy())
            else:
                out = self.model.linear(enc)
                preds = out.argmax(1).detach().cpu().numpy()
                proba = torch.softmax(out, 1).detach().cpu().numpy()
            if self.args.dloss == 'DANN':
                dloss = self.dceloss(dpreds, domains)

            elif self.args.dloss in ['inverseTriplet', 'inverse_softmax_contrastive'] and self.args.prototypes_to_use not in ['both', 'batch'] and self.triplet_dloss is not None:
                pos_batch_sample, neg_batch_sample = neg_batch_sample.to(
                    self.args.device).float(), pos_batch_sample.to(self.args.device).float()
                pos_enc, _ = self.model(pos_batch_sample)
                neg_enc, _ = self.model(neg_batch_sample)
                dloss = self.triplet_dloss(enc, neg_enc, pos_enc)
                # domain = domain.argmax(1)
            else:
                dloss = torch.Tensor([0]).to(self.args.device)[0]

            lists[group]['domains'] += [np.array([self.unique_batches[np.argmax(d)] for d in domains.detach().cpu().numpy()])]
            lists[group]['classes'] += [labels.detach().cpu().numpy()]
            lists[group]['encoded_values'] += [enc.view(enc.shape[0], -1).detach().cpu().numpy()]

            lists[group]['names'] += [names]
            lists[group]['cats'] += [labels.detach().cpu().numpy()]
            # lists[group]['inputs'] += [data.view(data.shape[0], -1).detach().cpu().numpy()]
            lists[group]['labels'] += [np.array(
                [self.unique_labels[x] for x in labels.detach().cpu().numpy()])]
            lists[group]['old_labels'] += [np.array(old_labels)]
            traces[group]['acc'] += [np.mean([0 if pred != dom else 1 for pred, dom in
                                                zip(preds,
                                                    labels.detach().cpu().numpy())])]
            traces[group]['dom_acc'] += [np.mean([0 if pred != dom else 1 for pred, dom in
                                                zip(dpreds.detach().cpu().numpy().argmax(1),
                                                    domains.argmax(1).detach().cpu().numpy())])]
            traces[group]['closs'] += [0]
            traces[group]['dloss'] += [dloss.item()]
            lists[group]['preds'] += [proba]
            lists[group]['inputs'] += [data.detach().cpu()]

        try:
            traces[group]['mcc'] += [np.round(
                MCC(np.concatenate(lists[group]['cats']), np.concatenate(lists[group]['preds']).argmax(1))
                , 3)
            ]
        except:
            pass
        # Add other metrics here
        if len(np.unique(np.concatenate(lists[group]['cats']))) > 1:
            try:
                traces[group]['tpr'] += [np.round(
                    metrics.recall_score(np.concatenate(lists[group]['cats']), np.concatenate(lists[group]['preds']).argmax(1), average='macro', pos_label=1)
                    , 3)
                ]
                traces[group]['tnr'] += [np.round(
                    metrics.recall_score(np.concatenate(lists[group]['cats']), np.concatenate(lists[group]['preds']).argmax(1), average='macro', pos_label=0)
                    , 3)
                ]
                traces[group]['ppv'] += [np.round(
                    metrics.precision_score(np.concatenate(lists[group]['cats']), np.concatenate(lists[group]['preds']).argmax(1), average='macro', pos_label=1)
                    , 3)
                ]
                traces[group]['npv'] += [np.round(
                    metrics.precision_score(np.concatenate(lists[group]['cats']), np.concatenate(lists[group]['preds']).argmax(1), average='macro', pos_label=0)
                    , 3)
                ]
            except:
                pass

        if group == 'test':
            # plot the PCA of the encoded values
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            valid_encs = np.concatenate(lists['valid']['encoded_values'])
            valid_cats = np.concatenate(lists['valid']['cats'])
            try:
                test_encs = np.concatenate(lists['test']['encoded_values'])
            except:
                pass
            test_cats = np.concatenate(lists['test']['cats'])
            train_encs = np.concatenate(lists['train']['encoded_values'])
            train_cats = np.concatenate(lists['train']['cats'])
            train_encs_pca = pca.fit_transform(train_encs)
            valid_encs_pca = pca.transform(valid_encs)
            test_encs_pca = pca.transform(test_encs)
            # plot the values and save img
            plt.figure(figsize=(10, 10))
            plt.scatter(train_encs_pca[:, 0], train_encs_pca[:, 1], c=train_cats, marker='o')
            plt.scatter(valid_encs_pca[:, 0], valid_encs_pca[:, 1], c=valid_cats, marker='x')
            plt.scatter(test_encs_pca[:, 0], test_encs_pca[:, 1], c=test_cats, marker='^')
            plt.colorbar()
            plt.title('PCA of encoded values')
            plt.savefig(f'{self.complete_log_path}/pca_batches.png')

        return None, lists, traces, KNeighborsClassifier

    def cluster_and_visualize(self, run, lists, groups, n_clusters=5, min_cluster_size=10, update_lists=True):
        """
        Perform clustering on all samples (including previously removed ones), visualize the clusters,
        and dynamically filter samples that belong to small clusters.
        
        Args:
            lists: Dictionary containing the encoded values, labels, etc.
            n_clusters: Number of clusters to form.
            min_cluster_size: Minimum size of a cluster to be considered valid.
        
        Returns:
            lists: Updated lists with filtered samples.
        """
        
        # Concatenate all samples (including previously removed ones)
        all_encoded_values = np.concatenate([
            self.all_samples['encoded_values'][group] for group in groups
        ])
        # all_encoded_values = all_encoded_values.reshape(all_encoded_values.shape[0], -1)
        all_labels = np.concatenate([self.all_samples['labels'][group] for group in groups])
        all_cats = np.concatenate([self.all_samples['cats'][group] for group in groups])
        all_names = np.concatenate([self.all_samples['names'][group] for group in groups])
        all_subcenters = np.concatenate([self.all_samples['subcenters'][group] for group in groups])
        all_batches = np.concatenate([self.all_samples['batches'][group] for group in groups])
        all_batches_cat = np.array([
            np.argwhere(x == self.unique_batches) for x in all_batches
        ]).flatten()
        
        # Perform K-Means clustering with sub-centers as initial centroids
        subcenters = self.model.get_subcenters().detach().cpu().numpy()
        subcenters = subcenters.reshape(-1, subcenters.shape[-1])
        kmeans = KMeans(n_clusters=n_clusters, init=subcenters, random_state=42)
        try:
            cluster_labels = kmeans.fit_predict(all_encoded_values)
        except:
            pass
        for ordin in ["PCA", "UMAP"]:
            # Reduce dimensionality for visualization
            if ordin == "PCA":
                pca = PCA(n_components=3)
            elif ordin == "Isomap":
                pca = Isomap(n_components=3)
            elif ordin == "MDS":
                pca = MDS(n_components=3)
            elif ordin == "UMAP":
                pca = UMAP(n_components=3)
            else:
                raise ValueError("Invalid ordination method")
            reduced_data = pca.fit_transform(all_encoded_values)
            class_prototypes = copy.deepcopy(self.class_prototypes)
            batch_prototypes = copy.deepcopy(self.batch_prototypes)
            for group in groups:
                for clas in np.unique(all_labels):
                    class_prototypes[group][clas] = pca.transform(class_prototypes[group][clas].reshape(1, -1))
                for batch in np.unique(all_batches):
                    try:
                        batch_prototypes[group][batch] = pca.transform(batch_prototypes[group][batch].reshape(1, -1))
                    except:
                        # Remove batch from batch_prototypes if it is not group
                        batch_prototypes[group].pop(batch, None)
            batch_prototypes.pop('all', None)
            class_prototypes.pop('all', None)
            # Get percentage of variance explained by the two components
            if ordin == "PCA":
                explained_variance = np.round(pca.explained_variance_ratio_ * 100, 2)
            else:
                explained_variance = None

            if 'subcenter' in self.args.classif_loss:        
                plot_pca(reduced_data, all_subcenters, all_batches,
                         explained_variance, self.complete_log_path,
                         'subcenters', ordin)
            classes = np.array(list(class_prototypes['train'].values()), dtype=object).squeeze()
            # Extract and filter arrays that don't contain NaNs
            # Collect valid arrays (those without NaNs)
            array_list = []
            for subdict in batch_prototypes.values():
                if isinstance(subdict, dict):  # Ensure subdict is a dictionary
                    for arr in subdict.values():
                        if isinstance(arr, np.ndarray) and not np.isnan(arr).any():  # Check for NaNs
                            array_list.append(arr)

            # Convert to a NumPy array of arrays (dtype=object to preserve separate arrays)
            batches = np.concatenate(array_list, dtype=object).squeeze()
            plot_pca(reduced_data, all_cats, all_batches, classes,
                        explained_variance, self.complete_log_path,
                        'labels', ordin)
            plot_pca(reduced_data, cluster_labels, all_batches, None,
                        explained_variance, self.complete_log_path,
                        'clusters', ordin)
            plot_pca(reduced_data, all_batches_cat, all_batches, batches,
                        explained_variance, self.complete_log_path,
                        'batches', ordin)
            if self.log_neptune:
                run[f"clusters_{ordin}"].upload(f'{self.complete_log_path}/clusters_{ordin}.png')
                run[f"batches_{ordin}"].upload(f'{self.complete_log_path}/batches_{ordin}.png')  # TODO make the batches pca in this function
                run[f"labels_{ordin}"].upload(f'{self.complete_log_path}/labels_{ordin}.png')
                if 'subcenter' in self.args.classif_loss:
                    run[f"subcenters_{ordin}"].upload(f'{self.complete_log_path}/subcenters_{ordin}.png')
            
            if self.log_mlflow:
                mlflow.log_artifact(f'{self.complete_log_path}/clusters_{ordin}.png')
                mlflow.log_artifact(f'{self.complete_log_path}/pca_batches_{ordin}.png')

        if 'arcfacewithsubcenters' not in self.args.classif_loss:
            # Identify small clusters
            cluster_sizes = np.bincount(cluster_labels)
            small_clusters = np.where(cluster_sizes < min_cluster_size)[0]  # Clusters with fewer than min_cluster_size samples
            noisy_samples = np.isin(cluster_labels, small_clusters)
        else:
            cluster_sizes = np.bincount(all_subcenters)
            small_clusters = np.where(cluster_sizes < min_cluster_size)[0]  # Clusters with fewer than min_cluster_size samples
            noisy_samples = np.isin(all_subcenters, small_clusters)
            
        # Get indices of noisy samples
        noisy_indices = np.where(noisy_samples)[0]
        
        # Log the noisy samplesminimi
        noisy_names = all_names[noisy_indices]
        noisy_labels = all_labels[noisy_indices]
        
        # Save noisy samples to a CSV file
        noisy_df = pd.DataFrame({
            'names': noisy_names,
            'labels': noisy_labels,
            'cluster': cluster_labels[noisy_indices]
        })
        noisy_df.to_csv(f'{self.complete_log_path}/noisy_samples.csv', index=False)
        
        if self.log_neptune:
            run["noisy_samples"].track_files(f'{self.complete_log_path}/noisy_samples.csv')
        
        if self.log_mlflow:
            mlflow.log_artifact(f'{self.complete_log_path}/noisy_samples.csv')
        
        if update_lists:
            # Update the lists to exclude noisy samples
            for group in ['train', 'valid', 'test']:
                mask = ~np.isin(self.all_samples['names'][group], noisy_names)
                
                # Update the lists
                lists[group]['inputs'] = self.all_samples['inputs'][group][mask]
                lists[group]['labels'] = self.all_samples['labels'][group][mask]
                lists[group]['old_labels'] = self.all_samples['old_labels'][group][mask]
                lists[group]['names'] = self.all_samples['names'][group][mask]
                lists[group]['cats'] = self.all_samples['cats'][group][mask]
                lists[group]['batches'] = self.all_samples['batches'][group][mask]
                lists[group]['encoded_values'] = self.all_samples['encoded_values'][group][mask]
                if 'arcfacewithsubcenters' in self.args.classif_loss:
                    lists[group]['subcenters'] = self.all_samples['subcenters'][group][mask]
        
        return lists

    def set_means_classes(self, list1):
        encodings = np.concatenate(list1['train']['encoded_values'])
        labels = np.concatenate(list1['train']['labels'])
        means = {}
        for label in self.unique_labels:
            inds = np.where(labels == label)[0]
            means[label] = np.mean(encodings[inds], axis=0)
        self.means = means


def set_run(run, params):
    for k in params.keys():
        run[k] = params[k]
    run['SHAP'] = 1
    return run


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--random_recs', type=int, default=0)
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--n_trials', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--n_repeats', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--groupkfold', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--path', type=str, default='./data/otite_ds_64')
    parser.add_argument('--path_original', type=str, default='./data/otite_ds_-1')
    parser.add_argument('--exp_id', type=str, default='otite')
    parser.add_argument('--bs', type=int, default=32, help='Batch size')
    parser.add_argument('--dloss', type=str, default="inverseTriplet", help='domain loss')
    parser.add_argument('--classif_loss', type=str, default='softmax_contrastive', help='triplet or cosine')
    parser.add_argument('--task', type=str, default='notNormal', help='Binary classification?')
    parser.add_argument('--is_stn', type=int, default=1, help='Transform train data?')
    parser.add_argument('--weighted_sampler', type=int, default=1, help='Weighted sampler?')
    parser.add_argument('--n_calibration', type=int, default=0, help='n_calibration?')
    parser.add_argument('--remove_noisy_samples', type=int, default=0, help='Remove noisy samples?')
    parser.add_argument('--noisy_cluster_limit', type=int, default=10, help='Noisy cluster limit?')
    parser.add_argument('--prototypes_to_use', type=str, default='no', help='Which prototypes?')
    parser.add_argument('--n_positives', type=int, default=1, help='Number of positive samples per anchor for Tuplet Loss')
    parser.add_argument('--n_negatives', type=int, default=1, help='Number of negative samples per anchor for Tuplet Loss')
    parser.add_argument('--fgsm', type=int, default=1, help='Use Fast Gradient Sign Method')
    parser.add_argument('--valid_dataset', type=str, default='Banque_Viscaino_Chili_2020', help='Validation dataset')
    parser.add_argument('--new_size', type=int, default=64, help='New size for images')
    parser.add_argument('--normalize', type=str, default='no', help='Normalize images')

    args = parser.parse_args()
    args.exp_id = f'{args.exp_id}_{args.task}'
    try:
        mlflow.create_experiment(
            args.dloss,
            # artifact_location=Path.cwd().joinpath("mlruns").as_uri(),
            # tags={"version": "v1", "priority": "P1"},
        )
    except Exception as e:
        print(f"\n\nExperiment {args.exp_id} already exists: {e}\n\n")

    train = TrainAE(args, args.path, load_tb=False, log_metrics=True, keep_models=True,
                    log_inputs=False, log_plots=True, log_tb=False, log_neptune=True,
                    log_mlflow=False, groupkfold=args.groupkfold)

    # train.train()
    # List of hyperparameters getting optimized
    parameters = [
        {"name": "lr", "type": "range", "bounds": [1e-6, 1e-3], "log_scale": True},
        {"name": "wd", "type": "range", "bounds": [1e-8, 1e-2], "log_scale": True},
        {"name": "smoothing", "type": "range", "bounds": [0., 0.2]},
        # {"name": "is_transform", "type": "choice", "values": [0, 1]},
    ]
    if args.classif_loss in ['triplet', 'softmax_contrastive'] or args.dloss in ['revTriplet', 'inverseTriplet', 'normae', 'inverse_softmax_contrastive']:
        parameters += [{"name": "dist_fct", "type": "choice", "values": ['cosine', 'euclidean']}]
        parameters += [{"name": "dmargin", "type": "range", "bounds": [0., 1.]}]
    if args.dloss in ['revTriplet', 'revDANN', 'DANN', 'inverseTriplet', 'normae', 'inverse_softmax_contrastive']:
        # gamma = 0 will ensure DANN is not learned
        parameters += [{"name": "gamma", "type": "range", "bounds": [1e-2, 1e2], "log_scale": True}]
    if args.classif_loss in ['triplet', 'softmax_contrastive']:
        parameters += [{"name": "margin", "type": "range", "bounds": [0., 10.]}]
    if args.fgsm:
        parameters += [{"name": "epsilon", "type": "range", "bounds": [1e-4, 5e-1], "log_scale": True}]
    if args.classif_loss not in ['ce', 'hinge'] or args.dloss in ['inverse_softmax_contrastive', 'inverseTriplet']:
        parameters += [{"name": "n_neighbors", "type": "range", "bounds": [1, 10], "log_scale": True}]
    # Some hyperparameters are not always required. They are set to a default value in Train.train()

    best_parameters, values, experiment, model = optimize(
        parameters=parameters,
        evaluation_function=train.train,
        objective_name='mcc',
        minimize=False,
        total_trials=args.n_trials,
      random_seed=41)

    lists, traces = get_empty_traces()
    # values, _, _, _ = get_empty_dicts()  # Pas élégant
    # Loading best model that was saved during training
    model_path = f'logs/best_models/notNormal/{train.args.model_name}/otite_ds_64/nsize{train.args.new_size}/fgsm{train.args.fgsm}/ncal{train.args.n_calibration}/{train.args.classif_loss}/{train.args.dloss}/prototypes_{train.args.prototypes_to_use}/npos{train.args.n_positives}/nneg{train.args.n_negatives}/model.pth'
    train.model.load_state_dict(
        torch.load(
            model_path,
            map_location=train.args.device
        )
    )
    # Need another model because the other can't be used to get shap values
    train.shap_model.load_state_dict(
        torch.load(
            model_path,  # Update path to match new model loading convention
            map_location=train.args.device
        )
    )
    train.model.eval()
    train.shap_model.eval()
    prototypes = {
        'combined': train.combined_prototypes,
        'class': train.class_prototypes,
        'batch': train.batch_prototypes,
    }
    loaders = get_images_loaders(data=train.data,
                                    random_recs=train.args.random_recs,
                                    weighted_sampler=train.args.weighted_sampler,
                                    is_transform=1,
                                    samples_weights=train.samples_weights,
                                    epoch=1,
                                    unique_labels=train.unique_labels,
                                    triplet_dloss=train.args.dloss, bs=train.args.bs,
                                    prototypes_to_use=train.args.prototypes_to_use,
                                    prototypes=prototypes,
                                    size=train.args.new_size,
                                    normalize=train.args.normalize,
                                    )
    with torch.no_grad():
        _, best_lists1, _ = train.loop('train', None, train.params['gamma'], loaders['train'], lists, traces)
        for group in ["train", "valid", "test"]:
            _, best_lists2, traces, knn = train.predict(group, loaders[group], lists, traces)
    best_lists = {**best_lists1, **best_lists2}
    if train.log_neptune:
        run = neptune.init_run(
            project=NEPTUNE_PROJECT_NAME,
            api_token=NEPTUNE_API_TOKEN,
        )  # your credentials
        run = set_run(run, train.best_params)
        train.log_predictions(best_lists, run, 0)
        train.save_wrong_classif_imgs(run, {'cnn': train.shap_model, 'knn': knn}, best_lists, best_lists['test']['preds'], 
                                    best_lists['test']['names'], 'test')
        
        run.stop()

