NEPTUNE_API_TOKEN = "YOUR-API-KEY"
NEPTUNE_PROJECT_NAME = "YOUR-PROJECT-NAME"
NEPTUNE_MODEL_NAME = "YOUR-MODEL-NAME"

import os
import uuid
import shutil
import neptune
import mlflow
import warnings
import matplotlib
import pickle
import random
import json
import torch
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression
from torch import nn
from torchvision.transforms import v2 as transforms
from sklearn import metrics
from RandConv.lib.networks import RandConvModule

matplotlib.use('Agg')
CUDA_VISIBLE_DEVICES = ""

import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from ax.service.managed_loop import optimize
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import precision_score as PPV
from sklearn.metrics import recall_score
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from otitenet.dataset import Dataset1
from otitenet.data_getters import GetData, get_distance_fct, get_images_loaders
from otitenet.loggings import LogConfusionMatrix, save_roc_curve, log_pca
from otitenet.cnn import Net
from otitenet.utils import get_optimizer, to_categorical, get_empty_dicts, get_empty_traces, \
    log_traces, get_best_values, add_to_logger, add_to_neptune, \
    add_to_mlflow
from datetime import datetime
import torchvision
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.decomposition import PCA
# from otitenet.SVHModule import SpatialTransformer

warnings.filterwarnings("ignore")

random.seed(1)
torch.manual_seed(1)
np.random.seed(1)

def get_random_module(net, device):
    return RandConvModule(net,
                          in_channels=3,
                          out_channels=3,
                          kernel_size=[3, 3],
                          mixing=1,
                          identity_prob=0.0,
                          rand_bias=1,
                          distribution='kaiming_normal',
                          data_mean=[0.5, 0.5, 0.5],
                          data_std=[0.5, 0.5, 0.5],
                          clamp_output=1,
                          ).to(device)

class TrainAE:

    def __init__(self, args, path, fix_thres=-1, load_tb=False, log_metrics=False, keep_models=True, log_inputs=True,
                 log_plots=False, log_tb=False, log_neptune=False, log_mlflow=True, groupkfold=True):
        """

        Args:
            args: contains multiple arguments passed in the command line
            log_path: Path where the tensorboard logs are saved
            path: Path to the data (in .csv format)
            fix_thres: If 1 > fix_thres >= 0 then the threshold is fixed to that value.
                       any other value means the threshold won't be fixed and will be
                       learned as an hyperparameter
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
        self.keep_models = keep_models
        self.fix_thres = fix_thres
        self.load_tb = load_tb
        self.groupkfold = groupkfold
        self.foldername = None

        self.verbose = 1

        self.n_cats = None
        self.data = None
        self.unique_labels = None
        self.unique_batches = None
        self.triplet_loss = None
        self.knn = None
        self.gaps = None
        self.celoss = nn.CrossEntropyLoss()
        

    def train_knn(self, lists, traces, params):
        if self.args.classifier == 'knn':
            self.knn = KNN(n_neighbors=params['n_neighbors'], metric=params['knn_metric'])
        elif self.args.classifier == 'logreg':
            self.knn = LogisticRegression()
        train_encs = np.concatenate(lists['train']['encoded_values'])
        train_cats = np.concatenate(lists['train']['cats'])
        train_labels = np.concatenate(lists['train']['labels'])
        train_names = np.concatenate(lists['train']['names'])
        self.knn.fit(train_encs, train_cats)
        preds = self.knn.predict(train_encs)
        proba = self.knn.predict_proba(train_encs)
        for i, l in enumerate(train_cats):
            self.samples_weights['train'][train_names[i]] = max(0.01, (1 - float(proba[i][l])))
        # self.gaps =    [max(0.01, (1 - float(proba[i][l]))) for i, l in enumerate(train_cats)]
        # self.gaps = {x: gap for x, gap in zip(train_names, self.gaps)}
        lists['train']['preds'] += [preds]
        traces['train']['acc'] += [np.mean([0 if pred != dom else 1 for pred, dom in
                                            zip(preds, train_cats)])]
        traces['train']['mcc'] += [np.round(
            MCC(train_cats,
                preds), 3)
        ]
        traces['train']['ppv'] += [np.round(
            PPV(train_cats,
                preds), 3)
        ]
        traces['train']['npv'] += [np.round(
            PPV(
                np.logical_not(train_cats), 
                np.logical_not(preds)
                ), 3)
        ]
        traces['train']['tpr'] += [np.round(
            recall_score(train_cats,
                            preds), 3)
        ]
        traces['train']['tnr'] += [
            np.round(recall_score(
                np.logical_not(train_cats), 
                np.logical_not(preds)
                ), 3)
        ]
        return lists, traces

    def define_weights(self):
        self.class_weights = {
            label: self.args.negative_class_importance if label == 'Normal' else 1
            for label in self.unique_labels
        }
        self.unique_unique_labels = list(self.class_weights.keys())

        self.samples_weights = {
            group: {name: self.class_weights[label] for name, label in
                    zip(self.data['names'][group],
                        self.data['labels'][group])} if group == 'train' else {
                1 for name, label in
                zip(self.data['names'][group], self.data['labels'][group])} for group in
            ['train', 'valid', 'test']}
        self.n_cats = len(self.unique_labels)
        self.scaler = None
        

    def make_balanced_weights(self):
        self.class_weights = {
            label: 1 / (len(np.where(label == self.data['labels']['train'])[0]) /
                        self.data['labels']['train'].shape[0])
            for label in self.unique_labels if
            label in self.data['labels']['train']}
        self.unique_unique_labels = list(self.class_weights.keys())

        self.samples_weights = {
            group: {name: self.class_weights[label] for name, label in
                    zip(self.data['names'][group],
                        self.data['labels'][group])} if group == 'train' else {
                1 for name, label in
                zip(self.data['names'][group], self.data['labels'][group])} for group in
            ['train', 'valid', 'test']}
        self.n_cats = len(self.unique_labels)
        self.scaler = None
        self.celoss = nn.CrossEntropyLoss()

    def make_weights(self, is_transform, triplet_dloss, random_recs=False):
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(-180, 180)),
            transforms.RandomApply(nn.ModuleList([
                transforms.RandomAffine(
                    degrees=10,
                    translate=(.1, .1),
                    # scale=(0.9, 1.1),
                    shear=(.01, .01),
                    interpolation=transforms.InterpolationMode.BILINEAR
                )
            ]), p=0.5),
            torchvision.transforms.Normalize(0.5, 0.5),
        ])

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # transforms.Grayscale(),
            # transforms.Lambda(lambda x: x.broadcast_to(3, x.shape[1], x.shape[2]))
            # transforms.Resize(224),
            torchvision.transforms.Normalize(0.5, 0.5),
        ])

        if is_transform:
            transform_train = transform_train
        else:
            transform_train = transform

        train_set = Dataset1(data=self.data['inputs']['train'],
                            unique_batches=self.unique_batches,
                            unique_labels=self.unique_labels,
                            names=self.data['names']['train'],
                            labels=self.data['labels']['train'],
                            old_labels=self.data['old_labels']['train'],
                            batches=self.data['batches']['train'],
                            sets=None,
                            transform=transform_train, crop_size=-1, random_recs=random_recs,
                            triplet_dloss=triplet_dloss
                            )
                            
        samples_weights = [self.samples_weights['train'][name] for name in self.data['names']['train']]
        train_loader = DataLoader(train_set,
                            sampler=WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True),
                            batch_size=self.args.bs,
                            num_workers = 0,
                            # prefetch_factor = 8,
                            pin_memory=True,
                            drop_last=False,
                            # persistent_workers=True
                            )
        return train_loader

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

        # Assigns the hyperparameters not getting optimized
        if self.args.dloss not in ['revTriplet', 'revDANN', 'DANN',
                                    'inverseTriplet', 'normae'] or 'gamma' not in params:
            # gamma = 0 will ensure DANN is not learned
            params['gamma'] = 0
        if self.args.classifier != 'knn':
            params['knn_metric'] = 0
            params['n_neighbors'] = 0
        params['dropout'] = 0

        optimizer_type = 'adam'
        dist_fct = get_distance_fct(params['dist_fct'])
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=get_distance_fct(params['dist_fct']), margin=params['margin'], swap=True)  # NOT SURE IF SWAP IS RIGHT
        # self.triplet_loss = nn.TripletMarginLoss(margin=margin, swap=False)
        # self.log_path is where tensorboard logs are saved
        self.foldername = str(uuid.uuid4())

        if self.log_mlflow:
            mlflow.set_experiment(
                self.args.exp_id,
            )
            try:
                mlflow.start_run()
            except:
                mlflow.end_run()
                mlflow.start_run()

            mlflow.log_param('margin', params['margin'])
            # mlflow.log_param('smooth', params['smooth'])
            mlflow.log_param('wd', params['wd'])
            mlflow.log_param('lr', params['lr'])
            mlflow.log_param('gamma', params['gamma'])
            mlflow.log_param('dropout', params['dropout'])
            mlflow.log_param('optimizer_type', optimizer_type)
            mlflow.log_param('dloss', self.args.dloss)
            mlflow.log_param('foldername', self.foldername)
            mlflow.log_param('is_transform', params['is_transform'])
            mlflow.log_param('classif_loss', self.args.classif_loss)
            mlflow.log_param('task', self.args.task)
            mlflow.log_param('is_stn', self.args.is_stn)
            mlflow.log_param('n_calibration', self.args.n_calibration)
            mlflow.log_param('importance_weights', self.args.importance_weights)
            mlflow.log_param('new_shape', self.args.new_shape)
            mlflow.log_param('n_pixels', self.args.n_pixels)
            mlflow.log_param('model_name', self.args.model_name)
            mlflow.log_param('weighted_sampler', self.args.weighted_sampler)
            mlflow.log_param('negative_class_importance', self.args.negative_class_importance)
            mlflow.log_param('classifier', self.args.classifier)
            mlflow.log_param('groupkfold', self.args.groupkfold)
            mlflow.log_param('distance_fct', params['dist_fct'])
            mlflow.log_param('knn_metric', params['knn_metric'])
            mlflow.log_param('n_neighbors', params['n_neighbors'])
            mlflow.log_param('path', self.args.path)

        self.complete_log_path = f'logs/{self.args.task}/{self.foldername}'
        loggers = {'cm_logger': LogConfusionMatrix(self.complete_log_path)}
        print(f'See results using: tensorboard --logdir={self.complete_log_path} --port=6006')

        hparams_filepath = self.complete_log_path + '/hp'
        os.makedirs(hparams_filepath, exist_ok=True)
        seed = 0
        epoch = 0
        best_closses = []
        best_mccs = []
        self.best_loss = np.inf
        self.best_closs = np.inf
        self.best_acc = 0
        data_getter = GetData(self.path, self.args)
        event_acc = EventAccumulator(hparams_filepath)
        event_acc.Reload()

        if self.log_neptune:
            # Create a Neptune run object
            run = neptune.init_run(
                project=NEPTUNE_PROJECT_NAME,
                api_token=NEPTUNE_API_TOKEN,
            )  # your credentials
            model = neptune.init_model_version(
                model=NEPTUNE_MODEL_NAME,
                project=NEPTUNE_PROJECT_NAME,
                api_token=NEPTUNE_API_TOKEN,
                # your credentials
            )
            run["dataset"].track_files(f"{self.path}/{self.args.csv_file}")
            run["metadata"].track_files(
                f"{self.path}/subjects_experiment_ATN_verified_diagnosis.csv"
            )
            # Track metadata and hyperparameters by assigning them to the run
            model["inputs_type"] = run["inputs_type"] = self.args.csv_file.split(".csv")[0]
            model["best_unique"] = run["best_unique"] = self.args.best_features_file.split(".tsv")[0]
            model["parameters"] = run["parameters"] = params
            model["csv_file"] = run["csv_file"] = self.args.csv_file
            model["model_name"] = run["model_name"] = self.args.resnet50
            model["groupkfold"] = run["groupkfold"] = self.args.groupkfold
            model["foldername"] = run["foldername"] = self.foldername
        else:
            model = None
            run = None

        for h in range(self.args.n_repeats):
            print("Repeat:", h)
            self.best_mcc = -1
            self.best_mcc_mean = -1
            if not self.args.groupkfold:
                self.data, self.unique_labels, self.unique_batches = data_getter.split('valid', self.args.groupkfold, seed=1+h)
            else:
                self.data, self.unique_labels, self.unique_batches = data_getter.get_variables()
            # self.unique_batches = np.unique(self.data['batches']['all'])

            if self.args.n_calibration > 0:
                # take a few samples from train to be used for calibration. The indices
                for group in ['valid', 'test']:
                    indices = {label: np.argwhere(self.data['labels'][group] == label).flatten() for label in self.unique_labels}
                    calib_inds = np.concatenate([np.random.choice(indices[label], int(self.args.n_calibration/len(self.unique_labels)), replace=False) for label in self.unique_labels])

                    self.data['inputs']['calibration'] = self.data['inputs'][group][calib_inds]
                    self.data['labels']['calibration'] = self.data['labels'][group][calib_inds]
                    self.data['old_labels']['calibration'] = self.data['old_labels'][group][calib_inds]
                    self.data['names']['calibration'] = self.data['names'][group][calib_inds]
                    self.data['cats']['calibration'] = self.data['cats'][group][calib_inds]
                    self.data['batches']['calibration'] = self.data['batches'][group][calib_inds]

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

            print("Train Batches:", np.unique(self.data['batches']['train']))
            print("Valid Batches:", np.unique(self.data['batches']['valid']))
            print("Test Batches:", np.unique(self.data['batches']['test']))

            self.n_batches = len(
                set(np.concatenate((
                self.data['batches']['train'],
                self.data['batches']['valid'],
                self.data['batches']['test']
                )))
            )
            if not self.args.importance_weights:
                if self.args.weighted_sampler:           
                    self.make_balanced_weights()
                elif self.args.negative_class_importance > 0:
                    self.define_weights()
                else:
                    self.samples_weights = {'train': {name: 1 for name in self.data['names']['train']}}
            else:
                self.samples_weights = {'train': {name: 1 for name in self.data['names']['train']}}
            loaders = get_images_loaders(data=self.data,
                                         random_recs=self.args.random_recs,
                                         weighted_sampler=self.args.weighted_sampler,
                                         is_transform=params['is_transform'],
                                         samples_weights=self.samples_weights['train'].values(),
                                         unique_batches=self.unique_batches,
                                         unique_labels=self.unique_labels,
                                         triplet_dloss=self.args.dloss, bs=self.args.bs)
            # Initialize model
            model = Net(self.args.device, self.n_cats, self.n_batches, model_name=self.args.model_name, is_stn=self.args.is_stn)
            # self.rand_module = get_random_module(model, self.args.device)
            loggers['logger_cm'] = SummaryWriter(f'{self.complete_log_path}/cm')
            loggers['logger'] = SummaryWriter(f'{self.complete_log_path}/traces')
            # sceloss = nn.CrossEntropyLoss(label_smoothing=smooth)
            optimizer_model = get_optimizer(model, params['lr'], params['wd'], optimizer_type)

            values, best_values, _, best_traces = get_empty_dicts()
            early_stop_counter = 0
            best_vals = values

            for epoch in range(0, self.args.n_epochs):
                if early_stop_counter >= self.args.early_stop:
                    if self.verbose > 0:
                        print('EARLY STOPPING.', epoch)
                    break
                lists, traces = get_empty_traces()
                model.train()
                if args.dloss == 'no':
                    train_groups = ['train']
                else:
                    train_groups = ['all', 'train']
                for group in train_groups:
                    _, lists, traces = self.loop(group, optimizer_model, model,  params, loaders[group], lists, traces)
                lists, traces = self.train_knn(lists, traces, params)
                model.eval()
                for group in ["valid", "test"]:
                    _, _, _ = self.predict(group, model, loaders[group], lists, traces)
                if self.args.importance_weights:
                    loaders['train'] = self.make_weights(params['is_transform'], self.args.dloss, self.args.random_recs)
                # traces = self.get_mccs(lists, traces)
                # loggers['cm_logger'].add(lists)
                values = log_traces(traces, values)
                if self.log_tb:
                    try:
                        add_to_logger(values, loggers['logger'], epoch)
                    except:
                        print("Problem with add_to_logger!")
                if self.log_neptune:
                    add_to_neptune(values, run, epoch)
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
                    torch.save(model.state_dict(), f'{self.complete_log_path}/model.pth')
                    filehandler = open(f'{self.complete_log_path}/knn.pickle',"wb")
                    pickle.dump(self.knn, filehandler)
                    filehandler.close()
                    best_values = get_best_values(values.copy())
                    best_vals = values.copy()
                    best_vals['rec_loss'] = self.best_loss
                    # best_vals['dom_loss'] = self.best_dom_loss
                    # best_vals['dom_acc'] = self.best_dom_acc
                    early_stop_counter = 0
                if self.best_mcc_mean < np.mean(values['valid']['mcc']):
                    early_stop_counter = 0
                    self.best_mcc_mean = np.mean(values['valid']['mcc'])
                    print(self.best_mcc_mean, self.best_mcc)
                else:
                    m = np.mean(values['valid']['mcc'])
                    print('missed', m, self.best_mcc_mean)

                if values['valid']['acc'][-1] > self.best_acc:
                    print(f"Best Classification Acc Epoch {epoch}, "
                            f"Acc: {values['test']['acc'][-1]}"
                            f"Mcc: {values['test']['mcc'][-1]}"
                            f"Classification "
                            f" valid loss: {values['valid']['closs'][-1]},"
                            f" test loss: {values['test']['closs'][-1]}, dloss: {values['all']['dloss'][-1]}")

                    self.best_acc = values['valid']['acc'][-1]
                    early_stop_counter = 0

                if values['valid']['closs'][-1] < self.best_closs:
                    print(f"Best Classification Loss Epoch {epoch}, "
                            f"Acc: {values['test']['acc'][-1]} "
                            f"Mcc: {values['test']['mcc'][-1]} "
                            f"Classification "
                            f"valid loss: {values['valid']['closs'][-1]}, "
                            f"test loss: {values['test']['closs'][-1]}, dloss: {values['all']['dloss'][-1]}")
                    self.best_closs = values['valid']['closs'][-1]
                    early_stop_counter = 0
                else:
                    # if epoch > self.warmup:
                    early_stop_counter += 1

            best_mccs += [self.best_mcc]

        best_lists, traces = get_empty_traces()
        # Loading best model that was saved during training
        model.load_state_dict(torch.load(f'{self.complete_log_path}/model.pth'))
        # Need another model because the other cant be use to get shap values
        # shap_model.load_state_dict(torch.load(f'{self.complete_log_path}/model.pth'))
        # model.load_state_dict(sd)
        model.eval()
        # shap_model.eval()
        with torch.no_grad():
            filehandler = open(f'{self.complete_log_path}/knn.pickle',"rb")
            self.knn = pickle.load(filehandler)
            filehandler.close()
            for group in ['train', "valid", "test"]:
                closs, best_lists, traces = self.predict(group, model, loaders[group], best_lists, traces)
        if self.log_neptune:
            model["model"].upload(f'{self.complete_log_path}/model.pth')
            model["validation/closs"].log(self.best_closs)
        if self.log_mlflow:
            log_pca(lists, self.complete_log_path, logger=None, mlops='mlflow')        


        # Create a dataframe of best_lists valid and test. Have preds, classes, names and gaps between pred and class
        for group in ['valid', 'test']:
            df = pd.DataFrame(np.concatenate((np.concatenate(best_lists[group]['preds']),
                                                np.concatenate(best_lists[group]['labels']).reshape(-1, 1),
                                                np.concatenate(best_lists[group]['domains']).reshape(-1, 1),
                                                np.concatenate(best_lists[group]['names']).reshape(-1, 1),
                                                ), 1))
            df.columns = [f'probs_{l}' for l in self.unique_labels] + ['labels', 'domains', 'names']
            preds = df.iloc[:, :2].values
            preds = preds.astype(np.float64).argmax(1).astype(np.int32)
            df['correct'] = [1 if label == df.columns[pred].split('_')[1] else 0 for label, pred in zip(df['labels'], preds)]
            df['gaps'] = [1 - float(df[f'probs_{l}'].iloc[i]) for i, l in enumerate(df['labels'])]
            # reorder dataframe with gaps column
            df = df.sort_values(by=['gaps'], ascending=False)
            
            # save the valid dataframe
            df.to_csv(f'{self.complete_log_path}/{group}.csv')
            
        best_closses += [self.best_closs]
        shap_model = None
        # traces = self.get_mccs(lists, traces)
        loggers['cm_logger'].add(best_lists)
        values = log_traces(traces, values)
        self.log_rep(best_lists, run, 0)
        del model # , shap_model

        # Logging every model is taking too much resources and it makes it quite complicated to get information when
        # Too many runs have been made. This will make the notebook so much easier to work with
        if np.mean(best_mccs) > self.best_best_mcc:
            try:
                if os.path.exists(
                        f'logs/best_models/model_classifier/{self.args.model_name}'):
                    shutil.rmtree(
                        f'logs/best_models/model_classifier/{self.args.model_name}',
                        ignore_errors=True)
                shutil.copytree(f'{self.complete_log_path}',
                                f'logs/best_models/model_classifier/{self.args.model_name}')
                # print("File copied successfully.")

            # If source and destination are same
            except shutil.SameFileError:
                # print("Source and destination represents the same file.")
                pass
            self.best_best_mcc = np.mean(best_mccs)

        # Logs confusion matrices in the background. Also runs RandomForestClassifier on encoded and reconstructed
        # representations. This should be shorter than the actual calculation of the model above in the function,
        # otherwise the number of threads will keep increasing.
        # daemon = Thread(target=self.logging, daemon=True, name='Monitor', args=[run, cm_logger])
        # daemon.start()

        loggers['cm_logger'].plot(None, epoch, self.unique_labels, 'mlflow')
        if self.log_mlflow:
            mlflow.log_param('finished', 1)
        # self.logging(run, loggers['cm_logger'])

        if not self.keep_models:
            # shutil.rmtree(f'{self.complete_log_path}/traces', ignore_errors=True)
            # shutil.rmtree(f'{self.complete_log_path}/cm', ignore_errors=True)
            # shutil.rmtree(f'{self.complete_log_path}/hp', ignore_errors=True)
            shutil.rmtree(f'{self.complete_log_path}', ignore_errors=True)
        print('Duration: {}'.format(datetime.now() - start_time))
        best_closs = np.mean(best_closses)
        if best_closs < self.best_closs_final:
            self.best_closs_final = best_closs
            print("Best closs!")

        # It should not be necessary. To remove once certain the "Too many files open" error is no longer a problem
        # plt.close('all')

        return np.mean(best_mccs)

    def log_rep(self, best_lists, run, h):
        # if run is None:
        #     return None
        # best_traces = self.get_mccs(best_lists, traces)

        self.log_predictions(best_lists, run, h)

    def log_predictions(self, best_lists, run, step):
        cats, labels, preds, scores, names = [{'train': [], 'valid': [], 'test': []} for _ in range(5)]
        for group in ['train', 'valid', 'test']:
            cats[group] = np.concatenate(best_lists[group]['cats'])
            labels[group] = np.concatenate(best_lists[group]['labels'])
            scores[group] = torch.softmax(torch.Tensor(np.concatenate(best_lists[group]['preds'])), 1)
            preds[group] = scores[group].argmax(1)
            names[group] = np.concatenate(best_lists[group]['names'])
            pd.DataFrame(np.concatenate((labels[group].reshape(-1, 1), scores[group],
                                         np.array([self.unique_labels[x] for x in preds[group]]).reshape(-1, 1),
                                         names[group].reshape(-1, 1)), 1)).to_csv(
                f'{self.complete_log_path}/{group}_predictions.csv')
            if self.log_neptune:
                run[f"{group}_predictions"].track_files(f'{self.complete_log_path}/{group}_predictions.csv')
                run[f'{group}_AUC'] = metrics.roc_auc_score(y_true=cats[group], y_score=scores[group],
                                                            multi_class='ovr')
            if self.log_mlflow:
                mlflow.log_metric(f'{group}_AUC',
                            metrics.roc_auc_score(y_true=to_categorical(cats[group], len(self.unique_labels)), y_score=scores[group], multi_class='ovr'),
                            step=step)

                # TODO ADD OTHER FINAL METRICS HERE
                mlflow.log_metric(f'{group}_acc',
                            metrics.accuracy_score(cats[group], scores[group].argmax(1)),
                            step=step
                )
                mlflow.log_metric(f'{group}_mcc',
                            MCC(cats[group], scores[group].argmax(1)),
                            step=step
                )
                mlflow.log_metric(f'{group}_ppv',
                                  PPV(cats[group], scores[group].argmax(1)),
                                      step=step
                )
                mlflow.log_metric(f'{group}_npv',
                                  PPV(
                                      np.logical_not(cats[group]), 
                                      np.logical_not(scores[group].argmax(1))),
                                      step=step
                                      )
                mlflow.log_metric(f'{group}_tpr',
                    recall_score(cats[group], scores[group].argmax(1)),
                    step=step
                )
                mlflow.log_metric(f'{group}_tnr',
                    recall_score(
                        np.logical_not(cats[group]), 
                        np.logical_not(scores[group].argmax(1))),
                        step=step
                )


            save_roc_curve(
                scores[group],
                to_categorical(cats[group], len(self.unique_labels)),
                self.unique_labels,
                f'{self.complete_log_path}/{group}_roc_curve.png',
                binary=False,
                acc=metrics.accuracy_score(cats[group], scores[group].argmax(1)),
                mlops='mlflow',
                epoch=step,
                logger=None
            )
    
    def loop(self, group, optimizer_model, model, params, loader, lists, traces):
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
            if model.training:
                optimizer_model.zero_grad()
            data, names, labels, _, old_labels, domain, to_rec, not_to_rec, pos_batch_sample, neg_batch_sample = batch
            data = data.to(self.args.device).float()
            to_rec = to_rec.to(self.args.device).float()
            # self.rand_module.randomize()
            # data = self.rand_module(data)
            domains = to_categorical(domain.long(), self.n_batches).squeeze().float().to(self.args.device)
            enc, dpreds = model(data)

            if self.args.classif_loss == 'cosine':
                classif_loss = 1 - torch.nn.functional.cosine_similarity(enc, model(to_rec)).mean()
            elif self.args.classif_loss == 'triplet':
                not_to_rec = not_to_rec.to(self.args.device).float()
                pos_enc, _ = model(to_rec)
                neg_enc, _ = model(not_to_rec)
                classif_loss = self.triplet_loss(enc, model(to_rec)[0], model(not_to_rec)[0])

            if self.args.dloss == 'DANN':
                dloss = self.celoss(dpreds, domains)

            elif self.args.dloss == 'inverseTriplet':
                pos_batch_sample, neg_batch_sample = pos_batch_sample.to(self.args.device).float(), neg_batch_sample.to(self.args.device).float()
                pos_enc, _ = model(pos_batch_sample)
                neg_enc, _ = model(neg_batch_sample)
                dloss = self.triplet_loss(enc, neg_enc, pos_enc)
                # domain = domain.argmax(1)
            else:
                dloss = torch.Tensor([0]).to(self.args.device)[0]

            lists[group]['domains'] += [np.array([self.unique_batches[np.argmax(d)] for d in domains.detach().cpu().numpy()])]
            lists[group]['classes'] += [labels.detach().cpu().numpy()]
            # if group == 'train':
            #     lists[group]['dpreds'] += [dpreds.detach().cpu().numpy()]
            lists[group]['encoded_values'] += [enc.view(enc.shape[0], -1).detach().cpu().numpy()]
            lists[group]['names'] += [names]
            lists[group]['inputs'] += [data.detach().cpu().numpy().reshape(data.shape[0], self.args.new_shape, self.args.new_shape, 3)]
            lists[group]['labels'] += [np.array(
                [self.unique_labels[x] for x in labels.detach().cpu().numpy()])]
            lists[group]['old_labels'] += [np.array(old_labels)]
            traces[group]['dom_acc'] += [np.mean([0 if pred != dom else 1 for pred, dom in
                                                zip(dpreds.detach().cpu().numpy().argmax(1),
                                                    domains.argmax(1).detach().cpu().numpy())])]
            traces[group]['closs'] += [classif_loss.item()]
            traces[group]['dloss'] += [dloss.item()]
            lists[group]['cats'] += [labels.detach().cpu().numpy()]
            if model.training:
                if group in ['train']:
                    # total_loss = classif_loss + gamma * dloss
                    classif_loss.backward()
                    # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                elif group in ['all']:
                    if params['gamma'] > 0:
                        dloss = params['gamma'] * dloss
                        dloss.backward()
                optimizer_model.step()
            # pbar.update(1)

        return classif_loss, lists, traces

    def predict(self, group, model, loader, lists, traces):
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
        # import nearest_neighbors from sklearn
        classif_loss = None

        # plot_decision_boundary(KNeighborsClassifier, train_encs, train_cats)
        # plt.savefig(f'{self.complete_log_path}/decision_boundary.png')
        # with tqdm(total=len(loader), position=0, leave=True) as pbar:
        for i, batch in enumerate(loader):
            # if group in ['train']:
            data, names, labels, _, old_labels, domain, to_rec, not_to_rec, pos_batch_sample, neg_batch_sample = batch
            data = data.to(self.args.device).float()
            # self.rand_module.randomize()
            # data = self.rand_module(data)
            to_rec = to_rec.to(self.args.device).float()
            domains = to_categorical(domain.long(), self.n_batches).squeeze().float().to(self.args.device)
            enc, dpreds = model(data)
            preds = self.knn.predict(enc.view(enc.shape[0], -1).detach().cpu().numpy())
            proba = self.knn.predict_proba(enc.view(enc.shape[0], -1).detach().cpu().numpy())

            if self.args.dloss == 'DANN':
                dloss = self.celoss(dpreds, domains)

            elif self.args.dloss == 'inverseTriplet':
                pos_batch_sample, neg_batch_sample = neg_batch_sample.to(
                    self.args.device).float(), pos_batch_sample.to(self.args.device).float()
                pos_enc, _ = model(pos_batch_sample)
                neg_enc, _ = model(neg_batch_sample)
                dloss = self.triplet_loss(enc, pos_enc, neg_enc)
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
        traces[group]['mcc'] += [np.round(
            MCC(np.concatenate(lists[group]['cats']), np.concatenate(lists[group]['preds']).argmax(1)), 3)
        ]
        traces[group]['ppv'] += [np.round(
            PPV(np.concatenate(lists[group]['cats']), np.concatenate(lists[group]['preds']).argmax(1)), 3)
        ]
        traces[group]['npv'] += [np.round(
            PPV(
                np.logical_not(np.concatenate(lists[group]['cats'])), 
                np.logical_not(np.concatenate(lists[group]['preds']).argmax(1))
                ), 3)
        ]
        traces[group]['tpr'] += [np.round(
            recall_score(np.concatenate(lists[group]['cats']), np.concatenate(lists[group]['preds']).argmax(1)), 3)
        ]
        traces[group]['tnr'] += [
            np.round(recall_score(
                np.logical_not(np.concatenate(lists[group]['cats'])), 
                np.logical_not(np.concatenate(lists[group]['preds']).argmax(1))
                ), 3)
        ]
            # pbar.update(1)

        if group == 'test':
            # plot the PCA of the encoded values
            pca = PCA(n_components=2)
            train_encs = np.concatenate(lists['train']['encoded_values'])
            train_cats = np.concatenate(lists['train']['cats'])
            valid_encs = np.concatenate(lists['valid']['encoded_values'])
            valid_cats = np.concatenate(lists['valid']['cats'])
            test_encs = np.concatenate(lists['test']['encoded_values'])
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
            # The axes should have the percentage of variation explained
            plt.xlabel(f'PCA1: {np.round(pca.explained_variance_ratio_[0]*100, 2)}%')
            plt.ylabel(f'PCA2: {np.round(pca.explained_variance_ratio_[1]*100, 2)}%')
            plt.savefig(f'pca.png')

        return None, lists, traces

    @staticmethod
    def get_mccs(lists, traces):
        """
        Function that gets the Matthews Correlation Coefficients. MCC is a statistical tool for model evaluation.
        It is a balanced measure which can be used even if the classes are of very different sizes.
        Args:
            lists:
            traces:

        Returns:
            traces: Same list as in the inputs arguments, except in now contains the MCC values
        """
        for group in ['train', 'valid', 'test']:
            try:
                preds, classes = np.concatenate(lists[group]['preds']).argmax(1), np.concatenate(
                    lists[group]['classes'])
                traces[group]['mcc'] = MCC(preds, classes)
            except:
                pass
            

        return traces


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--random_recs', type=int, default=0)
    parser.add_argument('--early_stop', type=int, default=100)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--n_repeats', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--groupkfold', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--path', type=str, default='./data/otite_ds')
    parser.add_argument('--exp_id', type=str, default='otite')
    parser.add_argument('--bs', type=int, default=64, help='Batch size')
    parser.add_argument('--dloss', type=str, default="inverseTriplet", help='domain loss')
    parser.add_argument('--classif_loss', type=str, default='triplet', help='triplet or cosine')
    parser.add_argument('--task', type=str, default='notNormal', help='Binary classification?')
    parser.add_argument('--is_stn', type=int, default=0, help='Transform train data?')
    parser.add_argument('--weighted_sampler', type=int, default=0, help='Weighted sampler?')
    parser.add_argument('--n_pixels', type=int, default=64, help='N pixels?')
    parser.add_argument('--new_shape', type=int, default=224, help='New shape?')
    parser.add_argument('--n_calibration', type=int, default=0, help='n_calibration?')
    parser.add_argument('--importance_weights', type=int, default=0, help='importance_weights?')
    parser.add_argument('--negative_class_importance', type=float, default=0.1, help='negative_class_importance?')
    parser.add_argument('--classifier', type=str, default='osir', help='')
    # parser.add_argument('--siamese', type=int, default=1, help='')

    args = parser.parse_args()
    args.exp_id = f'{args.exp_id}_{args.task}'
    args.path = f'./{args.path}_{args.n_pixels}'
    try:
        mlflow.create_experiment(
            args.dloss,
            # artifact_location=Path.cwd().joinpath("mlruns").as_uri(),
            # tags={"version": "v1", "priority": "P1"},
        )
    except:
        print(f"\n\nExperiment {args.exp_id} already exists\n\n")

    train = TrainAE(args, args.path, fix_thres=-1, load_tb=False, log_metrics=True, keep_models=True,
                    log_inputs=False, log_plots=True, log_tb=False, log_neptune=False,
                    log_mlflow=True, groupkfold=args.groupkfold)

    # train.train()
    # List of hyperparameters getting optimized
    parameters = [
        {"name": "lr", "type": "range", "bounds": [1e-5, 1e-3], "log_scale": True},
        {"name": "wd", "type": "range", "bounds": [1e-8, 1e-2], "log_scale": True},
        # {"name": "smoothing", "type": "range", "bounds": [0., 0.2]},
        # {"name": "knn_metric", "type": "choice", "values": ['cosine', 'minkowski']},
        {"name": "dist_fct", "type": "choice", "values": ['euclidean']},
        {"name": "is_transform", "type": "choice", "values": [0, 1]},
        {"name": "margin", "type": "range", "bounds": [1., 10.]},
        # {"name": "n_neighbors", "type": "range", "bounds": [1, 20]},
    ]
    if args.dloss in ['revTriplet', 'revDANN', 'DANN', 'inverseTriplet', 'normae']:
        # gamma = 0 will ensure DANN is not learned
        parameters += [{"name": "gamma", "type": "range", "bounds": [1e-4, 1e-2], "log_scale": True}]

    if args.classifier == 'knn':
        parameters += [{"name": "n_neighbors", "type": "range", "bounds": [1, 20]}]
        parameters += [{"name": "knn_metric", "type": "choice", "values": ['cosine', 'minkowski']}]
    # Some hyperparameters are not always required. They are set to a default value in Train.train()

    best_parameters, values, experiment, model = optimize(
        parameters=parameters,
        evaluation_function=train.train,
        objective_name='closs',
        minimize=True,
        total_trials=args.n_trials,
        random_seed=2,

    )

    fig = plt.figure()
    # render(plot_contour(model=model, param_x="learning_rate", param_y="weight_decay", metric_name='Loss'))
    # fig.savefig('test.jpg')
    print('Best Loss:', values[0]['loss'])
    print('Best Parameters:')
    print(json.dumps(best_parameters, indent=4))
