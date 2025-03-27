import torch
import pickle
import neptune
from ..train.train_triplet_new import TrainAE, set_run, NEPTUNE_PROJECT_NAME, NEPTUNE_API_TOKEN
from ..utils.utils import get_empty_traces
from ..data.data_getters import get_images_loaders
from ..data.data_getters import GetData
from ..models.cnn import Net, Net_shap

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="resnet18")
    parser.add_argument('--fgsm', type=int, default=0)
    parser.add_argument('--n_calibration', type=int, default=0)
    parser.add_argument('--loss', type=str, default='triplet')
    parser.add_argument('--dloss', type=str, default='no')
    parser.add_argument('--prototypes_to_use', type=str, default='class')
    parser.add_argument('--n_positives', type=int, default=1)
    parser.add_argument('--n_negatives', type=int, default=1)
    parser.add_argument('--new_size', type=int, default=224)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--task', type=str, default='notNormal')
    parser.add_argument('--classif_loss', type=str, default='softmax_contrastive', help='triplet or cosine')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--path', type=str, default='./data/otite_ds_64')
    parser.add_argument('--path_original', type=str, default='./data/otite_ds_-1')
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--groupkfold', type=int, default=1)
    parser.add_argument('--random_recs', type=int, default=0)
    parser.add_argument('--valid_dataset', type=str, default='Banque_Viscaino_Chili_2020', help='Validation dataset')

    args = parser.parse_args()
    data_getter = GetData(args.path, args.valid_dataset, args)
    data, unique_labels, unique_batches = data_getter.get_variables()
    n_cats = len(unique_labels)
    n_batches = len(unique_batches)
    train = TrainAE(args, args.path, load_tb=False, log_metrics=True, keep_models=True,
                    log_inputs=False, log_plots=True, log_tb=False, log_neptune=True,
                    log_mlflow=False, groupkfold=args.groupkfold)
    train.n_batches = n_batches
    train.n_cats = n_cats
    train.unique_batches = unique_batches
    train.unique_labels = unique_labels
    train.epoch = 1
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

    model = Net(args.device, n_cats, n_batches,
                        model_name=args.model_name, is_stn=0,
                        n_subcenters=n_batches)
    train.model = model
    model.load_state_dict(torch.load(model_path))
    model.to(args.device)
    model.eval()
    shap_model = Net_shap(args.device, n_cats, n_batches,
                        model_name=args.model_name, is_stn=0,
                        n_subcenters=n_batches)
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
    if train.log_neptune:
        run = neptune.init_run(
            project=NEPTUNE_PROJECT_NAME,
            api_token=NEPTUNE_API_TOKEN,
        )  # your credentials
        run = set_run(run, train.best_params)
        train.log_predictions(best_lists, run, 0)
        train.save_wrong_classif_imgs(run, {'cnn': shap_model, 'knn': knn},
                                      best_lists, best_lists['test']['preds'], 
                                      best_lists['test']['names'], 'test')
        
        run.stop()
