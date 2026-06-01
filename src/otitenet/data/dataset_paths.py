import os
import sys
from itertools import permutations


DEFAULT_DATASETS = [
    "Banque_Calaman_USA_2020_trie_CM",
    "Banque_Viscaino_Chili_2020",
    "Banque_Comert_Turquie_2020_jpg",
    "GMFUNL_jan2023",
]

DATASET_SHORT_NAMES = {
    "Banque_Calaman_USA_2020_trie_CM": "USA",
    "Banque_Viscaino_Chili_2020": "Chili",
    "Banque_Comert_Turquie_2020_jpg": "Turquie",
    "GMFUNL_jan2023": "GMFUNL",
    "inference": "inference",
}

DATASET_SUBDIR_ORDER = [
    "Banque_Calaman_USA_2020_trie_CM",
    "Banque_Comert_Turquie_2020_jpg",
    "Banque_Viscaino_Chili_2020",
    "GMFUNL_jan2023",
    "inference",
]


def normalize_dataset_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        parts = [str(x).strip() for x in value]
    else:
        parts = [x.strip() for x in str(value).replace(";", ",").split(",")]
    return list(dict.fromkeys([x for x in parts if x and x.lower() not in {"none", "nan", "null"}]))


def dataset_output_subdir(include_datasets=None, exclude_datasets=None):
    include = normalize_dataset_list(include_datasets) or DEFAULT_DATASETS
    exclude = set(normalize_dataset_list(exclude_datasets))
    selected = [d for d in include if d not in exclude]

    ordered = [dataset for dataset in DATASET_SUBDIR_ORDER if dataset in selected]
    ordered.extend(dataset for dataset in selected if dataset not in DATASET_SUBDIR_ORDER)
    short_names = [DATASET_SHORT_NAMES.get(dataset, dataset) for dataset in ordered]
    return "_".join(short_names)


def infer_output_subdir_from_split_datasets(train_datasets=None, valid_dataset=None, test_dataset=None):
    datasets = normalize_dataset_list(train_datasets)
    datasets.extend(normalize_dataset_list(valid_dataset))
    datasets.extend(normalize_dataset_list(test_dataset))
    known_datasets = [dataset for dataset in datasets if dataset in DATASET_SHORT_NAMES]
    return dataset_output_subdir(known_datasets) if known_datasets else ""


def _dataset_tokens(name):
    return frozenset(
        token.lower()
        for token in str(name or "").replace("-", "_").split("_")
        if token
    )


def resolve_processed_dataset_path(path, train_datasets=None, valid_dataset=None, test_dataset=None, for_model_lookup=False):
    """Resolve processed dataset folders even when short-name tokens are reordered.
    
    Args:
        path: The dataset path to resolve
        train_datasets: Training datasets
        valid_dataset: Validation dataset
        test_dataset: Test dataset
        for_model_lookup: If True, return the base path without dataset subdirectory for model lookup
    """
    requested = os.path.normpath(str(path or ""))
    if os.path.isfile(os.path.join(requested, "infos.csv")):
        return requested

    parent = os.path.dirname(requested)
    leaf = os.path.basename(requested)

    if not parent or not os.path.isdir(parent):
        return requested

    # If for_model_lookup is True and the leaf is a base directory name, return the base path
    # This ensures old models can be found without the dataset subdirectory
    if for_model_lookup and leaf in ['otite_ds_64', 'otite_ds_224', 'otite_ds_-1']:
        if os.path.isdir(requested):
            # Check if there are subdirectories with infos.csv
            subdirs_with_infos = []
            for entry in os.scandir(requested):
                if not entry.is_dir():
                    continue
                infos_path = os.path.join(entry.path, "infos.csv")
                if os.path.isfile(infos_path):
                    subdirs_with_infos.append(entry.name)
            
            if subdirs_with_infos:
                # Return the base path for model lookup (without dataset subdirectory)
                print(f"[DatasetPath] Using base path for model lookup: '{requested}'", file=sys.stderr)
                return requested

    # If the leaf is a base directory name (like 'otite_ds_64'), try to find the correct subdirectory
    # by using the dataset information to generate all possible combinations
    if leaf in ['otite_ds_64', 'otite_ds_224', 'otite_ds_-1'] or not leaf or leaf == '.':
        # First check if the requested path itself has subdirectories with infos.csv
        if os.path.isdir(requested):
            subdirs_with_infos = []
            for entry in os.scandir(requested):
                if not entry.is_dir():
                    continue
                infos_path = os.path.join(entry.path, "infos.csv")
                if os.path.isfile(infos_path):
                    subdirs_with_infos.append(entry.name)
            
            if subdirs_with_infos:
                # If dataset information is provided, try to match based on actual datasets
                if train_datasets or valid_dataset or test_dataset:
                    # Collect all unique datasets
                    all_datasets = set()
                    if train_datasets:
                        if isinstance(train_datasets, str):
                            # Handle 'from_infos_csv' case by reading from the first available directory
                            if train_datasets == 'from_infos_csv' and subdirs_with_infos:
                                # Try to read dataset names from infos.csv in available subdirectories
                                import pandas as pd
                                for dir_name in sorted(subdirs_with_infos):
                                    test_infos_path = os.path.join(requested, dir_name, "infos.csv")
                                    if os.path.isfile(test_infos_path):
                                        try:
                                            infos_df = pd.read_csv(test_infos_path)
                                            if 'dataset' in infos_df.columns:
                                                all_datasets.update(infos_df['dataset'].unique())
                                                break
                                        except Exception:
                                            continue
                            else:
                                all_datasets.update([d.strip() for d in train_datasets.split(',')])
                        else:
                            all_datasets.update(train_datasets)
                    if valid_dataset and valid_dataset != 'from_infos_csv':
                        all_datasets.add(valid_dataset)
                    if test_dataset and test_dataset != 'from_infos_csv':
                        all_datasets.add(test_dataset)
                    
                    # Map to short names
                    short_names = [DATASET_SHORT_NAMES.get(d, d) for d in all_datasets if d in DATASET_SHORT_NAMES]
                    
                    if short_names:
                        # Generate all permutations and check for matches
                        for perm in permutations(short_names):
                            candidate_name = "_".join(perm)
                            if candidate_name in subdirs_with_infos:
                                resolved = os.path.join(requested, candidate_name)
                                print(f"[DatasetPath] Resolved missing dataset path '{requested}' -> '{resolved}' (matched datasets)", file=sys.stderr)
                                return resolved
                
                # Fallback: prefer directories that don't have "_old" or "_inference" suffix
                preferred_dirs = [d for d in subdirs_with_infos if not d.endswith('_old') and not d.endswith('_inference')]
                if preferred_dirs:
                    # Return the first preferred directory (alphabetically sorted)
                    resolved = os.path.join(requested, sorted(preferred_dirs)[0])
                    print(f"[DatasetPath] Resolved missing dataset path '{requested}' -> '{resolved}' (fallback)", file=sys.stderr)
                    return resolved
                # Fallback to any available directory
                resolved = os.path.join(requested, sorted(subdirs_with_infos)[0])
                print(f"[DatasetPath] Resolved missing dataset path '{requested}' -> '{resolved}' (fallback)", file=sys.stderr)
                return resolved
        
        # If the requested path doesn't have subdirectories, check the parent directory
        available_dirs = []
        for entry in os.scandir(parent):
            if not entry.is_dir():
                continue
            infos_path = os.path.join(entry.path, "infos.csv")
            if os.path.isfile(infos_path):
                available_dirs.append(entry.name)
        
        if available_dirs:
            # If dataset information is provided, try to match based on actual datasets
            if train_datasets or valid_dataset or test_dataset:
                # Collect all unique datasets
                all_datasets = set()
                if train_datasets:
                    if isinstance(train_datasets, str):
                        # Handle 'from_infos_csv' case by reading from the first available directory
                        if train_datasets == 'from_infos_csv' and available_dirs:
                            # Try to read dataset names from infos.csv in available directories
                            import pandas as pd
                            for dir_name in sorted(available_dirs):
                                test_infos_path = os.path.join(parent, dir_name, "infos.csv")
                                if os.path.isfile(test_infos_path):
                                    try:
                                        infos_df = pd.read_csv(test_infos_path)
                                        if 'dataset' in infos_df.columns:
                                            all_datasets.update(infos_df['dataset'].unique())
                                            break
                                    except Exception:
                                        continue
                        else:
                            all_datasets.update([d.strip() for d in train_datasets.split(',')])
                    else:
                        all_datasets.update(train_datasets)
                if valid_dataset and valid_dataset != 'from_infos_csv':
                    all_datasets.add(valid_dataset)
                if test_dataset and test_dataset != 'from_infos_csv':
                    all_datasets.add(test_dataset)
                
                # Map to short names
                short_names = [DATASET_SHORT_NAMES.get(d, d) for d in all_datasets if d in DATASET_SHORT_NAMES]
                
                if short_names:
                    # Generate all permutations and check for matches
                    for perm in permutations(short_names):
                        candidate_name = "_".join(perm)
                        if candidate_name in available_dirs:
                            resolved = os.path.join(parent, candidate_name)
                            print(f"[DatasetPath] Resolved missing dataset path '{requested}' -> '{resolved}' (matched datasets)", file=sys.stderr)
                            return resolved
            
            # Fallback: prefer directories that don't have "_old" or "_inference" suffix
            preferred_dirs = [d for d in available_dirs if not d.endswith('_old') and not d.endswith('_inference')]
            if preferred_dirs:
                # Return the first preferred directory (alphabetically sorted)
                resolved = os.path.join(parent, sorted(preferred_dirs)[0])
                print(f"[DatasetPath] Resolved missing dataset path '{requested}' -> '{resolved}' (fallback)", file=sys.stderr)
                return resolved
            # Fallback to any available directory
            resolved = os.path.join(parent, sorted(available_dirs)[0])
            print(f"[DatasetPath] Resolved missing dataset path '{requested}' -> '{resolved}' (fallback)", file=sys.stderr)
            return resolved

    # Original token-based matching for specific directory names
    requested_tokens = _dataset_tokens(leaf)
    if not requested_tokens:
        return requested

    candidates = []
    for entry in os.scandir(parent):
        if not entry.is_dir():
            continue
        infos_path = os.path.join(entry.path, "infos.csv")
        if not os.path.isfile(infos_path):
            continue
        candidate_tokens = _dataset_tokens(entry.name)
        if not candidate_tokens:
            continue
        if candidate_tokens == requested_tokens:
            candidates.append((0, 0, entry.name, entry.path))
        elif requested_tokens.issubset(candidate_tokens):
            candidates.append((1, len(candidate_tokens - requested_tokens), entry.name, entry.path))

    if not candidates:
        return requested

    candidates.sort()
    resolved = candidates[0][3]
    print(f"[DatasetPath] Resolved missing dataset path '{requested}' -> '{resolved}'", file=sys.stderr)
    return resolved
