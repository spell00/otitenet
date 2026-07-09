"""Test learned embedding classification head persistence to database."""

import pytest
from unittest.mock import Mock, patch, MagicMock


def test_ensure_learned_embedding_heads_table():
    """Test that the learned_embedding_heads table is created."""
    from otitenet.app.database import ensure_learned_embedding_heads_table
    
    # Mock database connection and cursor
    mock_conn = Mock()
    mock_cursor = Mock()
    
    # Mock table doesn't exist initially
    mock_cursor.fetchone.return_value = [0]
    
    ensure_learned_embedding_heads_table(mock_conn, mock_cursor)
    
    # Verify CREATE TABLE was called
    assert mock_cursor.execute.called
    create_call_args = mock_cursor.execute.call_args[0][0]
    assert "CREATE TABLE learned_embedding_heads" in create_call_args
    assert "config VARCHAR(255)" in create_call_args
    assert "head_family VARCHAR(64)" in create_call_args
    assert "valid_mcc FLOAT" in create_call_args
    
    # Verify commit was called
    assert mock_conn.commit.called


def test_save_learned_embedding_heads():
    """Test saving learned embedding heads to database."""
    from otitenet.app.database import save_learned_embedding_heads
    
    mock_cursor = Mock()
    mock_conn = Mock()
    
    heads = [
        {
            'config': '5',
            'family': 'knn',
            'n_aug': 0,
            'valid_mcc': 0.75,
            'test_mcc': 0.73,
            'valid_auc': 0.85,
            'test_auc': 0.83,
            'train_datasets': 'train_a,train_b',
            'valid_dataset': 'valid_a',
            'test_dataset': 'test_a',
        },
        {
            'config': 'baseline_logreg',
            'family': 'logreg',
            'n_aug': 0,
            'valid_mcc': 0.70,
            'test_mcc': 0.68,
        }
    ]
    
    save_learned_embedding_heads(mock_cursor, mock_conn, model_id=123, log_path='logs/best_models/test/resnet18/model', heads=heads)
    
    # Verify INSERT was called for each head
    assert mock_cursor.execute.call_count >= 2
    assert mock_conn.commit.called


def test_load_learned_embedding_heads():
    """Test loading learned embedding heads from database."""
    from otitenet.app.database import load_learned_embedding_heads
    
    mock_cursor = Mock()
    
    # Mock database response
    mock_cursor.fetchall.return_value = [
        ('5', 'knn', 0, 2, 0.72, 0.75, 0.73, None, 0.80, 0.85, 0.83, None, 0.90, 'train_a,train_b', 'valid_a', 'test_a', 'train|valid|test', '', 123, 'logs/best_models/test/resnet18/model'),
    ]
    
    heads = load_learned_embedding_heads(mock_cursor, model_id=123)
    
    # Verify heads were loaded
    assert len(heads) == 1
    assert heads[0]['config'] == '5'
    assert heads[0]['family'] == 'knn'
    assert heads[0]['valid_mcc'] == 0.75
    assert heads[0]['model_id'] == 123


def test_infer_head_family_from_config():
    """Test head family inference from config string."""
    from otitenet.app.database import _infer_head_family_from_config
    
    assert _infer_head_family_from_config('5') == 'knn'
    assert _infer_head_family_from_config('10') == 'knn'
    assert _infer_head_family_from_config('protot_gmm_2') == 'gmm'
    assert _infer_head_family_from_config('protot_mean_1') == 'mean'
    assert _infer_head_family_from_config('baseline_logreg') == 'logreg'
    assert _infer_head_family_from_config('baseline_random_forest') == 'random_forest'
    assert _infer_head_family_from_config('unknown') == 'unknown'


def test_convert_db_heads_to_cache():
    """Test converting database heads back to cache format."""
    from otitenet.app.utils import _convert_db_heads_to_cache
    
    heads = [
        {
            'config': '5',
            'family': 'knn',
            'n_aug': 0,
            'valid_mcc': 0.75,
            'test_mcc': 0.73,
            'valid_auc': 0.85,
            'test_auc': 0.83,
            'train_datasets': 'train_a',
            'valid_dataset': 'valid_a',
            'test_dataset': 'test_a',
        }
    ]
    
    mock_args = Mock()
    cache = _convert_db_heads_to_cache(heads, mock_args)
    
    # Verify cache structure
    assert 0 in cache
    assert 'knn' in cache[0]
    assert 'mcc_per_k' in cache[0]['knn']
    assert len(cache[0]['knn']['mcc_per_k']) == 1
    assert cache[0]['knn']['mcc_per_k'][0]['k'] == 5
    assert cache[0]['knn']['mcc_per_k'][0]['valid_mcc'] == 0.75
    assert cache[0]['best_config'] == '5'
    assert cache[0]['best_k'] == '5'


def test_convert_db_heads_to_cache_best_overall_can_be_gmm():
    """DB fallback must not collapse the best classifier head to KNN."""
    from otitenet.app.utils import _convert_db_heads_to_cache

    heads = [
        {
            'config': '1',
            'family': 'knn',
            'n_aug': 2,
            'valid_mcc': 0.55,
            'test_mcc': 0.52,
        },
        {
            'config': 'protot_gmm_2',
            'family': 'gmm',
            'n_aug': 2,
            'valid_mcc': 0.81,
            'test_mcc': 0.78,
            'valid_auc': 0.91,
            'test_auc': 0.88,
        },
    ]

    cache = _convert_db_heads_to_cache(heads, Mock())

    assert cache[2]['best_config'] == 'protot_gmm_2'
    assert cache[2]['best_k'] == 'protot_gmm_2'
    assert cache[2]['best_mcc'] == 0.81
    assert cache[2]['best_head_metrics']['valid_mcc'] == 0.81
    assert cache[2]['prototypes']['gmm']['best_n_components'] == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
