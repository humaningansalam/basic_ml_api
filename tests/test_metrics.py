from src.common.errors import ErrorCode


def test_metrics_endpoint(client, get_metric_value):
    response = client.get('/metrics')
    assert response.status_code == 200
    assert response.mimetype == 'text/plain'
    
    metrics_data = response.data.decode('utf-8')
    
    assert 'model_cache_usage' in metrics_data
    assert 'predictions_completed_total' in metrics_data
    assert 'cache_hits_total' in metrics_data
    assert 'cache_misses_total' in metrics_data

    assert 'ml_api_errors_total' in metrics_data
    assert get_metric_value('ml_api_errors', {'type': ErrorCode.INTERNAL_ERROR.value}) is not None
    assert 'ml_api_cpu_usage_percent' in metrics_data
    assert 'ml_api_ram_usage_mb' in metrics_data
