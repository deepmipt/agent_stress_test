test_config = {
    'max_batch_size': {
        'test_name': 'max_batch_size',
        'test_params': {
            'batch_size': list(range(50, 301, 5)),
            'utt_length': 10,
            'infers_num': 1,
            'infer_timeout': 600
        }
    },
    'max_string_length_batch_1': {
        'test_name': 'max_string_length_batch_1',
        'test_params': {
            'batch_size': 1,
            'utt_length': list(range(100, 1001, 50)),
            'infers_num': 1,
            'infer_timeout': 600
        }
    },
    'max_infers_num': {
        'test_name': 'max_infers_num',
        'test_params': {
            'batch_size': 1,
            'utt_length': 10,
            'infers_num': list(range(50, 1001, 50)),
            'infer_timeout': 600
        }
    },
    'avg_server_response_time': {
        'test_name': 'avg_server_response_time',
        'test_params': {
            'batch_size': 1,
            'utt_length': 10,
            'infers_num': [1, 10, 100, 200, 500, 1000],
            'infer_timeout': 600
        }
    }
}

tests_pipeline = [
    test_config['max_batch_size'],
    test_config['max_string_length_batch_1'],
    test_config['max_infers_num'],
    test_config['avg_server_response_time']
]
