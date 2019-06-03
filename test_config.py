test_config = [
    {
        'test_name': 'max_batch_size',
        'test_params': {
            'batch_size': list(range(5, 301, 5)),
            'utt_length': 10,
            'infers_num': 1,
            'infer_timeout': 600
        }
    },
    {
        'test_name': 'max_string_length_batch_1',
        'test_params': {
            'batch_size': 1,
            'utt_length': list(range(10, 1001, 20)),
            'infers_num': 1,
            'infer_timeout': 600
        }
    },
    {
        'test_name': 'max_infers_num',
        'test_params': {
            'batch_size': 1,
            'utt_length': 10,
            'infers_num': list(range(10, 1001, 10)),
            'infer_timeout': 600
        }
    },
    {
        'test_name': 'avg_server_response_time',
        'test_params': {
            'batch_size': 1,
            'utt_length': 10,
            'infers_num': [1, 10, 100, 200, 500, 1000],
            'infer_timeout': 600
        }
    }
]
