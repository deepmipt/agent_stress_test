test_config = {
    'max_batch_size': {
        'test_name': 'max_batch_size',
        'test_params': {
            'batch_size': list(range(10, 201, 10)),
            'utt_length': 20,
            'infers_num': 1,
            'infer_timeout': 600
        }
    },
    'max_string_length_batch_1': {
        'test_name': 'max_string_length_batch_1',
        'test_params': {
            'batch_size': 1,
            'utt_length': list(range(50, 2001, 50)),
            'infers_num': 1,
            'infer_timeout': 600
        }
    }
}

tests_pipeline = [
    test_config['max_batch_size'],
    test_config['max_string_length_batch_1']
]
