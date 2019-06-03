test_config = [
    {
        'test_name': 'max_batch_size',
        'test_params': {
            'batch_size': list(range(5, 21, 5)),
            'utt_length': 5,
            'infers_num': 1,
            'infer_timeout': 600
        }
    }
]
