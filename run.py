import shutil
import asyncio
import logging
import bisect
import random
import statistics
from pprint import pprint
from collections import defaultdict
from datetime import datetime
from typing import Tuple
from pathlib import Path

from deeppavlov.core.common.paths import _root_path as dp_root_dir

from agent import get_infer_agent
from test_config import tests_pipeline


MAX_FAULTS_RATE = 0


results_dir = Path(__file__).resolve().parent / 'results'
shutil.copy(Path(__file__).resolve().parent / 'agent_config.yaml', dp_root_dir / 'deeppavlov/core/agent_v2/config.yaml')
agent_inferer = get_infer_agent()


class UtteranceGenerator:
    def __init__(self) -> None:
        dialogs_path = Path(__file__).resolve().parent / 'dialogs.txt'
        with dialogs_path.open('r') as f:
            dialogs_str: str = f.read()
            self._examples = defaultdict(set)

            tokens = dialogs_str.replace('\n\n', ' ').replace('\n', ' ').split(' ')
            replicas = [replica for dialog in dialogs_str.split('\n\n') for replica in dialog.split('\n')]
            dialogs = [dialog.replace('\n', ' ') for dialog in dialogs_str.split('\n\n')]

            for token in tokens:
                self._examples[len(token)].add(token)

            for repl in replicas:
                self._examples[len(repl)].add(repl)

            for dial in dialogs:
                self._examples[len(dial)].add(dial)

            for key, value in self._examples.items():
                self._examples[key] = list(self._examples[key])

            self._indexes = list(self._examples.keys())
            self._indexes.sort()

    def __call__(self, symbols_num: int) -> str:
        i = bisect.bisect_right(self._indexes, symbols_num)
        substring_symbols_num = self._indexes[i - 1] if i > 0 else 1
        substring = random.choice(self._examples.get(substring_symbols_num, ['!']))
        symbols_num_delta = symbols_num - substring_symbols_num

        if symbols_num_delta < 1:
            result = substring
        elif symbols_num_delta == 1:
            result = f'{substring}{self.__call__(symbols_num_delta)}'
        else:
            result = f'{substring} {self.__call__(symbols_num_delta - 1)}'

        return result


utterance_generator = UtteranceGenerator()


async def infer_agent(utterances: list, ids: list) -> list:
    return agent_inferer(utterances, ids)


async def infer(utterances: list, ids: list, loop: asyncio.AbstractEventLoop,
                infer_timeout: int) -> Tuple[bool, float]:

    time_begin: float = loop.time()

    try:
        await asyncio.wait_for(infer_agent(utterances, ids), float(infer_timeout))
    except asyncio.TimeoutError:
        return False, -1.0

    time_end: float = loop.time()

    return True, time_end - time_begin


def run_single_test(batch_size: int, utt_length: int = 5, infers_num: int = 1,
                    infer_timeout: int = 600) -> Tuple[bool, float, float]:

    loop = asyncio.get_event_loop()
    results = []

    for _ in range(infers_num):
        utt_batch = [utterance_generator(utt_length) for _ in range(batch_size)]
        ids_batch = [str(utt_id) for utt_id in range(batch_size)]
        results.append(loop.run_until_complete(infer(utt_batch, ids_batch, loop, infer_timeout)))

    passed, await_times = list(zip(*results))
    faults_num = passed.count(False)
    await_times = [t for t in await_times if t >= 0]

    await_avg = statistics.mean(await_times) if await_times else 0
    await_var = statistics.variance(await_times) if len(await_times) >= 2 else 0

    return faults_num, await_avg, await_var


def run_tests():
    print('Starting stress test sequence')

    results_dir.mkdir(exist_ok=True)
    log_file_path = results_dir / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}_test.log"

    logger = logging.getLogger('dp_agent_stress_test')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    for test in tests_pipeline:
        print(f'Starting {test["test_name"]}')
        logger.info(f'Starting {test["test_name"]}')

        batch_size = test['test_params']['batch_size']
        utt_length = test['test_params']['utt_length']
        infers_num = test['test_params']['infers_num']
        infer_timeout = test['test_params']['infer_timeout']

        batch_size = list(range(batch_size, batch_size + 1, 1)) if isinstance(batch_size, int) else list(batch_size)
        utt_length = list(range(utt_length, utt_length + 1, 1)) if isinstance(utt_length, int) else list(utt_length)
        infers_num = list(range(infers_num, infers_num + 1, 1)) if isinstance(infers_num, int) else list(infers_num)

        test_grid = [(bs, ul, inum) for bs in batch_size for ul in utt_length for inum in infers_num]

        for test_attempt in test_grid:
            result = run_single_test(test_attempt[0], test_attempt[1], test_attempt[2], infer_timeout)
            report = '{}:: batch_size: {}, utt_length: {}, infers_num: {}, FAULTS: {}, AVG_TIME: {}, DISP {}'.format(
                test['test_name'],
                test_attempt[0],
                test_attempt[1],
                test_attempt[2],
                result[0],
                result[1],
                result[2])

            logger.info(report)

            if result[0] / test_attempt[2] > MAX_FAULTS_RATE:
                logger.info(f'Interrupted {test["test_name"]} with {result[0]} error attempts of {test_attempt[2]}')
                break

        print(f'Finished{test["test_name"]}')

    print('Finished')


if __name__ == '__main__':
    run_tests()

