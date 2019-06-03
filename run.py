import shutil
import asyncio
from typing import Tuple
from pathlib import Path

from deeppavlov.core.common.paths import _root_path as dp_root_dir

from agent import get_infer_agent


shutil.copy(Path(__file__).resolve().parent / 'agent_config.yaml',
            dp_root_dir / 'deeppavlov/core/agent_v2/config.yaml')
agent_inferer = get_infer_agent()


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
                    infer_timeout: int = 600) -> Tuple[bool, float]:

    loop = asyncio.get_event_loop()

    utterances = [''.join(['t'] * utt_length)] * batch_size
    ids = [str(utt_id) for utt_id in range(batch_size)]
    infers = []

    for _ in range(infers_num):
        infers.append(infer(utterances, ids, loop, infer_timeout))

    results = loop.run_until_complete(asyncio.gather(*infers))
    passed, await_times = list(zip(*results))
    faults_num = passed.count(False)
    avg_await = -1.0 if faults_num else round(sum(await_times) / len(await_times), 4)

    return faults_num, avg_await
