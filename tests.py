import asyncio
from typing import Callable, Union, Tuple
from abc import ABCMeta, abstractmethod


class BaseTester(metaclass=ABCMeta):
    _test_name: str
    _result_description: str
    _agent: Callable
    _timeout_secs: int

    def __init__(self,
                 test_name: str,
                 result_description: str,
                 agent: Callable,
                 timeout_secs: int = 600) -> None:

        self._test_name = test_name
        self._result_description = result_description
        self._agent = agent
        self._timeout_secs = timeout_secs

    def __call__(self, repeats: int = 1, **kwargs) -> Tuple[bool, dict]:
        try:
            loop = asyncio.get_event_loop()
            test_result = loop.run_until_complete(asyncio.wait_for(self._test(**kwargs), float(self._timeout_secs)))
            test_succeeded = True
        except asyncio.TimeoutError:
            test_result = False
            test_succeeded = False

        result = {
            'test_name': self._test_name,
            'result_description': self._result_description,
            'test_params': {**kwargs},
            'test_result': test_result
        }

        return test_succeeded, result

    @abstractmethod
    async def _test(self, **kwargs) -> Union[bool, int]:
        pass


class MaxBatchSizeTester(BaseTester):
    def __init__(self, test_name: str, agent: Callable, timeout_secs: int = 600) -> None:
        super(MaxBatchSizeTester, self).__init__(test_name, 'test_passed', agent, timeout_secs)

    async def _test(self, batch_size: int, utterance: str = 'test') -> Union[bool, int]:
        utterances_ids = [str(u_id) for u_id in range(batch_size)]
        utterances = [utterance] * batch_size
        results = self._agent(utterances, utterances_ids)
        print(results)

        return True
