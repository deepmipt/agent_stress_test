from pathlib import Path
from typing import Callable, Union
from abc import ABCMeta, abstractmethod


class TestBase(metaclass=ABCMeta):
    _agent: Callable

    def __init__(self, agent: Callable) -> None:
        self._agent = agent

    def test(self, *args, **kwargs) -> Union[bool, int, str]:
        pass

    @abstractmethod
    def _test(self):
        pass