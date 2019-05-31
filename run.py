import shutil
from pathlib import Path

from deeppavlov.core.common.paths import _root_path as dp_root_dir

from agent import get_infer_agent


shutil.copy(Path(__file__).resolve().parent / 'agent_config.yaml', dp_root_dir / 'deeppavlov/core/agent_v2/config.yaml')

agent_inferer = get_infer_agent()

print(agent_inferer(['Привет', 'Пока'], ['1', '2']))
