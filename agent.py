import logging
from datetime import datetime
from typing import Tuple, List, Callable


def get_infer_agent() -> Callable:
    from deeppavlov.core.agent_v2.agent import Agent
    from deeppavlov.core.agent_v2.state_manager import StateManager
    from deeppavlov.core.agent_v2.skill_manager import SkillManager
    from deeppavlov.core.agent_v2.rest_caller import RestCaller
    from deeppavlov.core.agent_v2.preprocessor import IndependentPreprocessor
    from deeppavlov.core.agent_v2.response_selector import ConfidenceResponseSelector
    from deeppavlov.core.agent_v2.skill_selector import ChitchatQASelector
    from deeppavlov.core.agent_v2.config import MAX_WORKERS, ANNOTATORS, SKILL_SELECTORS

    logging.getLogger('requests.packages.urllib3.connectionpool').setLevel(logging.WARNING)

    state_manager = StateManager()

    anno_names, anno_urls = zip(*[(annotator['name'], annotator['url']) for annotator in ANNOTATORS])
    preprocessor = IndependentPreprocessor(
        rest_caller=RestCaller(max_workers=MAX_WORKERS, names=anno_names, urls=anno_urls))

    skill_caller = RestCaller(max_workers=MAX_WORKERS)
    response_selector = ConfidenceResponseSelector()
    ss_names, ss_urls = zip(*[(annotator['name'], annotator['url']) for annotator in SKILL_SELECTORS])
    skill_selector = ChitchatQASelector(rest_caller=RestCaller(max_workers=MAX_WORKERS, names=ss_names, urls=ss_urls))
    skill_manager = SkillManager(skill_selector=skill_selector, response_selector=response_selector,
                                 skill_caller=skill_caller)

    agent = Agent(state_manager, preprocessor, skill_manager)

    def infer(utterances: List[str], dialog_ids: List[str]) -> List[str]:
        u_d_types = [None] * len(utterances)
        date_times = [datetime.utcnow()] * len(utterances)
        locations = [None] * len(utterances)
        ch_types = ['telegram'] * len(utterances)

        answers = agent(utterances=utterances, user_telegram_ids=dialog_ids, user_device_types=u_d_types,
                        date_times=date_times, locations=locations, channel_types=ch_types)
        return answers

    return infer
