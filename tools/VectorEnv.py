import warnings
from typing import Any, Callable, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import packaging

from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.env.utils import ENV_TYPE, gym_new_venv_step_type
from tianshou.env.worker import (
    DummyEnvWorker,
    EnvWorker,
    RayEnvWorker,
    SubprocEnvWorker,
)
from tianshou.env import BaseVectorEnv


class CustomDummyVectorEnv(BaseVectorEnv):
    """Dummy vectorized environment wrapper, implemented in for-loop.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for other APIs' usage.
    """

    def __init__(self, env_fns: List[Callable[[], ENV_TYPE]], **kwargs: Any) -> None:
        super().__init__(env_fns, DummyEnvWorker, **kwargs)

    def set_selects(self, selects: Any, id: Optional[Union[int, List[int], np.ndarray]] = None,) -> None:
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)
        for j in id:
            self.workers[j].get_env_attr('set_selects')(selects)
