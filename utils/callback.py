import os
from abc import ABC, abstractmethod
import torch
import io
import zipfile

class BaseCallback(ABC):
    """
    Base class for callback.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, verbose: int = 0):
        super().__init__()
        # Number of time the callback was called
        self.n_calls = 0  # type: int
        # n_envs * n times env.step() was called
        self.num_timesteps = 0  # type: int
        self.verbose = verbose

    # Type hint as string to avoid circular import
    def init_callback(self, model) -> None:
        """
        Initialize the callback by saving references to the
        RL model and the training environment for convenience.
        """
        self.model = model
        # self.num_timesteps = self.model.num_timesteps
        self._init_callback()

    def _init_callback(self) -> None:
        pass

    @abstractmethod
    def _on_step(self) -> bool:
        """
        :return: If the callback returns False, training is aborted early.
        """
        return True


    def on_step(self) -> bool:
        """
        This method will be called by the model after each call to ``env.step()``.

        For child callback (of an ``EventCallback``), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        self.n_calls += 1
        self.num_timesteps = self.model.num_timesteps
        return self._on_step()




class CheckpointCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` calls
    to ``env.step()``.
    By default, it only saves model checkpoints,
    you need to pass ``save_replay_buffer=True``,
    and ``save_vecnormalize=True`` to also save replay buffer checkpoints
    and normalization statistics checkpoints.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

    :param save_freq: Save checkpoints every ``save_freq`` call of the callback.
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param save_replay_buffer: Save the model replay buffer
    :param max_files: Maximum number of files
    :param verbose: Verbosity level: 0 for no output, 2 for indicating when saving model checkpoint
    """
    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        max_files: int = 5,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer
        self.zip_manager =ZipFileManager(max_zip_files=max_files)

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
        """
        Helper to get checkpoint path for each type of checkpoint.

        :param checkpoint_type: empty for the model, "replay_buffer_"
            or "vecnormalize_" for the other checkpoints.
        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        return os.path.join(self.save_path, f"{self.name_prefix}_{checkpoint_type}{self.n_calls}_episode.{extension}")

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = self._checkpoint_path(extension="zip")
            if self.verbose >= 2:
                print(f"\nSaving model checkpoint to {model_path}")
            
            self.zip_manager.create_save_zip(model_path,self.model)

            if self.save_replay_buffer and hasattr(self.model, "buffer") and self.model.buffer is not None:
                replay_buffer_path = self._checkpoint_path("replay_buffer_", extension="pkl")
                self.model.save_replay_buffer(replay_buffer_path)  # type: ignore[attr-defined]
                if self.verbose >= 2:
                    print(f"Saving model replay buffer checkpoint to {replay_buffer_path}\n")
            else:
                print('\n')

        return True

class ZipFileManager:
    def __init__(self, max_zip_files=5):
        self.max_zip_files = max_zip_files
        self.zip_files = []

    def create_save_zip(self, zip_filename,model):
        zip_filename = zip_filename
        buffer = io.BytesIO()

        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for agent_id in range(model.num_agents):
                policy = model.trainer[agent_id].policy.q_net
                pt_data = io.BytesIO()
                torch.save(policy.state_dict(), pt_data)
                zipf.writestr(f"{agent_id}.pt", pt_data.getvalue())

        with open(zip_filename, 'wb') as zip_file:
            zip_file.write(buffer.getvalue())

        self.zip_files.append(zip_filename)

        if len(self.zip_files) > self.max_zip_files:
            oldest_zip = self.zip_files.pop(0)
            os.remove(oldest_zip)

    def get_latest_zip_files(self):
        return self.zip_files