# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Pipeline for ActionPiece."""

import logging
import os
from typing import Any,Dict,Union

import accelerate as accelerate_lib
from genrec import utils
from genrec import utils_log
from genrec.utils import get_file_name
from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
import torch
from torch.utils import data


DataLoader = data.DataLoader


class Pipeline:
  """Pipeline for ActionPiece.

  This class orchestrates the training and evaluation of an ActionPiece model,
  including:
    - Loading and configuring the dataset.
    - Initializing the tokenizer.
    - Setting up the model.
    - Creating the trainer.
    - Preparing data loaders.
    - Running the training and evaluation loop.

  Attributes:
      config: A dictionary containing the configuration parameters.
      project_dir: The directory for the accelerator.
      accelerator: An instance of the accelerate.Accelerator class.
      logger: An instance of the logging.Logger class.
      raw_dataset: The raw dataset instance.
      split_datasets: A dictionary containing the split datasets (train, val,
        test).
      tokenizer: The tokenizer instance.
      tokenized_datasets: A dictionary containing the tokenized datasets.
      model: The model instance.
      trainer: The trainer instance.
  """

  def __init__(
      self,
      # model_name: str | AbstractModel,
      # dataset_name: str | AbstractDataset,
      # tokenizer: AbstractTokenizer | None = None,
      model_name:  Union[str, AbstractModel],
      dataset_name: Union[str, AbstractDataset],
      tokenizer: AbstractTokenizer | None = None,
      trainer=None,
      #config_dict: dict[str, Any] | None = None,
      config_dict: Union[Dict[str, Any], None] = None,
      config_file: str = None,
  ):
    self.config = utils.get_config(
        model_name=model_name,
        dataset_name=dataset_name,
        config_file=config_file,
        config_dict=config_dict,
    )
    # Automatically set devices and ddp
    self.config['device'], self.config['use_ddp'] = utils.init_device()

    # Accelerator
    self.config['accelerator'] = accelerate_lib.Accelerator(
        log_with='tensorboard', project_dir=self.project_dir
    )

    # Seed and Logger
    utils.init_seed(self.config['rand_seed'], self.config['reproducibility'])
    utils.init_logger(self.config)
    self.logger = logging.getLogger()
    self.log(f'Device: {self.config["device"]}')

    # Dataset
    self.raw_dataset = utils.get_dataset(dataset_name)(self.config)
    self.log(self.raw_dataset)
    self.split_datasets = self.raw_dataset.split()

    # Tokenizer
    if tokenizer is not None:
      self.tokenizer = tokenizer(self.config, self.raw_dataset)
    else:
      assert isinstance(
          model_name, str
      ), 'Tokenizer must be provided if model_name is not a string.'
      self.tokenizer = utils.get_tokenizer(model_name)(
          self.config, self.raw_dataset
      )
    self.tokenized_datasets = self.tokenizer.tokenize(self.split_datasets)

    # Model
    with self.config['accelerator'].main_process_first():
      self.model = utils.get_model(model_name)(
          self.config, self.raw_dataset, self.tokenizer
      )
    self.log(self.model)
    self.log(self.model.n_parameters)

    # Trainer
    if trainer is not None:
      self.trainer = trainer
    else:
      self.trainer = utils.get_trainer(model_name)(
          self.config, self.model, self.tokenizer
      )
  @property
  def project_dir(self) -> str:
    """Returns the directory for the accelerator."""
    # return os.path.join(
    #     self.config['tensorboard_log_dir'],
    #     self.config['dataset'],
    #     self.config['model'],
    # )
    base = self.config.get('tensorboard_log_dir', 'runs')
    # 若被错误转换成列表，这里做兜底
    if isinstance(base, (list, tuple)):
        base = base[0] if base else 'runs'
    dataset = self.config.get('dataset', 'dataset')
    model = self.config.get('model', 'model')
    return os.path.join(str(base), str(dataset), str(model))

  @property
  def accelerator(self) -> accelerate_lib.Accelerator:
    """Returns the accelerator instance."""
    return self.config['accelerator']

  def run(self):
      """Runs the training and evaluation pipeline.

      This method sets up data loaders, trains the model, evaluates it on the test
      set, and logs the results.
      """

      def get_dataloader(split, batch_size, shuffle):
          # return DataLoader(
          #     self.tokenized_datasets[split],
          #     batch_size=batch_size,
          #     shuffle=shuffle,
          #     collate_fn=self.tokenizer.collate_fn[split],
          # )
          num_workers = int(self.config.get('num_workers', 0))
          pin_memory = bool(self.config.get('pin_memory', False)) and torch.cuda.is_available()
          persistent_workers = bool(self.config.get('persistent_workers', False)) and num_workers > 0
          prefetch_factor = self.config.get('prefetch_factor', None)

          kwargs = dict(
              batch_size=batch_size,
              shuffle=shuffle,
              collate_fn=self.tokenizer.collate_fn[split],
              num_workers=num_workers,
              pin_memory=pin_memory,
              persistent_workers=persistent_workers,
              drop_last=False,
          )
          if prefetch_factor is not None and num_workers > 0:
              kwargs['prefetch_factor'] = int(prefetch_factor)
          return DataLoader(self.tokenized_datasets[split], **kwargs)

      # DataLoader
      train_dataloader = get_dataloader(
          'train', self.config['train_batch_size'], True
      )
      val_dataloader = get_dataloader(
          'val', self.config['eval_batch_size'], False
      )
      if self.config['n_inference_ensemble'] == -1:
          #eval_batch_size = self.config['eval_batch_size']
          test_batch_size = self.config['eval_batch_size']
      else:
          test_batch_size = max(
              self.config['eval_batch_size'] // self.config['n_inference_ensemble'],
              1,
          )
      # val_dataloader = get_dataloader('val', eval_batch_size, False)
      # test_dataloader = get_dataloader('test', eval_batch_size, False)
      test_dataloader = get_dataloader('test', test_batch_size, False)

      self.trainer.fit(train_dataloader, val_dataloader)

      self.accelerator.wait_for_everyone()
      self.model = self.accelerator.unwrap_model(self.model)

      self.model.load_state_dict(torch.load(self.trainer.saved_model_ckpt))
      self.model, test_dataloader = self.accelerator.prepare(
          self.model, test_dataloader
      )
      if self.accelerator.is_main_process:
          self.log(
              f'Loaded best model checkpoint from {self.trainer.saved_model_ckpt}'
          )
      test_results = self.trainer.evaluate(test_dataloader)

      if self.accelerator.is_main_process:
          for key in test_results:
              self.trainer.accelerator.log({f'Test_Metric/{key}': test_results[key]})
      self.log(f'Test Results: {test_results}')

      self.trainer.end()

  def log(self, message, level='info'):
      return utils_log.log(
          message, self.config['accelerator'], self.logger, level=level
      )