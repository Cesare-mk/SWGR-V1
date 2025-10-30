# -*- coding: utf-8 -*-
"""
@Time ： 2025/9/4 9:33
@Auth ： mm
@File ：utils_log.py
@IDE ：PyCharm
"""
import logging
def log(message, accelerator, logger, level='info'):
  """Logs a message to the logger.

  Args:
      message (str): The message to log.
      accelerator (Accelerator): The accelerator object.
      logger (logging.Logger): The logger object.
      level (str): The log level ('info', 'error', 'warning', 'debug').
  """
  if accelerator.is_main_process:
    # Map level names to their numeric values for compatibility with older Python versions
    level_mapping = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    try:
      # level = logging.getLevelNamesMapping()[level.upper()]
        level_num = level_mapping[level.upper()]
    except KeyError as exc:
      raise ValueError(f'Invalid log level: {level}') from exc

    logger.log(level_num, message)

