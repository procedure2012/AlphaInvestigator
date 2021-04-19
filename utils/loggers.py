from utils.misc import setup_logger

LOGGER_DISABLED = {
    'main': False,
    'memory': False,
    'tourney': False,
    'mcts': False,
    'model': False
}

logger_mcts = setup_logger('logger_mcts', './run/logs/logger_mcts.log')
logger_mcts.disabled = LOGGER_DISABLED['mcts']
logger_main = setup_logger('logger_main', './run/logs/logger_main.log')
logger_main.disabled = LOGGER_DISABLED['main']
logger_tourney = setup_logger('logger_tourney', './run/logs/logger_tourney.log')
logger_tourney.disabled = LOGGER_DISABLED['tourney']
logger_memory = setup_logger('logger_memory', './run/logs/logger_memory.log')
logger_memory.disabled = LOGGER_DISABLED['memory']
logger_model = setup_logger('logger_model', './run/logs/logger_model.log')
logger_model.disabled = LOGGER_DISABLED['model']
