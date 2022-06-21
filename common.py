import logging
import os


def load_dot_env(dot_env_file: str = '.env'):
    """Load environment variables from a .env file.
    Args:
        dot_env_file (str): filename of the .env file.
    """
    # if dot_env_file is list, load first existent file
    if isinstance(dot_env_file, list):
        for f in dot_env_file:
            if os.path.exists(f):
                dot_env_file = f
                break

    if not os.path.exists(dot_env_file):
        raise FileNotFoundError(f'{dot_env_file} not found.')

    with open(dot_env_file, encoding='utf-8') as f:
        for l in f.readlines():
            # strip comments
            l = l.split('#')[0]
            # skip empty lines
            if not l.strip():
                continue
            # strip `export`
            if l.startswith('export '):
                l = l[len('export '):].strip()
            # split key and value
            k, v = l.strip().split('=')
            os.environ[k] = v
    # debug
    if bool(os.environ.get('DEBUG', None)):
        print('Loaded environment variables:')
        for k, v in os.environ.items():
            print('  {}: {}'.format(k, v))


def init_logger(logger: logging.Logger) -> logging.Logger:
    """Initialize a logger.

    Args:
        logger (logging.Logger): logger

    Returns:
        logging.Logger: logger
    """
    GRAY_COLOR = '\033[1;30m'
    RESET_COLOR = '\033[0m'
    formatter = logging.Formatter(GRAY_COLOR +
                                  '[%(levelname)s] %(filename)s:%(lineno)d ' +
                                  RESET_COLOR + '%(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger