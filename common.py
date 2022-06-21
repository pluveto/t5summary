import logging
import os


def load_dotenv(dotenv_file: str = '.env'):
    """Load environment variables from a .env file.
    Args:
        dotenv_file (str): filename of the .env file.
    """
    # if dotenv_file is list, load first existent file
    if isinstance(dotenv_file, list):
        for f in dotenv_file:
            if os.path.exists(f):
                dotenv_file = f
                break

    if not os.path.exists(dotenv_file):
        raise FileNotFoundError(f'{dotenv_file} not found.')

    with open(dotenv_file, encoding='utf-8') as f:
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


def init_logger(logger: logging.Logger, debug=False) -> logging.Logger:
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
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    return logger