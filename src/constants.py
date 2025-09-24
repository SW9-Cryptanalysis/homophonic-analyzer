import pathlib

FEASIBILITY_THRESHOLD = 1_000_000_000 # 1 billion
FREQUENCY_TOLERANCE = 0.01 # This is 1%

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / 'data'
EXAMPLE_CIPHERS_PATH = DATA_PATH / 'example_ciphers'
FREQUENCIES_PATH = DATA_PATH / 'frequencies'