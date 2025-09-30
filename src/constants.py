import pathlib

FEASIBILITY_THRESHOLD = 100_000_000_000_000 # 1 Trillion
FREQUENCY_TOLERANCE = 0.0005 # This is 0.05%

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / 'data'
EXAMPLE_CIPHERS_PATH = DATA_PATH / 'example_ciphers'
FREQUENCIES_PATH = DATA_PATH / 'frequencies'