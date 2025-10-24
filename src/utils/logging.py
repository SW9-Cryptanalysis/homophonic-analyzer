import logging

class AnsiColorFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        # Define color codes
        self.colors = {
			"grey": "\033[90m",
			"blue": "\033[94m",
			"green": "\033[92m",
			"yellow": "\033[93m",
			"red": "\033[91m",
			"cyan": "\033[96m",
			"reset": "\033[0m",
		}
        self.decorations = {
			"bold": "\033[1m",
			"underline": "\033[4m",
		}
        self.color_codes = {
            "time": "\033[90m",       # Grey
            "level": {
                10: self.colors['blue'],   # Blue
                20: self.colors['green'],    # Green
                30: self.colors['yellow'], # Yellow
                40: self.colors['red'],      # Red
                50: self.colors['red'] + self.decorations['bold'], # Bold Red
            },
            "name": self.colors['cyan'],       # Cyan
            "message": self.colors['reset'],     # Reset (default)
            "reset": self.colors['reset'],
        }

    def format(self, record: logging.LogRecord):
        asctime = self.formatTime(record, self.datefmt)
        levelname = record.levelname
        levelno = record.levelno
        name = record.name
        message = record.getMessage()

        time_str = f"{self.color_codes['time']}{asctime}{self.color_codes['reset']}"
        level_str = f"{self.color_codes['level'].get(levelno, self.color_codes['reset'])}{levelname}{self.color_codes['reset']}"
        name_str = f"{self.color_codes['name']}{name}{self.color_codes['reset']}"
        message_str = f"{self.color_codes['message']}{message}{self.color_codes['reset']}"
        filename_str = f"{self.colors['grey']}({record.filename}:{record.lineno}){self.color_codes['reset']}"

        return f"{time_str} {self.colors['grey']}| {level_str} | {name_str} | {message_str} {filename_str}"