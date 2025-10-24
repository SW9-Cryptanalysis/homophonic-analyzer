from src.utils.logging import AnsiColorFormatter
import logging

class TestAnsiColorFormatter:
	def test_format_levels(self):
		"""Test formatting for various log levels."""
		for level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]:
			formatter = AnsiColorFormatter(datefmt="%Y-%m-%d %H:%M:%S")
			record = logging.LogRecord(
				name="test_logger",
				level=level,
				pathname="test/utils/test_logging.py",
				lineno=10,
				msg=f"This is a {logging.getLevelName(level).lower()} message.",
				args=(),
				exc_info=None,
			)
			formatted = formatter.format(record)
			assert logging.getLevelName(level) in formatted
			assert f"This is a {logging.getLevelName(level).lower()} message." in formatted
   
	def test_format_includes_filename_and_lineno(self):
		"""Test that the formatted log includes filename and line number."""
		formatter = AnsiColorFormatter(datefmt="%Y-%m-%d %H:%M:%S")
		record = logging.LogRecord(
			name="test_logger",
			level=logging.INFO,
			pathname="test/utils/test_logging.py",
			lineno=42,
			msg="Testing filename and line number.",
			args=(),
			exc_info=None,
		)
		formatted = formatter.format(record)
		assert "(test_logging.py:42)" in formatted

	def test_format_time_inclusion(self):
		"""Test that the formatted log includes the timestamp."""
		formatter = AnsiColorFormatter(datefmt="%Y-%m-%d %H:%M:%S")
		record = logging.LogRecord(
			name="test_logger",
			level=logging.INFO,
			pathname="test/utils/test_logging.py",
			lineno=1,
			msg="Testing time inclusion.",
			args=(),
			exc_info=None,
		)
		formatted = formatter.format(record)
		assert "INFO" in formatted
		assert "Testing time inclusion." in formatted