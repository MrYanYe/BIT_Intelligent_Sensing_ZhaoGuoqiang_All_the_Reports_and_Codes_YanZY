# logger.py
import sys
import os
from pathlib import Path
from datetime import datetime
import threading

class _TeeStream:
    """A stream that writes to both original stream and a file."""
    def __init__(self, orig_stream, file_obj):
        self.orig = orig_stream
        self.file = file_obj
        self.lock = threading.Lock()

    def write(self, data):
        with self.lock:
            try:
                self.orig.write(data)
            except Exception:
                pass
            try:
                self.file.write(data)
            except Exception:
                pass

    def flush(self):
        try:
            self.orig.flush()
        except Exception:
            pass
        try:
            self.file.flush()
        except Exception:
            pass

class Logger:
    def __init__(self):
        self._orig_stdout = None
        self._orig_stderr = None
        self._file = None
        self._stdout_tee = None
        self._stderr_tee = None
        self.log_path = None

    def start_logging(self, title="log", folder="logs", timestamp_format="%Y%m%d_%H%M%S"):
        """
        Start logging: create folder, open file, and tee stdout/stderr to file.
        Returns the Path to the log file.
        """
        if self._file is not None:
            return self.log_path  # already started

        # determine script directory (fallback to cwd if __file__ not present)
        if "__file__" in globals():
            base_dir = Path(__file__).resolve().parent
        else:
            base_dir = Path.cwd()

        logs_dir = base_dir / folder
        logs_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime(timestamp_format)
        filename = f"{title}_{timestamp}.txt"
        self.log_path = logs_dir / filename

        # open file
        self._file = open(self.log_path, "w", encoding="utf-8")

        # write header
        self._file.write(f"Log start time: {datetime.now().isoformat()}\n")
        self._file.write(f"Working dir: {base_dir}\n\n")
        self._file.flush()

        # replace stdout/stderr with tee streams
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        self._stdout_tee = _TeeStream(self._orig_stdout, self._file)
        self._stderr_tee = _TeeStream(self._orig_stderr, self._file)
        sys.stdout = self._stdout_tee
        sys.stderr = self._stderr_tee

        print(f"Logging started. Log file: {self.log_path}")  # goes to both
        return self.log_path

    def stop_logging(self):
        """Stop logging and restore original stdout/stderr."""
        if self._file is None:
            return None

        # write footer
        try:
            print(f"\nLog end time: {datetime.now().isoformat()}")
        except Exception:
            pass

        # restore
        sys.stdout = self._orig_stdout or sys.__stdout__
        sys.stderr = self._orig_stderr or sys.__stderr__

        try:
            self._file.flush()
            self._file.close()
        except Exception:
            pass

        path = self.log_path
        # clear internal state
        self._file = None
        self._orig_stdout = None
        self._orig_stderr = None
        self._stdout_tee = None
        self._stderr_tee = None
        self.log_path = None
        return path

# module-level singleton for convenience
_default_logger = Logger()

def start_logging(title="LeNet5_MNIST", folder="logs"):
    return _default_logger.start_logging(title=title, folder=folder)

def stop_logging():
    return _default_logger.stop_logging()
