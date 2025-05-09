import json
import logging
from datetime import datetime, timezone
from typing import Dict, Optional

LOG_RECORD_BUILTIN_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class JSONFormatter(logging.Formatter):
    """Custom logging formatter that outputs logs in JSON format."""

    def __init__(self, *, fmt_keys: Optional[Dict[str, str]] = None):
        """Initialize the JSON formatter."""
        super().__init__()
        self.fmt_keys = fmt_keys or {}

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record into a JSON string."""
        message = self._prepare_log_dict(record)
        return json.dumps(message, default=str)

    def _prepare_log_dict(self, record: logging.LogRecord) -> dict:
        """Prepare a dictionary with log details."""
        always_fields = {
            "timestamp": (
                datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
            ),
            "message": record.getMessage(),
        }

        if record.exc_info is not None:
            always_fields["exc_info"] = self.formatException(record.exc_info)

        if record.stack_info is not None:
            always_fields["stack_info"] = self.formatStack(record.stack_info)

        message = {
            key: always_fields.pop(val, None) or getattr(record, val, None)
            for key, val in self.fmt_keys.items()
        }
        message.update(always_fields)

        for key, val in record.__dict__.items():
            if key not in LOG_RECORD_BUILTIN_ATTRS:
                message[key] = val

        return message
