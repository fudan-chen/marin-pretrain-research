import cloudpickle
import os
import sys
import traceback
import logging

# Reinitialize logging with the unified Iris format.
# Uses single-letter level prefix: I=INFO, W=WARNING, E=ERROR, D=DEBUG, C=CRITICAL.
# NOTE: This duplicates LevelPrefixFormatter and _LEVEL_PREFIX from rigging.log_setup
# because CALLABLE_RUNNER executes inside an isolated task container that may not
# have the rigging package installed (e.g. user-provided Docker images).
_LEVEL_PREFIX = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

class _LevelPrefixFormatter(logging.Formatter):
    def format(self, record):
        record.levelprefix = _LEVEL_PREFIX.get(record.levelname, "?")
        return super().format(record)

_root = logging.getLogger()
_root.handlers.clear()
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(_LevelPrefixFormatter(
    fmt="%(levelprefix)s%(asctime)s %(name)s %(message)s",
    datefmt="%Y%m%d %H:%M:%S",
))
_root.addHandler(_handler)
_root.setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
# botocore/aiobotocore log credential discovery + retry chatter at INFO once per
# fresh S3 session; pure noise on S3-backed tasks (mirror of rigging.log_setup).
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("aiobotocore").setLevel(logging.WARNING)

workdir = os.environ["IRIS_WORKDIR"]

try:
    with open(os.path.join(workdir, "_callable.pkl"), "rb") as f:
        fn, args, kwargs = cloudpickle.loads(f.read())
    fn(*args, **kwargs)
except Exception:
    traceback.print_exc()
    sys.exit(1)
