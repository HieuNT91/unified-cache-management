from importlib.metadata import distribution
from pathlib import Path
__path__.append(str(Path(distribution("uc-manager").locate_file("ucm/store/pcstore"))))
