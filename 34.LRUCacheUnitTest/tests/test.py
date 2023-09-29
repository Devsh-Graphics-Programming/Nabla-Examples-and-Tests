import sys
sys.path.append("Path to framework submodule")

from ConsoleOutputTest import ConsoleOutputTestExpectedFileAsDependency as TestClass
from ArgHandler import get_args

newArgv, config_json_filepaths, nabla_dir, warnings = get_args()
if TestClass("LRU cache test", config_json_filepaths, nabla_dir, warnings).run():
    print("Test finished, passed")
    exit(0)
else:
    print()
    exit(1)