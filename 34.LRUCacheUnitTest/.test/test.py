from ConsoleOutputTest import ConsoleOutputTestExpectedFileAsDependency as TestClass
from TestUtils import get_args

def main():
    newArgv, config_json_filepaths, nabla_dir, warnings = get_args()
    if TestClass("LRU cache test", config_json_filepaths, nabla_dir, warnings).run():
        print("Test finished, passed")
        exit(0)
    else:
        print()
        exit(1)
