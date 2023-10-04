from .nbl import *

def main():
    newArgv, config_json_filepaths, nabla_dir, warnings = nbl.Utils.get_args()
    if nbl.ConsoleOutput.ExpectedFileAsDependencyTest("LRU cache test", config_json_filepaths, nabla_dir, warnings).run():
        print("Test finished, passed")
        exit(0)
    else:
        print()
        exit(1)
