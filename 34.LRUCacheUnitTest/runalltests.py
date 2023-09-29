import sys
import os
os.chdir(os.path.dirname(__file__))
sys.path.append("../../tests/src")
sys.path.append(".test")
import test
test.main()