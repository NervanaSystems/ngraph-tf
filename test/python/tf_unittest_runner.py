import unittest
import sys
import argparse
import os

#Import ngraph in builds without ngraph embedded
try:
    import ngraph
except:
    pass

loader = unittest.TestLoader()
parser = argparse.ArgumentParser()
optional = parser._action_groups.pop() 
required = parser.add_argument_group('required arguments')
required.add_argument('--tensorflow_path', help="Specify the path where Tensorflow is installed. Eg:/localdisk/skantama/tf-ngraph/tensorflow \n", required=True)
optional.add_argument('--list_tests', help="Prints the list of test cases in this package. Eg:math_ops_test \n")
optional.add_argument('--run_test', help="Runs the testcase and returns the output. Eg:math_ops_test.math_ops_test.DivNoNanTest.testBasic")
parser._action_groups.append(optional)
arguments = parser.parse_args()

def main():
    all_dirs(arguments.tensorflow_path)

    if(arguments.list_tests):
        list_tests()

    if(arguments.run_test):
        run_test()

from fnmatch import fnmatch
def all_dirs(dirname):
    dirname = arguments.tensorflow_path
    pattern = "*_test.py"
    for path, subdirs, files in os.walk(dirname):
        for name in files:
            if fnmatch(name, pattern):
                sys.path.append(path)

def list_tests():
    test_module = arguments.list_tests
    module = __import__(test_module)
    test_modules = loader.loadTestsFromModule(module)
    alltests = []
    for test_class in test_modules:
        alltests.append( ([i.id() for i in test_class._tests]))
    print ('\n'.join((sorted(sum(alltests, [])))))


def run_test():
    test_name = arguments.run_test
    tests = loader.loadTestsFromName(test_name)
    test_result = unittest.TextTestRunner(verbosity=2).run(tests)
    testsRun = 0
    tests_run = []
    failures = []
    errors = []
    if test_result.wasSuccessful():
        tests_run.append([testsRun, 1])
        sys.exit()
    elif test_result.errors:
        errors.append([test_name, test_result.errors])
        tests_run.append([testsRun, 0])
    elif test_result.failures:
        failures.append([test_name, test_result.failures])
        tests_run.append([testsRun, -1])

if __name__ == '__main__':
    main()


