import unittest
import pdb
import sys
import argparse
import os

#todo:explain try catch
try:
    import ngraph
except:
    pass

loader = unittest.TestLoader()
parser = argparse.ArgumentParser()
parser.add_argument('--list_tests', help="List of test cases in this module")
parser.add_argument('--run_test', help="Runs the testcase and returns the output")
parser.add_argument('--tensorflow_path', help="Specify the path where Tensorflow is installed")
arguments = parser.parse_args()

def main():
    set_path()

    if(arguments.list_tests):
        list_tests()

    if(arguments.run_test):
        run_test()

def set_path():
    tf_path = arguments.tensorflow_path
    sys.path.append(tf_path + "/tensorflow/python/kernel_tests")
    print(sys.path)

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


