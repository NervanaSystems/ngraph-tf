from __future__ import print_function
import os, sys, datetime

def check_copyright(dirname, year = None, pred = lambda x : True):
    no_copyrights = []; bad_year = []
    files = os.walk(dirname).next()[2]
    for file_under_test in [dirname + "/" + file for file in files if pred(file)]:
        with open(file_under_test) as f:
            copyright_found = False
            for line in f.readlines():
                if 'Copyright' in line and 'Intel' in line:
                    copyright_found = True
                    break
            if not copyright_found:
                no_copyrights.append(file_under_test)
            elif year is not None:  #check year
                year_found = False
                for item in line.split(' '):
                    str_under_test = item.split('-')[-1]
                    if str_under_test.isdigit():
                        if int(str_under_test) == year:
                            year_found = True
                            break
                if not year_found:
                    bad_year.append((file_under_test, line))

    return no_copyrights, bad_year


def iterate_over_subdirs_and_check_copyright(start_folder, children_to_explore = None):
    year = datetime.datetime.now().year
    start_folder = start_folder.rstrip('/')
    if children_to_explore is None:
        subdirs = [x[0] for x in os.walk(start_folder)]
    else:
        immediate_subdirs = [name for name in os.listdir(start_folder) if os.path.isdir(os.path.join(start_folder, name))]
        for item in children_to_explore:
            if item not in immediate_subdirs:
                assert False, item + " not found in " + start_folder
        subdirs = children_to_explore
    subdirs = [i.rstrip('/') for i in subdirs]
    file_exts = ['py', 'cpp', 'h', 'cc', 'c', 'C', 'c++', 'cxx', 'hpp', 'h++']
    for subdir in subdirs:
        no_copyrights, bad_year = check_copyright(start_folder + '/' + subdir, year, pred = lambda x : any([i == x.split('.')[-1] for i in file_exts]))
        for flname in no_copyrights:
            print ('No copyright line found: ' + flname)
        for flname, line in bad_year:
            print ('Has copyright line, but possibly bad year. Copyright line is: ' + line + ' in ' + flname)
    
if __name__ == '__main__':
    iterate_over_subdirs_and_check_copyright(sys.argv[1], ['src', 'examples', 'test', 'logging', 'tools', 'diagnostics', 'python'])