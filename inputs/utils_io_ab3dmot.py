import copy
import glob
import os

import numpy as np
# The full file is taken from AB3DMOT https://github.com/xinshuoweng/AB3DMOT to load their detections


def isstring(string_test):
    try:
        return isinstance(string_test, str)
    except NameError:
        return isinstance(string_test, str)


def islist(list_test):
    return isinstance(list_test, list)


def islogical(logical_test):
    return isinstance(logical_test, bool)


def isinteger(integer_test):
    if isinstance(integer_test, np.ndarray):
        return False
    try:
        return isinstance(integer_test, int) or int(integer_test) == integer_test
    except (TypeError, ValueError):
        return False


def is_path_valid(pathname):
    try:
        if not isstring(pathname) or not pathname:
            return False
    except TypeError:
        return False
    else:
        return True


def is_path_creatable(pathname):
    """
    if any previous level of parent folder exists, returns true
    """
    if not is_path_valid(pathname):
        return False
    pathname = os.path.normpath(pathname)
    pathname = os.path.dirname(os.path.abspath(pathname))

    # recursively to find the previous level of parent folder existing
    while not is_path_exists(pathname):
        pathname_new = os.path.dirname(os.path.abspath(pathname))
        if pathname_new == pathname:
            return False
        pathname = pathname_new
    return os.access(pathname, os.W_OK)


def is_path_exists(pathname):
    try:
        return is_path_valid(pathname) and os.path.exists(pathname)
    except OSError:
        return False


def is_path_exists_or_creatable(pathname):
    try:
        return is_path_exists(pathname) or is_path_creatable(pathname)
    except OSError:
        return False


def safe_path(input_path, warning=True, debug=True):
    """
    convert path to a valid OS format, e.g., empty string '' to '.', remove redundant '/' at the end from 'aa/' to 'aa'

    parameters:
        input_path:		a string

    outputs:
        safe_data:		a valid path in OS format
    """
    if debug: assert isstring(input_path), 'path is not a string: %s' % input_path
    safe_data = copy.copy(input_path)
    safe_data = os.path.normpath(safe_data)
    return safe_data


def fileparts(input_path, warning=True, debug=True):
    """
    this function return a tuple, which contains (directory, filename, extension)
    if the file has multiple extension, only last one will be displayed

    parameters:
        input_path:     a string path

    outputs:
        directory:      the parent directory
        filename:       the file name without extension
        ext:            the extension
    """
    good_path = safe_path(input_path, debug=debug)
    if len(good_path) == 0: return '', '', ''
    if good_path[-1] == '/':
        if len(good_path) > 1:
            return good_path[:-1], '', ''  # ignore the final '/'
        else:
            return good_path, '', ''  # ignore the final '/'

    directory = os.path.dirname(os.path.abspath(good_path))
    filename = os.path.splitext(os.path.basename(good_path))[0]
    ext = os.path.splitext(good_path)[1]
    return directory, filename, ext


def load_list_from_folder(folder_path, ext_filter=None, depth=1, recursive=False, sort=True, save_path=None,
                          debug=True):
    """
    load a list of files or folders from a system path

    parameters:
        folder_path:    root to search
        ext_filter:     a string to represent the extension of files interested
        depth:          maximum depth of folder to search, when it's None, all levels of folders will be searched
        recursive:      False: only return current level
                        True: return all levels till to the input depth

    outputs:
        full_list:       a list of elements
        num_elem:       number of the elements
    """
    folder_path = safe_path(folder_path)
    if not is_path_exists(folder_path):
        print('the input folder does not exist\n')
        return [], 0
    if debug:
        assert islogical(recursive), 'recursive should be a logical variable: {}'.format(recursive)
        assert depth is None or (isinteger(depth) and depth >= 1), 'input depth is not correct {}'.format(depth)
        assert ext_filter is None or (
            islist(ext_filter) and all(isstring(ext_tmp) for ext_tmp in ext_filter)) or isstring(
            ext_filter), 'extension filter is not correct'
    if isstring(ext_filter): ext_filter = [ext_filter]  # convert to a list
    # zxc

    full_list = list()
    if depth is None:  # find all files recursively
        recursive = True
        wildcard_prefix = '**'
        if ext_filter is not None:
            for ext_tmp in ext_filter:
                # wildcard = os.path.join(wildcard_prefix, '*' + string2ext_filter(ext_tmp))
                wildcard = os.path.join(wildcard_prefix, '*' + ext_tmp)
                curlist = glob2.glob(os.path.join(folder_path, wildcard))
                if sort: curlist = sorted(curlist)
                full_list += curlist
        else:
            wildcard = wildcard_prefix
            curlist = glob2.glob(os.path.join(folder_path, wildcard))
            if sort: curlist = sorted(curlist)
            full_list += curlist
    else:  # find files based on depth and recursive flag
        wildcard_prefix = '*'
        for index in range(depth - 1): wildcard_prefix = os.path.join(wildcard_prefix, '*')
        if ext_filter is not None:
            for ext_tmp in ext_filter:
                # wildcard = wildcard_prefix + string2ext_filter(ext_tmp)
                wildcard = wildcard_prefix + ext_tmp
                curlist = glob.glob(os.path.join(folder_path, wildcard))
                if sort: curlist = sorted(curlist)
                full_list += curlist
        # zxc
        else:
            wildcard = wildcard_prefix
            curlist = glob.glob(os.path.join(folder_path, wildcard))
            # print(curlist)
            if sort:
                curlist = sorted(curlist)
            full_list += curlist
        if recursive and depth > 1:
            newlist, _ = load_list_from_folder(folder_path=folder_path, ext_filter=ext_filter, depth=depth - 1,
                                               recursive=True)
            full_list += newlist

    full_list = [os.path.normpath(path_tmp) for path_tmp in full_list]
    num_elem = len(full_list)

    # save list to a path
    if save_path is not None:
        save_path = safe_path(save_path)
        if debug: assert is_path_exists_or_creatable(save_path), 'the file cannot be created'
        with open(save_path, 'w') as file:
            for item in full_list: file.write('%s\n' % item)
        file.close()

    return full_list, num_elem
