from functools import wraps

function_call = ""

# test_dictionary0 = {
#     "f1": 1,
#     "f2": 2,
#     "f3": 3,
#
#     1: "f1",
#     2: "f2",
#     3: "f3"
# }
#
# executed_methods0 = {
#     1: False,
#     2: False,
#     3: False
# }
#
# dependent_methods0 = {
#     1: [],
#     2: [1],
#     3: [2, 1]
# }
#
# test_dictionary = None
# executed_methods = None
# dependent_methods = None


def function_details(func):
    # Getting the argument names of the
    # called function
    argnames = func.__code__.co_varnames[:func.__code__.co_argcount]

    # Getting the Function name of the
    # called function
    fname = func.__name__

    @wraps(func)
    def inner_func(*args, **kwargs):
        # global dependent_methods0, executed_methods0, test_dictionary0
        # global dependent_methods, executed_methods, test_dictionary
        #
        # if fname == "__init__":
        #     dependent_methods = dependent_methods0.copy()
        #     executed_methods = executed_methods0.copy()
        #     test_dictionary =  test_dictionary0.copy()
        #     return func(*args, **kwargs)
        #
        # fnum = test_dictionary[fname]
        # can_be_executed = True
        # for i in dependent_methods[fnum]:
        #     if not executed_methods[i]:
        #         can_be_executed = False
        #         print(f"{test_dictionary[i]} hasn't been executed")
        # if not can_be_executed:
        #     raise Exception("Cannot calculate given method before methods above!")
        # else:
        #     print("yaaay method can be executed")
        # executed_methods[fnum] = True
        #
        # ---------
        # {'add_aligned_dict': 18,
        #  'add_files': 2,
        #  'generate_table': 8,
        #  'check_for_formulas': 15,
        #  'match_features': 12,
        #  'calculate_mean': 9,
        #  'merge_duplicate_rows': 13,
        #  'calculate_suspect_list_scores': 11,
        #  'export_filtered_ms': 14,
        #  'extract_sim': 3,
        #  'export_tables': 16,
        #  'export_tables_averaged': 10,
        #  'get_ms2_spectrum': 4,
        #  'filter_constant_ions': 5,
        #  'merge_features': 6,
        #  'merged_msms_from_table': 7}
        # d2
        # Out[14]:
        # {18: 'add_aligned_dict',
        #  2: 'add_files',
        #  8: 'generate_table',
        #  15: 'check_for_formulas',
        #  12: 'match_features',
        #  9: 'calculate_mean',
        #  13: 'merge_duplicate_rows',
        #  11: 'calculate_suspect_list_scores',
        #  14: 'export_filtered_ms',
        #  3: 'extract_sim',
        #  16: 'export_tables',
        #  10: 'export_tables_averaged',
        #  4: 'get_ms2_spectrum',
        #  5: 'filter_constant_ions',
        #  6: 'merge_features',
        #  7: 'merged_msms_from_table'}

        global function_call
        function_call += fname + "("
        # printing the function arguments
        function_call += ', '.join('% s = % r' % entry for entry in zip(argnames, args[:len(argnames)])) + ", "
        # Printing the variable length Arguments
        function_call += "args =" + repr(list(args[len(argnames):])) + ", "
        # Printing the variable length keyword
        # arguments
        function_call += "kwargs =" + repr(kwargs) + ")\n"
        return func(*args, **kwargs)

    return inner_func

    # @wraps(func)
    # def inner_func(*args, **kwargs):
    #     global function_call
    #     function_call += fname + "("
    #
    #     # printing the function arguments
    #     function_call += ', '.join('% s = % r' % entry for entry in zip(argnames, args[:len(argnames)])) + ", "
    #
    #     # Printing the variable length Arguments
    #     function_call += "args =" + repr(list(args[len(argnames):])) + ", "
    #
    #     # Printing the variable length keyword
    #     # arguments
    #     function_call += "kwargs =" + repr(kwargs) + ")\n"
    #     return func(*args, **kwargs)
