import datetime
import json
import os
# from great_expectation_function.build_expectation_suite import DataQuality
from script.great_expectation_function.build_expectation_suite import DataQuality

# initial the root directory for great expectation
root_directory = "/great_expectation"
# initial our data quality checker
def data_validation(dataframe, datasource_name:str, overwrite: bool):
    f = open('/code/notebook/script/great_expectation_function/myfile.txt', 'r')
    list_datasource = [x for x in f.readlines()]
    if datasource_name not in list_datasource:
        dq_checker = DataQuality(root_directory, datasource_name, dataframe, partition_date = datetime.datetime.now().date())
        # for the first time, create expectation suite with the similar name to your datasource_name
        dq_checker.create_expectation_suite_if_not_exist()
        # build the suite with automatic profiling.
        validator = dq_checker.get_validator(with_profiled=True, append_suit = True)
        # validate your data
        result = dq_checker.validate_data()
    else:
        if overwrite:
            dq_checker = DataQuality(root_directory, datasource_name, dataframe, partition_date = datetime.datetime.now().date())
            # for the first time, create expectation suite with the similar name to your datasource_name
            dq_checker.create_expectation_suite_if_not_exist()
            # build the suite with automatic profiling.
            validator = dq_checker.get_validator(with_profiled=True, append_suit = True)
            # validate your data
            result = dq_checker.validate_data()
        else:
            dq_checker = DataQuality(root_directory, datasource_name, dataframe, partition_date = datetime.datetime.now().date())
            result = dq_checker.validate_data()
    return dq_checker.render_file

def path_to_json(dataframe, datasource_name:str, overwrite:bool = False):
    path_to_list  = data_validation(dataframe, datasource_name, overwrite).split("/")
    template_dir = os.path.abspath("/".join(path_to_list[:-1]))
    a = {}
    a["flask_path"] = template_dir
    a["html_path"] = path_to_list[-1]
    with open("/code/notebook/script/great_expectation_function/path.json", "w") as outfile:
        json.dump(a, outfile)
    with open("/great_expectation/path.json", "w") as outfile:
        json.dump(a, outfile)
    

