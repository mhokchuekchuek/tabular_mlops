from typing import TypeVar
from pandas_profiling import ProfileReport
import os
import json
Pandas_df= TypeVar('pandas')

def _check(name:str):
    f = open('/code/notebook/script/eda/myfile.txt', 'r')
    list_datasource = [x for x in f.readlines()]
    if name not in list_datasource:
        file1 = open('/code/notebook/script/eda/myfile.txt', 'w')
        file1.writelines([name])
        return True
    return False
    
def _eda(df: Pandas_df, name: str, overwrite: bool = False) -> str:
    root_path = "/eda_html"
    html_path = f"{root_path}/{name}.html"
    if _check(name):
        profile = ProfileReport(df)
        profile.to_file(output_file= html_path)
    else:
        if overwrite:
            profile = ProfileReport(df)
            profile.to_file(output_file= html_path)
    return html_path

def eda(df: Pandas_df, name: str, overwrite: bool = False) -> str:
    path_to_list  = _eda(df, name, overwrite).split("/")
    template_dir = os.path.abspath("/".join(path_to_list[:-1]))
    a = {}
    a["flask_path"] = template_dir
    a["html_path"] = path_to_list[-1]
    with open("/code/notebook/script/eda/path.json", "w") as outfile:
        json.dump(a, outfile)
    with open("/eda_html/path.json", "w") as outfile:
        json.dump(a, outfile)