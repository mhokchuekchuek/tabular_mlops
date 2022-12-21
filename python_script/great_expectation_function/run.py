import os
def runner():
    os.system("kill $(lsof -t -i:8080)")
    return os.system("cd script/great_expectation_function ; python3 app.py")

