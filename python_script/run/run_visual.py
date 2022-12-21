import os
def runner():
    os.system("kill $(lsof -t -i:8080)")
    os.system("kill $(lsof -t -i:8081)")
    return os.system("cd script/run ; python3 app.py")
