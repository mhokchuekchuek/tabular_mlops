import os
def runner():
    os.system("kill $(lsof -t -i:8081)")
    return os.system("cd script/eda ; python3 app.py")

