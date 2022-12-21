from save_evaluate import to_csv_eval
if __name__ == "__main__":
    task = input('Enter your task:')
    myFile = open("/code/notebook/script/myFile1.txt", "w")
    myFile.write(task)
    myFile.close()
    to_csv_eval(task)