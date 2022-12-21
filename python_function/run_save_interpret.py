from save_interpret import save_to_csv

if __name__ == "__main__":

    task = input('Enter your task:')
    target = input('Enter your target:')
    save_to_csv(task, target)