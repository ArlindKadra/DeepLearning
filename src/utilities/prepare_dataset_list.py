import openml

def prepare_openml100():

    # revised version of OpenML100
    benchmark_suite = openml.study.get_study("99", "tasks")
    for task in list(benchmark_suite.tasks):
        with open("/home/kadraa/Documents/classification_tasks_100.txt", "a") as file:
            file.write(str(task))
            file.write(" ")
