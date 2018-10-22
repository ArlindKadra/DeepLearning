import os
import json

import xlsxwriter


def create_results(working_dir):

    workbook = xlsxwriter.Workbook(os.path.join(working_dir, 'Thesis_experiment.xlsx'))
    worksheet = workbook.add_worksheet('Results')
    highlight_format = workbook.add_format({'bold': True, 'font_color': 'red'})
    networks = ['FcResNet', 'FcNet']
    worksheet.write(0, 0, 'Task')
    # write network names in order
    for i in range(0, len(networks)):
        worksheet.write(0, i + 1, networks[i])
    # start from 1 as the 0th row contains column names
    row = 1
    for task_id in read_task_ids(working_dir):

        values = []
        for network in networks:
            try:
                with open(os.path.join(working_dir, 'task_%d' % task_id, network, "best_config_info.txt"), "r") as fp:
                    output = json.load(fp)
                    values.append(output['test_accuracy'])
            except FileNotFoundError:
                """ 
                The experiment has failed, a value
                of -1 showcases that
                """
                values.append(-1)
        # add task id
        worksheet.write(row, 0, task_id)
        # find the max index
        max_value = max(values)
        max_index = values.index(max_value)
        for i in range(0, len(values)):
            if i == max_index:
                worksheet.write(row, i + 1, values[i], highlight_format)
            else:
                worksheet.write(row, i + 1, values[i])
        # increment row
        row += 1

    workbook.close()


def read_task_ids(working_dir):

    numbers = []
    with open(os.path.join(working_dir, "classification_tasks_100.txt"), "r") as fp:

        # there is an illegal value for the int
        # cast if the last space is not removed
        line = fp.readline().rstrip()
        for number in line.split(" "):
            numbers.append(int(number))

    return numbers

# create_results("give input")
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000