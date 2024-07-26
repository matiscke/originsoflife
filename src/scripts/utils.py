# Code that we need that doesn't produce figures
import paths
import time


def save_var_latex(key, value, datafile=paths.tex / "variables.dat"):
    # https://stackoverflow.com/questions/59408823/pdf-latex-with-python-script-custom-python-variables-into-latex-output on 2023-01-13
    import csv
    import os

    dict_var = {}

    file_path = os.path.join(os.getcwd(), datafile)

    try:
        with open(file_path, newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                dict_var[row[0]] = row[1]
    except FileNotFoundError:
        pass

    # Check if the value is a float, if so format it to have a maximum of two decimal places
    if isinstance(value, float):
        value = round(value, 2)

    # escape any % characters
    value = str(value).replace("%", "\\%")

    dict_var[key] = value

    with open(file_path, "w") as f:
        for key in dict_var.keys():
            f.write(f"{key},{dict_var[key]}\n")

    return None


def read_var_latex(key, datafile=paths.tex / "variables.dat"):
    """read the value of variable `key` from a file."""
    import csv
    import os

    dict_var = {}

    file_path = os.path.join(os.getcwd(), datafile)

    try:
        with open(file_path, newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                dict_var[row[0]] = row[1]
    except FileNotFoundError:
        pass

    return dict_var[key]


def timeit(func):
    """
    Decorator for measuring a function's processing time.
    """

    def measure_time(*args, **kw):
        start_time = time.time()
        result = func(*args, **kw)
        elapsed_time = time.time() - start_time

        # Calculate hours, minutes, and seconds
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        # Print formatted time
        print(
            f"Processing time of {func.__qualname__}(): {int(hours):02}:{int(minutes):02}:{seconds:.0f}"
        )
        return result

    return measure_time
