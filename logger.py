import sys
import pathlib
import datetime

# shoddy Logger class.
# don't recommended for consumption or use.
# unless you are desperate.


class Logger:
    def __init__(self):
        print("!!! LOGGING DATA IN THE BACKGROUND. DO NOT CLOSE. !!!")
        print("Logging:", pathlib.Path().absolute())
        self.filename = str(pathlib.Path().absolute().name) + \
            "-" + str(datetime.datetime.now()) + ".txt"
        print("Path:", str(pathlib.Path().absolute()) + "/" + self.filename)
        # this is global, not local to the class
        sys.stdout = open(str(pathlib.Path().absolute()) +
                          "/" + self.filename, "w")
        print("Log date:", str(datetime.datetime.now()), "\n")

    def __del__(self):
        sys.stdout = sys.__stdout__
        print("Log saved at:", str(pathlib.Path().absolute()) + "/" + self.filename)
