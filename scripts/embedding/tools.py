import codecs
import pickle
import numpy as np
from functools import lru_cache
import codecs
from directories import dicrectories

class tools:      
    @staticmethod    
    @lru_cache(maxsize=None)
    def read_pickle_data(path):
        if dicrectories.pickle_exist(path):
            with open(path, "rb") as saved:
                try:
                    return pickle.load(saved)
                except (pickle.UnpicklingError, EOFError):
                    # print("Error: The file could not be unpickled: ",path)
                    return []
        else:
            # print("Error: The file does not exist: ",path)
            return []

    @staticmethod
    def print_training_time(seconds):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        print(f"Training duration: {hours} hours, {minutes} minutes, {seconds} seconds")
