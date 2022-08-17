from pathlib import Path
import os 
def get_data_path():
    return Path(__file__).parents[1]/"data"

def get_logging_path():

    logging_path = Path(__file__).parents[1] / "logging"
    if not os.path.exists(logging_path):
        logging_path.mkdir()
    return str(logging_path)

if __name__ == "__main__":
    print(get_data_path())