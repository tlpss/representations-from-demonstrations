from pathlib import Path

def get_data_path():
    return Path(__file__).parents[1]/"data"


if __name__ == "__main__":
    print(get_data_path())