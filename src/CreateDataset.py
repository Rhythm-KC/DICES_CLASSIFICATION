import pandas as pd
import os
from DicesFields import Fields
from utils.utils import set_env_variables

MAJORITY_OVERALL_KEY_NAME = Fields.MAJORITY_Q_OVERALL.value
CONTEXT = Fields.CONTEXT.value
RESPONSE = Fields.RESPONSE.value
RELEVENT_KEYS = [CONTEXT, RESPONSE, Fields.DEGREE_OF_HARM.value, 
                 MAJORITY_OVERALL_KEY_NAME, Fields.RATER_RACE.value]

base_dir = ""
set_env_variables("ROOT_DIR", base_dir)

DATA_DIR = ""
set_env_variables("DATA_DIR", DATA_DIR)

def get_max_overall_safety(series):
    return series.mode()[0]

def join_context(data):
    if Fields.Q_OVERALL.value not in data.columns:
        raise ValueError(f"{Fields.Q_OVERALL.value} not in the data")
    majority_rating = data.groupby(Fields.CONTEXT.value)[Fields.Q_OVERALL.value].apply(get_max_overall_safety).reset_index(name=MAJORITY_OVERALL_KEY_NAME)
    merged = pd.merge(data, majority_rating, on=Fields.CONTEXT.value)
    return merged.drop_duplicates(subset=[Fields.CONTEXT.value, MAJORITY_OVERALL_KEY_NAME])

def join_dices(data: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(data, join="outer", ignore_index=True)

def seperate_on_white_or_not(data):
    return (data[data[Fields.RATER_RACE.value] == "White"], data[data[Fields.RATER_RACE.value] != "White"])

def get_only_relevent_keys(data, keys):
    return data[keys]

def replace_string(data, column_name, replace_this, replace_with):
    data[column_name] = data[column_name].apply(lambda x: x.replace(replace_this, replace_with))

def open_data(paths: list[str]) -> list[pd.DataFrame]:
    data = []
    for path in paths:
        data.append(pd.read_csv(path))
    return data

def create_dataset():
    dices_paths = [os.path.join(DATA_DIR, "dices/350/diverse_safety_adversarial_dialog_350.csv"),
                   os.path.join(DATA_DIR, "dices/990/diverse_safety_adversarial_dialog_350.csv"),
                   ]

    # joining all the dices data set together 
    data = open_data(dices_paths)
    if len(data) > 0:
        data = join_dices(data)
    else:
        data = data[0]

    # seperating based on race
    dices_white, dices_non_white = seperate_on_white_or_not(data)

    # making sure that the seperate is properly
    assert(len(dices_white) + len(dices_non_white) == len(data))

    # merging dices data base on the context so we can get the majority vote
    dices_white = get_only_relevent_keys(join_context(dices_white), RELEVENT_KEYS)
    replace_string(dices_white, CONTEXT, "\n", "\\n")
    replace_string(dices_white, RESPONSE, "\n", "\\n")


    dices_non_white = get_only_relevent_keys(join_context(dices_non_white), RELEVENT_KEYS)
    replace_string(dices_non_white, CONTEXT, "\n", "\\n")
    replace_string(dices_non_white, RESPONSE, "\n", "\\n")

    dices_white.to_csv(os.path.join(DATA_DIR, "dices/white_responses.csv"), sep=",")
    dices_non_white.to_csv(os.path.join(DATA_DIR, "dices/non_white_responses.csv"), sep=",")

def run():

    output = "CREATED DATA"

    if (os.path.exists(os.path.join(DATA_DIR, "dices/white_responses.csv")) and 
        os.path.exists(os.path.join(DATA_DIR, "dices/non_white_responses.csv"))):
        output = "DATA ALREADY EXIST"
    else:
        create_dataset()
    print(output)
    return 0

if __name__ == "__main__":
    exitcode = run()
    if exitcode == 0:
        exit(0)
    else:
        exit(1)



    



