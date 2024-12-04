import argparse
from pathlib import Path
import os
from dotenv import load_dotenv, set_key

def set_env_variables():
    if not os.getenv("ROOT_DIR"):
        print("Saving ROOT_DIR as env variable")
        set_key(env_path, "ROOT_DIR", root_dir)
    
    if not os.getenv("HUGGINGFACE_KEY"):
        if not args.huggingface_key:
            print(f"[ERROR]: HUGGINGFACE_KEY not found as an env variable. SETUP INCOMPLETE.. ")
            print(f"To Fix: Add an Add key=HUGGINGFACE_KEY to your .env file at {env_path}") 
        else:
            print("Saving HUGGINGFACE_KEY as env variable")
            set_key(env_path, "HUGGINGFACE_KEY", args.huggingface_key)
    
    if not os.getenv("OUTPUT_DIR"):
        OUT_DIR = os.path.join(root_dir, "output")
        print("Saving OUTPUT_DIR as env variable")
        set_key(env_path, "OUTPUT_DIR", OUT_DIR)

    if not os.getenv("DATA_DIR"):
        DATA_DIR = os.path.join(root_dir, "data")
        print("Saving DATA_DIR as env variable")
        set_key(env_path, "OUTPUT_DIR", DATA_DIR)

def create_dir():
    DATA_PATH = os.getenv("DATA_DIR")
    custom_data_dir = os.path.join(DATA_PATH, "custom")
    if not os.path.exists(custom_data_dir):
        print("Creating dir for custom data")
        os.mkdir(custom_data_dir)

    


def main():
    parser = argparse.ArgumentParser(description="A script to setup the development env of DICES_CLASSIFICATION.")
    parser.add_argument('--project_dir', type=str, required=True, help="Absolute path to the root dir of the project")
    parser.add_argument("--huggingface_key", type=str, required=False, help="Hugginface hub key")
    args = parser.parse_args()

    root_dir = args.project_dir
    env_path = Path(os.path.join(root_dir, ".env"))

    if not env_path.exists():
        print(f".env does not exists at {env_path}. Create one..")
        env_path.touch()

    load_dotenv(env_path)    
    set_env_variables()
    create_dir()



    pass

if __name__ == "__main__":
    main()