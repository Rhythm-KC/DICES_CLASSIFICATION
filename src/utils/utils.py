import os
def get_env_variables(variable_name:str):
    var_val =os.getenv(variable_name) 
    if not var_val :
        raise KeyError(f"Environemnt variable {variable_name}  not found. Run the setup script to configure the project")
    return var_val