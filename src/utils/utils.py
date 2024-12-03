import os
def set_env_variables(variable_name:str, local_var: any):
    var_val =os.getenv(variable_name) 
    if not var_val :
        raise KeyError(f"Environemnt variable {variable_name}  not found. Run the setup script to configure the project")
    local_var = var_val