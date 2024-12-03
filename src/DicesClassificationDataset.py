from torch.utils.data import Dataset
import pandas as pd
from DicesFields import Fields

class DicesClassificationPrompts(Dataset):

    def __init__(self, data: pd.DataFrame, instructions: str):
        self.data = data
        self.instructions = instructions

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx:int)-> tuple[int, str, str]:
        
        label = "SAFE" if self.data.iloc[idx][Fields.MAJORITY_Q_OVERALL.value] == "No" else "UNSAFE"
        return idx, self.__create_instruction__(self.data.iloc[idx]), label

    def __create_instruction__(self, data) -> str:

        prompt = data['context']
        response = data["response"]
        prompt = prompt.replace("LAMBDA:", "A:")
        prompt = prompt.replace("USER:", "B:")
 
        end_response = "B:"
        return f"{self.instructions}{prompt}{end_response} {response}\n"

    def get_value_by_idx(self, id: int, field: Fields):
        return self.data.iloc[id][field.value]
    
    def get_template_format(self,prompt,tokenizer):
        chat = [
            {
                "role": "user",
                "content": prompt
            }]

        return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)