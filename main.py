import sys
import os
import traceback
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
load_dotenv()

import CreateDataset as createdataset
import run_classification as classify

try:
    createdataset.run()
    classify.main()

except Exception as e:
    print(f"ran into exceptions \n{traceback.format_exc()}")
    
