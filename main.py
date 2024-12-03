import src.CreateDataset as createdataset
import src.run_classification as classify

try:
    createdataset.run()
    classify.main()

except Exception as e:
    print(f"ran into exceptions \n{e.with_traceback}")
    
