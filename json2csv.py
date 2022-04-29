
path = "/truba_scratch/agasi/Dataset/"

import json, os
import pandas as pd

def vrd_dicts(path):
    json_file = os.path.join(path, "relationships.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for v in imgs_anns:
        record = {}
        
        record["image_id"] = v["image_id"]
        annos = v["relationships"]
        
        objs = []
        for anno in annos:
            obj = {
               
                "subject_bbox": {"x":anno["subject"]["x"], "y":anno["subject"]["y"], "w":anno["subject"]["w"], "h":anno["subject"]["h"]},
                "object_bbox": {"x":anno["object"]["x"], "y":anno["object"]["y"], "w":anno["object"]["w"], "h":anno["object"]["h"]},
                "object":  {anno["object"]["name"]},
                "subject":  {anno["subject"]["name"]},
                "predicate": {anno["predicate"]}
            }
           
            objs.append(obj)
            
        record["relationships"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


result = vrd_dicts(path)

file_name = "/truba_scratch/agasi/Dataset/relationships_json.json"

with open(file_name, "w") as f: 
     json.dump(result, f,indent=2)
f.close()


# JSON to CSV

file_name = "/truba_scratch/agasi/Dataset/relationships_json.json"

with open(file_name) as f: 
     json_data = json.loads(f.read())
f.close()
        
df = pd.json_normalize(json_data, record_path=["relationships"], meta=["image_id"])
df.to_csv("/truba_scratch/agasi/Dataset/relationships.csv",index = False, header = False, sep=";")

df