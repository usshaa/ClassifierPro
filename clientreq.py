import json
import requests
from pathlib import Path

URL = "http://127.0.0.1:5000/classify"

file = Path("images/watch.jpg")

categories = ["sneakers","wristwatch","office desk"]
def img_class(file,categories):

    with open(file,"rb") as img:
        files = {
            "image":(file.name,img,"image/jpeg")
        }
        data = {
            "categories":json.dumps(categories)
        }

        response = requests.post(URL,files=files, data=data)
        print(response.json())

if __name__ == "__main__":
    img_class(file,categories)