import os
from mistralai import Mistral
import json
api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

ocr_response = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "document_url",
        "document_url": "https://openreview.net/pdf?id=U0SijGsCHJ"
    },
    include_image_base64=True
)

markdown_list = [ocr_response.pages[i].markdown for i in range(len(ocr_response.pages))]

file = '\n'.join(markdown_list)

with open('ocr_response.md', 'w') as f:
    f.write(file)
# breakpoint()
# # save as json
# with open('ocr_response.json', 'w') as f:
#     json.dump(ocr_response, f)
