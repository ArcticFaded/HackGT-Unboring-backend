import requests
import json
from pymongo import MongoClient
from requests.auth import HTTPBasicAuth


client = MongoClient()
db = client['hackgt']
collection = db['ncr']

url = "https://gateway-staging.ncrcloud.com/transaction-document/transaction-documents/find"

payload = "{\n  \"siteInfoIds\": [\"7c54465e9f5344598276ec1f941f5a3c\"]\n}"
headers = {
    'Content-Type': "application/json",
    'Cache-Control': "no-cache",
    'Accept': "application/json",
    'nep-organization': "ncr-market",
    'nep-service-version': "2.2.0:2",
    'nep-application-key': "8a0084a165d712fd01668f6c00f30068",
    'cache-control': "no-cache",
    }

response = requests.post(url, data=payload, headers=headers, auth=HTTPBasicAuth('acct:unboring@unboringserviceuser','Hoosiers2'))

collection.insert_many(response.json()["pageContent"])
print("Done")
#print(json.loads(response.json()))

