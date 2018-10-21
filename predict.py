from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
import numpy as np
import pandas as pd
import os

def predict(X_test):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="mykey.json"
    project = "go-scraper-209117"
    model = "HackGTKiller"
    version = "version1"
    print (X_test.shape)
    print (len([x for x in X_test]))
    instances = []
    for y in [x.tolist() for x in X_test]:
        instances.append({"image": y})
    # instances = [{"image": [x for x in X_test]}]

    service = discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    # print (response['predictions'])

    df2 = pd.DataFrame(response['predictions'])
    i = []
    for entry in df2.iterrows():
        i.append(entry[1]['scores'][0])
    return np.asarray(i).reshape(-1,1)