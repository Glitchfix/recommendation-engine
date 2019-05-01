import numpy as np

from lightfm.datasets import fetch_movielens

data = fetch_movielens(min_rating=5.0)

import pickle
fp=open("prototypeV2.3.pkl","rb")
model = pickle.load(fp)
fp.close()

def sample_recommendation(model, data, user_ids):

    # n_users, n_items = data['train'].shape
    _, n_items = data['train'].shape

    for user_id in user_ids:
        known_positives = data['item_labels'][data['train'].tocsr()[
            user_id].indices]

        scores = model.predict(user_id, np.arange(n_items))
        top_items = data['item_labels'][np.argsort(-scores)]

        print("User %s" % user_id)
        print("     Known positives:")

        for x in known_positives[:3]:
            print("        %s" % x)

        print("     Recommended:")

        for x in top_items[:3]:
            print("        %s" % x)
        return known_positives[:3], top_items[:3]

def detect(x):
    return sample_recommendation(model, data, [x])
