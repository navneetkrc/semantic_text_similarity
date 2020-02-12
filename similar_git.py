# from similar_new import get_similar_tickets

import numpy as np
import faiss
from faiss import normalize_L2
import pandas as pd
import sys, os
from datetime import datetime, timedelta
import requests
import json
import timeit
from datetime import datetime
import os.path, time
from datetime import datetime, timedelta
from threading import Timer
import pdb


# ids = np.load(f'ids_gensen.npy')
# index_faiss=faiss.read_index(f"index_gensen")

ids = np.load(f'ids_gloveee.npy')
index_faiss=faiss.read_index(f"index_gloveee")

print(index_faiss.is_trained)
print(index_faiss.ntotal)



def save_index():
    global PATH
    global index_faiss
    r=faiss.write_index(index_faiss, f"{PATH}/index_faiss_all")
    
    t = Timer(86400, save_index)
    t.start()
    
    return True


def apply_threshold(row, k=5, threshold=0.95):
    try:
        row=row.to_frame().T
        row.iloc[0,:k]=row.iloc[0,:k].astype(float)
        r=row.iloc[0, :k]
        l=len(r[r>=threshold])
        row.iloc[:, l:k]=np.nan
        row.iloc[:, k+l:k*2]=np.nan
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname =os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(e, exc_type, fname, exc_tb.tb_lineno)
    return row


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
        
        
def get_embeddings(texts):
    vecs = []
    if len(texts) > 1000:
        li_splitted = list(chunks(texts, 100))

        for i,grp in enumerate(li_splitted):
            print(f"{i}/{len(li_splitted)}", end="\r")
            r=requests.post(embeddings_service, json={"sentences_list": grp})
            arr = np.asarray(json.loads(r.text)["vectors"])
            vecs = np.append(vecs, arr)
        vecs=np.asarray(vecs).reshape(len(texts), 2048)
    else:
        r=requests.post(embeddings_service, json={"sentences_list": texts})
        vecs = np.asarray(json.loads(r.text)["vectors"])
    return vecs



    
    
    
def search_faiss(vec, k=5):
    print("Here we go !", type(index_faiss), vec.shape)
    top_k = index_faiss.search(vec, k)
    return list(top_k)

def get_similar_tickets(recs, k=5, threshold=0.95):
    print("function called")
    try:
        print("starting calculations")
        if "emb" not in recs.columns: 
            print("getting embeddings: ", recs.shape)
            
            texts=recs["text"].tolist()
            vecs = get_embeddings(texts)
            recs["emb"]= vecs.tolist()
        else:
            print("embedding col already present")

        xq = np.float32(np.asarray(recs["emb"].tolist()))
        normalize_L2(xq)

        print("query vector created !")
        
        result=search_faiss(xq, k)
        print("Got results from FAISS")
        
        nums = lambda t: "KEY"+str(int(t))#.zfill(7)
        vfunc = np.vectorize(nums)
        result[1]=vfunc(result[1])

        mydf=pd.DataFrame(np.column_stack(result))
        mydf["query_key"]=recs["key"].tolist()
        mydf["query_text"]=recs["text"].tolist()
        print("create result dataframe")
        mydf=mydf.rename(columns={
            0: "NN1_score",
            1: "NN2_score",
            2: "NN3_score",
            3: "NN4_score",
            4: "NN5_score",
            5: "NN1_number",
            6: "NN2_number",
            7: "NN3_number",
            8: "NN4_number",
            9: "NN5_number",
        })

        res=mydf.apply(apply_threshold, axis=1, args=(k,threshold))
        res=pd.concat(res.values.tolist())

        res = res.assign(is_similar=np.where(pd.isnull(res['NN1_number']) == True , False, True))

        res = pd.merge(recs, res.rename(columns={"query_key": "key"}), on="key", how="inner")
        
        return res
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname =os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("Exception get_similar_tickets: ", e, exc_type, fname, exc_tb.tb_lineno)
        return False
    
    

    
# x=datetime.today()
# y = x.replace(day=x.day, hour=23, minute=0, second=0, microsecond=0) + timedelta(days=1)
# delta_t=y-x

# secs=delta_t.total_seconds()

# t = Timer(secs, save_index)
# t.start()

# if __name__=="__main__":
#     pass
#     embeddings_service="http://0.0.0.0:5001/get_embeddings"

#     ids = np.load(f'ids_glove.npy')
#     index_faiss=faiss.read_index(f"index_glove")
#     print(index_faiss.is_trained)
#     print(index_faiss.ntotal)
    