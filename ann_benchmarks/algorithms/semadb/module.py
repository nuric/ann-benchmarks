import numpy as np
import requests
import sklearn.preprocessing
import ctypes

from ..base.module import BaseANN

VERSION="2.0.0"
base_url = "http://localhost:8081/v1/collections"
# base_url = "https://semadb.p.rapidapi.com/collections"
headers = {"X-User-Id": "benchmark", "X-Plan-Id": "BASIC"}
# headers = {
# 	"X-RapidAPI-Key": "",
# 	"X-RapidAPI-Host": "semadb.p.rapidapi.com"
# }

class SemaDBRemote(BaseANN):
    def __init__(self, metric, search_size, degree_bound, alpha):
        self.config = {
            "mode": "remote",
            "searchSize": search_size,
            "degreeBound": degree_bound,
            "alpha": alpha,
            "distMetric": metric,
            "version": VERSION,
        }

    def fit(self, X):
        # ---------------------------
        if self.config["distMetric"] == "angular":
            X = sklearn.preprocessing.normalize(X, axis=1, norm="l2")
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        # ---------------------------
        # Create collection
        self.config["embedDim"] = X.shape[1]
        payload = {
            "id": "benchmark",
            "vectorSize": X.shape[1],
            "distanceMetric": "cosine" if self.config["distMetric"] == "angular" else "euclidean",
        }
        res = requests.post(base_url, json=payload, headers=headers)
        if res.status_code == 409:
            print("Benchmark collection already exists, deleting")
            res = requests.delete(base_url + "/benchmark", headers=headers)
            if res.status_code != 200:
                print(res.text)
                raise Exception("Failed to delete benchmark collection")
            else:
                print("Deleted benchmark collection")
            res = requests.post(base_url, json=payload, headers=headers)
        if res.status_code != 200:
            print(res.text)
            raise Exception("Failed to create collection")
        # ---------------------------
        batch_size = 10000
        for i in range(0, X.shape[0], batch_size):
            print("Adding batch", i, "to", i + batch_size)
            data = X[i : i + batch_size]
            points = []
            # randLongStr = "1234567890" * 100
            for j in range(data.shape[0]):
                points.append({"vector": data[j].tolist(), "metadata": {'xid': i + j}})
                # points.append({"vector": data[j].tolist(), "metadata": {'xid': i + j, 'randLongStr': randLongStr}})
            res = requests.post(
                base_url + "/benchmark/points",
                json={
                    "points": points,
                },
                headers=headers,
            )
            if res.status_code != 200:
                print(res.text)
                raise Exception("Failed to add batch")

    def query(self, v, n):
        if self.config["distMetric"] == "angular":
            v = sklearn.preprocessing.normalize(v.reshape(1, -1), axis=1, norm="l2")[0]
        print(v.tolist())
        v.jumparound()
        res = requests.post(
            base_url + "/benchmark/points/search",
            json={
                "vector": v.tolist(),
                "limit": n,
            },
            headers=headers,
        )
        if res.status_code != 200:
            print(res.text)
            raise Exception("Failed to query")
        ids = [p["metadata"]["xid"] for p in res.json()["points"]]
        return ids

    def set_query_arguments(self, n_probe):
        self._n_probe = n_probe

    def get_additional(self):
        return {"additional": "more info"}

    def __str__(self):
        return f"SemaDB({self.config})"

class GoSlice(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("len", ctypes.c_int64),
        ("cap", ctypes.c_int64)
    ]

class SemaDBLocal(BaseANN):
    def __init__(self, metric, configStr):
        self.config = {
            "mode": "local",
            "quantizer": configStr,
            "distMetric": metric,
            "version": VERSION,
        }
        shardpy_path = "../semadb/internal/shardpy/shardpy.so"
        self.shardpy = ctypes.cdll.LoadLibrary(shardpy_path)
        self.shardpy.initShard.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        self.shardpy.fit.argtypes = [GoSlice]
        self.shardpy.query.argtypes = [GoSlice, ctypes.c_int, GoSlice]


    def fit(self, X):
        # ---------------------------
        if self.config["distMetric"] == "angular":
            X = sklearn.preprocessing.normalize(X, axis=1, norm="l2")
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        # ---------------------------
        # Create Shard
        metric = self.config["distMetric"]
        if self.config["distMetric"] == "angular":
            metric = "cosine"
        self.shardpy.initShard(self.config["quantizer"].encode("utf-8"), metric.encode("utf-8"), X.shape[1])
        # ---------------------------
        X = X.flatten()
        # self.shardpy.startProfile()
        self.shardpy.fit(GoSlice(ctypes.cast(X.ctypes.data, ctypes.c_void_p), X.shape[0], X.shape[0]))
        # self.shardpy.stopProfile()
        # import os; os.exit(1)
        self.shardpy.startProfile()

    def query(self, v, n):
        if v.dtype != np.float32:
            v = v.astype(np.float32)
        if self.config["distMetric"] == "angular":
            v = sklearn.preprocessing.normalize(v.reshape(1, -1), axis=1, norm="l2")[0]
        ids = np.zeros(n, dtype=np.uint32)
        self.shardpy.query(GoSlice(ctypes.cast(v.ctypes.data, ctypes.c_void_p), v.shape[0], v.shape[0]), n, GoSlice(ctypes.cast(ids.ctypes.data, ctypes.c_void_p), ids.shape[0], ids.shape[0]))
        return ids
    
    def done(self):
        self.shardpy.stopProfile()

    def __str__(self):
        return f"SemaDB({self.config})"
