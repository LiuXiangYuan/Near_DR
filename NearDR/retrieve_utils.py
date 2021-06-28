import faiss
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer


def init_faiss(faiss_omp_num_threads):
    faiss.omp_set_num_threads(faiss_omp_num_threads)


def index_retrieve(index, query_embeddings, topk, batch=None):
    print("Query Num", len(query_embeddings))
    start = timer()
    if batch is None:
        _, nearest_neighbors = index.search(query_embeddings, topk)
    else:
        query_offset_base = 0
        pbar = tqdm(total=len(query_embeddings))
        nearest_neighbors = []
        while query_offset_base < len(query_embeddings):
            batch_query_embeddings = query_embeddings[query_offset_base:query_offset_base + batch]
            batch_nn = index.search(batch_query_embeddings, topk)[1]
            nearest_neighbors.extend(batch_nn.tolist())
            query_offset_base += len(batch_query_embeddings)
            pbar.update(len(batch_query_embeddings))
        pbar.close()

    elapsed_time = timer() - start
    elapsed_time_per_query = 1000 * elapsed_time / len(query_embeddings)
    print(f"Elapsed Time: {elapsed_time:.1f}s, Elapsed Time per query: {elapsed_time_per_query:.1f}ms")
    return nearest_neighbors


def construct_flatindex_from_embeddings(embeddings, ids=None):
    dim = embeddings.shape[1]
    print('embedding shape: ' + str(embeddings.shape))
    index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    if ids is not None:
        assert embeddings.shape[0] == ids.shape[0], \
            'the number of embeddings and ids is different!'

        ids = ids.astype(np.int64)
        print(ids.shape, ids.dtype)
        index = faiss.IndexIDMap2(index)
        index.add_with_ids(embeddings, ids)
    else:
        index.add(embeddings)
    return index


gpu_resources = []


def convert_index_to_gpu(index, faiss_gpu_index, useFloat16=False):
    if type(faiss_gpu_index) == list and len(faiss_gpu_index) == 1:
        faiss_gpu_index = faiss_gpu_index[0]
    if isinstance(faiss_gpu_index, int):
        res = faiss.StandardGpuResources()
        res.setTempMemory(512*1024*1024)
        co = faiss.GpuClonerOptions()
        co.useFloat16 = useFloat16
        index = faiss.index_cpu_to_gpu(res, faiss_gpu_index, index, co)
    else:
        global gpu_resources
        if len(gpu_resources) == 0:
            import torch
            for i in range(torch.cuda.device_count()):
                res = faiss.StandardGpuResources()
                res.setTempMemory(256*1024*1024)
                gpu_resources.append(res)

        assert isinstance(faiss_gpu_index, list)
        vres = faiss.GpuResourcesVector()
        vdev = faiss.IntVector()
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = useFloat16
        for i in faiss_gpu_index:
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])
        index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)

    return index


def load_index(passage_embeddings, index_path, faiss_gpu_index, use_gpu):
    dim = passage_embeddings.shape[1]
    if index_path is None:
        index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
        index.add(passage_embeddings)
    else:
        index = faiss.read_index(index_path)
    if faiss_gpu_index and use_gpu:
        if len(faiss_gpu_index) == 1:
            res = faiss.StandardGpuResources()
            res.setTempMemory(1024 * 1024 * 1024)
            co = faiss.GpuClonerOptions()
            if index_path:
                co.useFloat16 = True
            else:
                co.useFloat16 = False
            index = faiss.index_cpu_to_gpu(res, faiss_gpu_index, index, co)
        else:
            assert not index_path  # Only need one GPU for compressed index
            global gpu_resources
            if len(gpu_resources) == 0:
                import torch
                for i in range(torch.cuda.device_count()):
                    res = faiss.StandardGpuResources()
                    res.setTempMemory(128 * 1024 * 1024)
                    gpu_resources.append(res)

            assert isinstance(faiss_gpu_index, list)
            vres = faiss.GpuResourcesVector()
            vdev = faiss.IntVector()
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            for i in faiss_gpu_index:
                vdev.push_back(i)
                vres.push_back(gpu_resources[i])
            index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)

    return index
