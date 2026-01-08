import faiss
import numpy as np
import math
from collections import defaultdict
import torch

# --- [NEW] 사용자 제공 Metric 함수 (Pure Python) ---
def f1(prec, recall):
    if (prec + recall) != 0:
        return 2 * prec * recall / (prec + recall)
    else:
        return 0

def computeTopNAccuracy(GroundTruth, predictedIndices, topN):
    precision = [] 
    recall = [] 
    NDCG = [] 
    MRR = []
    HR = []
    F1 = []
    
    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        sumForHR = 0
        for i in range(len(predictedIndices)):
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit=False
                
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0/math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0/(j+1.0))
                            mrrFlag = False
                        userHit += 1
                        hit=True
                
                    if idcgCount > 0:
                        idcg += 1.0/math.log2(j + 2)
                        idcgCount = idcgCount-1
                    
                            
                if(idcg != 0):
                    ndcg += (dcg/idcg)
                    
                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])               
                sumForNdcg += ndcg
                sumForMRR += userMRR

                if hit:
                    sumForHR += 1  # Increment sumForHR if there was a hit
        
        precision.append(round(sumForPrecision / len(predictedIndices), 8))
        recall.append(round(sumForRecall / len(predictedIndices), 8))
        NDCG.append(round(sumForNdcg / len(predictedIndices), 8))
        MRR.append(round(sumForMRR / len(predictedIndices), 8))
        HR.append(round(sumForHR / len(predictedIndices), 8))
        F1.append(round(f1(precision[index], recall[index]), 8))
        
    return precision, recall, NDCG, MRR, HR, F1
# ----------------------------------------------------

def num_faiss_evaluate(_test_ratings, _train_ratings, _topk_list, _user_matrix, _item_matrix, _test_users):
    '''
    GBSR Reproduction Evaluation
    '''
    # 1. Faiss Search
    query_vectors = _user_matrix
    dim = _user_matrix.shape[-1]
    index = faiss.IndexFlatIP(dim)
    index.add(_item_matrix)
    
    # ---------------------------------------------------------
    # [핵심 수정] Train Data 구조 정규화 (Dictionary로 통일)
    # 리스트여도, 딕셔너리(str key)여도 모두 {int_id: set(items)} 형태로 변환
    # ---------------------------------------------------------
    normalized_train = {}
    
    if isinstance(_train_ratings, list):
        # 리스트인 경우 인덱스가 유저 ID
        for u_id, items in enumerate(_train_ratings):
            if items is not None:
                normalized_train[u_id] = set(items)
    elif isinstance(_train_ratings, dict):
        # 딕셔너리인 경우 (key가 str일수도, int일수도 있음)
        for u_id, items in _train_ratings.items():
            if items is not None:
                try:
                    normalized_train[int(u_id)] = set(items)
                except:
                    pass
    
    # 마스킹 길이 계산 (검색 범위 설정을 위해)
    if normalized_train:
        max_mask_items_length = max(len(v) for v in normalized_train.values())
    else:
        max_mask_items_length = 0

    k_max = _topk_list[-1]
    search_k = k_max + max_mask_items_length
    
    # 전체 검색 수행
    sim, pred_items_all = index.search(query_vectors, search_k)

    # 2. 평가 수행
    GroundTruth = []
    predictedIndices = []
    
    # Test Data도 정규화하여 접근
    is_test_dict = isinstance(_test_ratings, dict)

    for idx, u in enumerate(_test_users):
        u = int(u) # 유저 ID 정수형 보장
        
        # (1) Ground Truth 가져오기
        gt_items = set()
        if is_test_dict:
            # 키 타입 체크 (int로 먼저 시도, 안되면 str로 시도)
            if u in _test_ratings:
                gt_items = set(_test_ratings[u])
            elif str(u) in _test_ratings:
                gt_items = set(_test_ratings[str(u)])
        else:
            if u < len(_test_ratings) and _test_ratings[u] is not None:
                gt_items = set(_test_ratings[u])
        
        GroundTruth.append(gt_items)

        # (2) Train Masking (정규화된 맵 사용 -> 데이터 유실 방지)
        train_items = normalized_train.get(u, set())

        # [디버깅] 첫 5명에 대해 마스킹 개수 출력 (이게 20~30개 이상 나와야 정상)
        if idx < 5:
            print(f"[DEBUG] User {u}: GT Size={len(gt_items)}, Masking Size={len(train_items)}")

        # Prediction 필터링
        preds = []
        # 반드시 유저 ID(u)로 Faiss 결과 인덱싱
        if u < len(pred_items_all):
            for item in pred_items_all[u]: 
                if item not in train_items:
                    preds.append(item)
                if len(preds) >= k_max:
                    break
        else:
            # 혹시 모를 인덱스 에러 방지 (더미)
            preds = []
            
        predictedIndices.append(preds)

    # 3. Metric 계산 (사용자 제공 함수)
    precision, recall, NDCG, MRR, HR, F1 = computeTopNAccuracy(GroundTruth, predictedIndices, _topk_list)

    hr_out, recall_out, ndcg_out, precision_out, mrr_out = {}, {}, {}, {}, {}
    for i, k in enumerate(_topk_list):
        precision_out[k] = precision[i]
        recall_out[k] = recall[i]
        ndcg_out[k] = NDCG[i]
        mrr_out[k] = MRR[i]
        hr_out[k] = HR[i]

    return hr_out, recall_out, ndcg_out, precision_out, mrr_out