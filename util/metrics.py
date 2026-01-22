import math
import numpy as np
import random

class RecommendMetric(object):
    def __init__(self):
        pass

    @staticmethod
    def hits(origin, res):
        hit_count = {}
        for user in origin:
            items = list(origin[user].keys())
            predicted = [item[0] for item in res[user]]
            hit_count[user] = len(set(items).intersection(set(predicted)))
        return hit_count

    @staticmethod
    def hit_ratio(origin, hits):
        """
        Note: This type of hit ratio calculates the fraction:
         (# retrieved interactions in the test set / #all the interactions in the test set)
        """
        total_num = 0
        for user in origin:
            items = list(origin[user].keys())
            total_num += len(items)
        hit_num = 0
        for user in hits:
            hit_num += hits[user]
        return hit_num/total_num

    @staticmethod
    def hit_ratio_candidate(origin, model, data, N):
        """
        Note: Global Scoring Hit Ratio (Leave-One-Out).
        Calculates the average hit rate across all users in the test set.
        For each user: 1 positive (from test) ranked against all items except history.
        """
        hit_counts = {n: 0 for n in N}
        total_users = 0
        
        for user_name in origin:
            if user_name not in data.user:
                continue
            
            # 1. Positive item (Leave-One-Out: taking the first item in test set)
            test_items = list(origin[user_name].keys())
            if not test_items:
                continue
            pos_item_name = test_items[0]
            if pos_item_name not in data.item:
                continue
            pos_item_id = data.item[pos_item_name]
            
            # 2. History to exclude (Train + Val + Test)
            history = set([data.item[i] for i in data.training_set_u[user_name] if i in data.item])
            if user_name in data.val_set:
                history.update([data.item[i] for i in data.val_set[user_name] if i in data.item])
            history.update([data.item[i] for i in origin[user_name] if i in data.item])
            
            # Masking: remove the positive item from history to keep it in candidate pool
            history.discard(pos_item_id)
            
            # 3. Global Scoring
            # 讓模型預測該用戶對「所有物品」的分數
            scores = model.predict(user_name)
            
            # 4. History Masking: set history items' scores to -inf
            scores[list(history)] = -np.inf
            
            # 5. Rank (higher score is better)
            pos_score = scores[pos_item_id]
            # rank = 1 + number of items with score > pos_score
            rank = (scores > pos_score).sum() + 1
            
            # 更新命中計數
            total_users += 1
            for n in N:
                if rank <= n:
                    hit_counts[n] += 1
        
        results = {n: (hit_counts[n] / total_users if total_users > 0 else 0) for n in N}
        return results

    @staticmethod
    def precision(hits, N):
        prec = sum([hits[user] for user in hits])
        return prec / (len(hits) * N)

    @staticmethod
    def recall(hits, origin):
        recall_list = [hits[user]/len(origin[user]) for user in hits]
        recall = sum(recall_list) / len(recall_list)
        return recall

    @staticmethod
    def F1(prec, recall):
        if (prec + recall) != 0:
            return 2 * prec * recall / (prec + recall)
        else:
            return 0

    @staticmethod
    def MAE(res):
        error = 0
        count = 0
        for entry in res:
            error+=abs(entry[2]-entry[3])
            count+=1
        if count==0:
            return error
        return error/count

    @staticmethod
    def RMSE(res):
        error = 0
        count = 0
        for entry in res:
            error += (entry[2] - entry[3])**2
            count += 1
        if count==0:
            return error
        return math.sqrt(error/count)

    @staticmethod
    def NDCG(origin,res,N):
        sum_NDCG = 0
        for user in res:
            DCG = 0
            IDCG = 0
            #1 = related, 0 = unrelated
            for n, item in enumerate(res[user]):
                if item[0] in origin[user]:
                    DCG+= 1.0/math.log(n+2)
            for n, item in enumerate(list(origin[user].keys())[:N]):
                IDCG+=1.0/math.log(n+2)
            sum_NDCG += DCG / IDCG
        return sum_NDCG / len(res)

def ranking_evaluation(origin, res, N):
    measure = []
    for n in N:
        predicted = {}
        for user in res:
            predicted[user] = res[user][:n]
        indicators = []
        if len(origin) != len(predicted):
            print('The Lengths of test set and predicted set do not match!')
            exit(-1)
        hits = RecommendMetric.hits(origin, predicted)
        hr = RecommendMetric.hit_ratio(origin, hits)
        indicators.append('Hit Ratio:' + str(hr) + '\n')
        prec = RecommendMetric.precision(hits, n)
        indicators.append('Precision:' + str(prec) + '\n')
        recall = RecommendMetric.recall(hits, origin)
        indicators.append('Recall:' + str(recall) + '\n')
        # F1 = Metric.F1(prec, recall)
        # indicators.append('F1:' + str(F1) + '\n')
        #MAP = Measure.MAP(origin, predicted, n)
        #indicators.append('MAP:' + str(MAP) + '\n')
        NDCG = RecommendMetric.NDCG(origin, predicted, n)
        indicators.append('NDCG:' + str(NDCG) + '\n')
        # AUC = Measure.AUC(origin,res,rawRes)
        # measure.append('AUC:' + str(AUC) + '\n')
        measure.append('Top ' + str(n) + '\n')
        measure += indicators
    return measure

def rating_evaluation(res):
    measure = []
    mae = RecommendMetric.MAE(res)
    measure.append('MAE:' + str(mae) + '\n')
    rmse = RecommendMetric.RMSE(res)
    measure.append('RMSE:' + str(rmse) + '\n')
    return measure


class AttackMetric(object):
    """
    param 
    targetItem:list, targetItem: id
    """
    def __init__(self, recommendModel, targetItem, top=[10]):
        self.recommendModel = recommendModel
        self.targetItem = targetItem
        self.top = top

    def precision(self):
        totalNum = [0 for i in range(len(self.top))]
        hit = [0 for i in range(len(self.top))]
        for i in self.recommendModel.data.user:
            score = self.recommendModel.predict(i)
            result = []
            for n, k in enumerate(self.top):
                result.append(np.argsort(-score)[:k])
                totalNum[n] += k
            for j in self.targetItem:
                for k in range(len(self.top)):
                    if j in result[k]:
                        hit[k] += 1
        result = []
        for i in range(len(self.top)):
            result.append(hit[i] / totalNum[i])
        return result

    def hitRate(self):
        totalNum = [0 for i in range(len(self.top))]
        hit = [0 for i in range(len(self.top))]
        for i in self.recommendModel.data.user:
            score = self.recommendModel.predict(i)
            result = []
            for n, k in enumerate(self.top):
                result.append(np.argsort(-score)[:k])
                totalNum[n] += 1
            for k in range(len(self.top)):
                hit[k] += int(len(set(self.targetItem) & set(result[k])) > 0)/len(self.targetItem)
        result = []
        for i in range(len(self.top)):
            result.append(hit[i] / totalNum[i])
        return result

    def recall(self):
        totalNum = [0 for i in range(len(self.top))]
        hit = [0 for i in range(len(self.top))]
        for i in self.recommendModel.data.user:
            score = self.recommendModel.predict(i)
            result = []
            for n, k in enumerate(self.top):
                result.append(np.argsort(-score)[:k])
                totalNum[n] += len(self.targetItem)
            for j in self.targetItem:
                for k in range(len(self.top)):
                    if j in result[k]:
                        hit[k] += 1
        result = []
        for i in range(len(self.top)):
            result.append(hit[i] / totalNum[i])
        return result

    def NDCG(self):
        totalNum = [0 for i in range(len(self.top))]
        hit = [0 for i in range(len(self.top))]
        for i in self.recommendModel.data.user:
            score = self.recommendModel.predict(i)
            result = []
            for n, k in enumerate(self.top):
                result.append(np.argsort(-score)[:k])
                idcg=0
                for s in range(k):
                    if s < len(self.targetItem):
                        idcg+= 1 / np.log2(2 + s)
                totalNum[n] += idcg
            for step,r in enumerate(result):
                for rank,j in enumerate(r):
                    if j in self.targetItem:
                        hit[step]+=1 / np.log2(2 + rank)
        result = []
        for i in range(len(self.top)):
            result.append(hit[i] / totalNum[i])
        return result

    def exposure_ratio_candidate(self):
        """
        Calculates Exposure Ratio (ER) using global ranking.
        """
        hit_counts = {n: 0 for n in self.top}
        total_users = 0

        used = 0
        skipped = 0
        data = self.recommendModel.data
        all_item_ids = list(data.item.values())

        # ✅ 固定攻擊目標（internal item id）
        assert isinstance(self.targetItem, (list, tuple)) and len(self.targetItem) > 0
        target_item_id = int(self.targetItem[0])


        for user_name in data.test_set: # Evaluate on test set users to have consistent history exclusion
            if user_name not in data.user:
                continue

            # ===== 1. build full interaction history =====
            history = set(
                data.item[i]
                for i in data.training_set_u.get(user_name, [])
                if i in data.item
            )

            if user_name in data.val_set:
                history.update(
                    data.item[i]
                    for i in data.val_set[user_name]
                    if i in data.item
                )

            history.update(
                data.item[i]
                for i in data.test_set[user_name]
                if i in data.item
            )

            # ===== 2. ONLY evaluate users who NEVER interacted with target =====
            if target_item_id in history:
                skipped += 1
                continue

            used += 1

            # ===== 3. Global Scoring =====
            scores = self.recommendModel.predict(user_name)

            # ===== 4. History Masking: set history items' scores to -inf =====
            # Note: target_item_id is already guaranteed NOT in history
            scores[list(history)] = -np.inf
            
            # Rank
            target_score = scores[target_item_id]
            rank = (scores > target_score).sum() + 1
            
            total_users += 1
            for n in self.top:
                if rank <= n:
                    hit_counts[n] += 1

        print(f"[ER] used={used}, skipped={skipped}, total_test_users={len(data.test_set)}")        
        results = [hit_counts[n] / total_users if total_users > 0 else 0 for n in self.top]
        return results