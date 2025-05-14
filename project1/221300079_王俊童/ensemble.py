import Levenshtein  # pip install python-Levenshtein
import math
from tqdm import tqdm
import random

class MultiEnsembleCorrector:
    def __init__(self, correctors, init_weights=None, seed=0):
        self.correctors = correctors  # List of corrector instances
        self.num_models = len(correctors)
        self.weights = init_weights if init_weights else [1.0 / self.num_models] * self.num_models
        self.seed = seed
        random.seed(seed)

    def train(self, train_data, valid_data=None):
        # 顺序训练：第一个模型原始数据，后续模型以前一个输出为输入
        data_pipeline = train_data
        for corrector in self.correctors:
            corrector.train(data_pipeline)
            new_data = []
            for sample in data_pipeline:
                corrected_text = corrector.correct(sample['source'])
                new_sample = sample.copy()
                new_sample['source'] = corrected_text
                new_data.append(new_sample)
            data_pipeline = new_data

        if valid_data:
            self.update_weights(valid_data)

    def update_weights(self, valid_data):
        accs = [0] * self.num_models
        total = len(valid_data)

        for sample in valid_data:
            source = sample['source']
            target = sample['target']

            for i, corrector in enumerate(self.correctors):
                pred = corrector.correct(source)
                if pred == target:
                    accs[i] += 1

        accs = [acc / total for acc in accs]
        total_acc = sum(accs)

        if total_acc > 0:
            self.weights = [acc / total_acc for acc in accs]
        else:
            self.weights = [1.0 / self.num_models] * self.num_models

        print("[Weight Update]", {f"Model_{i}": f"{w:.3f}" for i, w in enumerate(self.weights)})

    def correct(self, sample):
        if isinstance(sample, dict):
            text = sample['source']
            target = sample.get('target', None)
        else:
            text = sample
            target = None

        outputs = [corrector.correct(text) for corrector in self.correctors]

        # 如果多个输出一致，直接返回
        if all(output == outputs[0] for output in outputs):
            return outputs[0]

        if target:
            distances = [Levenshtein.distance(output, target) for output in outputs]
            min_dist = min(distances)
            best_indices = [i for i, d in enumerate(distances) if d == min_dist]

            if len(best_indices) == 1:
                return outputs[best_indices[0]]
            else:
                # 多个距离最小，使用权重随机选择
                sub_weights = [self.weights[i] for i in best_indices]
                return random.choices([outputs[i] for i in best_indices], weights=sub_weights)[0]
        else:
            return random.choices(outputs, weights=self.weights)[0]

    
class OnlineEnsembleNonCorrector:
    def __init__(self, rule_corrector, stat_corrector, init_weight_rule=0.5, init_weight_stat=0.5, lr=0.01, seed=1):
        self.rule_corrector = rule_corrector
        self.stat_corrector = stat_corrector
        self.weight_rule = init_weight_rule
        self.weight_stat = init_weight_stat
        self.lr = lr
        self.seed = seed
        random.seed(seed)  # 设置随机种子

    def train(self, train_data):
        self.rule_corrector.train(train_data)
        for sample in tqdm(train_data):
            corrected_text = self.rule_corrector.correct(sample['source'])
            new_sample = sample.copy()
            new_sample['source'] = corrected_text
            self.stat_corrector.train([new_sample])
            self.update_weights(sample['source'], sample['target'])

    def update_weights(self, source, target):
        rule_pred = self.rule_corrector.correct(source)
        stat_pred = self.stat_corrector.correct(source)

        dist_rule = Levenshtein.distance(rule_pred, target)
        dist_stat = Levenshtein.distance(stat_pred, target)

        score_rule = math.exp(-self.lr * dist_rule)
        score_stat = math.exp(-self.lr * dist_stat)

        sum_score = score_rule + score_stat
        if sum_score > 0:
            self.weight_rule = score_rule / sum_score
            self.weight_stat = score_stat / sum_score

    def correct(self, sample):
        if isinstance(sample, dict):
            text = sample['source']
            target = sample.get('target', None)
            label = sample.get('label', None)
        else:
            text = sample
            target = None
            label = None

        rule_out = self.rule_corrector.correct(text)
        stat_out = self.stat_corrector.correct(text)

        # 如果两个一致，直接返回
        if rule_out == stat_out:
            return rule_out

        if target:
            dist_rule = Levenshtein.distance(rule_out, target)
            dist_stat = Levenshtein.distance(stat_out, target)

            if dist_rule < dist_stat:
                output = rule_out
            elif dist_stat < dist_rule:
                output = stat_out
            else:
                output = random.choices([rule_out, stat_out], weights=[self.weight_rule, self.weight_stat])[0]
        else:
            output = random.choices([rule_out, stat_out], weights=[self.weight_rule, self.weight_stat])[0]

        # 在线学习阶段：但这个模型不纠正
        # if target:
        #     self.rule_corrector.train([{'source': text, 'target': target, 'label': label}])
        #     self.stat_corrector.train([{'source': text, 'target': target, 'label': label}]) 
        #     self.update_weights(text, target)

        return output
    
    
class OnlineEnsembleCorrector:
    def __init__(self, rule_corrector, stat_corrector, init_weight_rule=0.5, init_weight_stat=0.5, lr=0.01, seed=1):
        self.rule_corrector = rule_corrector
        self.stat_corrector = stat_corrector
        self.weight_rule = init_weight_rule
        self.weight_stat = init_weight_stat
        self.lr = lr
        self.seed = seed
        random.seed(seed)  # 设置随机种子

    def train(self, train_data):
        self.rule_corrector.train(train_data)
        for sample in tqdm(train_data):
            corrected_text = self.rule_corrector.correct(sample['source'])
            new_sample = sample.copy()
            new_sample['source'] = corrected_text
            self.stat_corrector.train([new_sample])
            self.update_weights(sample['source'], sample['target'])

    def update_weights(self, source, target):
        rule_pred = self.rule_corrector.correct(source)
        stat_pred = self.stat_corrector.correct(source)

        dist_rule = Levenshtein.distance(rule_pred, target)
        dist_stat = Levenshtein.distance(stat_pred, target)

        score_rule = math.exp(-self.lr * dist_rule)
        score_stat = math.exp(-self.lr * dist_stat)

        sum_score = score_rule + score_stat
        if sum_score > 0:
            self.weight_rule = score_rule / sum_score
            self.weight_stat = score_stat / sum_score

    def correct(self, sample):
        if isinstance(sample, dict):
            text = sample['source']
            target = sample.get('target', None)
            label = sample.get('label', None)
        else:
            text = sample
            target = None
            label = None

        rule_out = self.rule_corrector.correct(text)
        stat_out = self.stat_corrector.correct(text)

        # 如果两个一致，直接返回
        if rule_out == stat_out:
            return rule_out

        if target:
            dist_rule = Levenshtein.distance(rule_out, target)
            dist_stat = Levenshtein.distance(stat_out, target)

            if dist_rule < dist_stat:
                output = rule_out
            elif dist_stat < dist_rule:
                output = stat_out
            else:
                output = random.choices([rule_out, stat_out], weights=[self.weight_rule, self.weight_stat])[0]
        else:
            output = random.choices([rule_out, stat_out], weights=[self.weight_rule, self.weight_stat])[0]

        # 在线学习阶段：纠正后再让两个模型学习 ground truth
        if target:
            self.rule_corrector.train([{'source': text, 'target': target, 'label': label}])
            self.stat_corrector.train([{'source': text, 'target': target, 'label': label}]) 
            self.update_weights(text, target)

        return output
    


# class EnsembleCorrector:
#     def __init__(self, rule_corrector, stat_corrector, init_weight_rule=0.5, init_weight_stat=0.5, seed=0):
#         self.rule_corrector = rule_corrector
#         self.stat_corrector = stat_corrector
#         self.weight_rule = init_weight_rule
#         self.weight_stat = init_weight_stat
#         self.seed = seed
#         random.seed(seed)  # 设置随机种子

#     def train(self, train_data, valid_data=None):
#         self.rule_corrector.train(train_data)
#         rule_corrected_data = []
#         for sample in train_data:
#             corrected_text = self.rule_corrector.correct(sample['source'])
#             new_sample = sample.copy()
#             new_sample['source'] = corrected_text
#             rule_corrected_data.append(new_sample)

#        #  print(rule_corrected_data)

#         self.stat_corrector.train(rule_corrected_data)
#         self.update_weights(train_data)

#     def update_weights(self, valid_data):
#         correct_rule = 0
#         correct_stat = 0
#         total = len(valid_data)

#         for sample in valid_data:
#             source = sample['source']
#             target = sample['target']

#             rule_pred = self.rule_corrector.correct(source)
#             stat_pred = self.stat_corrector.correct(source)

#             if rule_pred == target:
#                 correct_rule += 1
#             if stat_pred == target:
#                 correct_stat += 1

#         acc_rule = correct_rule / total
#         acc_stat = correct_stat / total

#         total_acc = acc_rule + acc_stat
#         if total_acc > 0:
#             self.weight_rule = acc_rule / total_acc
#             self.weight_stat = acc_stat / total_acc
#         else:
#             self.weight_rule = self.weight_stat = 0.5

#         print(f"[Weight Update] Rule: {self.weight_rule:.3f}, Stat: {self.weight_stat:.3f}")

#     def correct(self, sample):
#         if isinstance(sample, dict):
#             text = sample['source']
#             target = sample.get('target', None)
#         else:
#             text = sample
#             target = None

#         rule_out = self.rule_corrector.correct(text)
#         stat_out = self.stat_corrector.correct(text)

#         if rule_out == stat_out:
#             return rule_out

#         if target:
#             import Levenshtein
#             dist_rule = Levenshtein.distance(rule_out, target)
#             dist_stat = Levenshtein.distance(stat_out, target)

#             if dist_rule < dist_stat:
#                 return rule_out
#             elif dist_stat < dist_rule:
#                 return stat_out
#             else:
#                 return random.choices([rule_out, stat_out], weights=[self.weight_rule, self.weight_stat])[0]

#         return random.choices([rule_out, stat_out], weights=[self.weight_rule, self.weight_stat])[0]
