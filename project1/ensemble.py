import Levenshtein  # pip install python-Levenshtein
import math
from tqdm import tqdm
import random

class EnsembleCorrector:
    def __init__(self, rule_corrector, stat_corrector, init_weight_rule=0.5, init_weight_stat=0.5, seed=0):
        self.rule_corrector = rule_corrector
        self.stat_corrector = stat_corrector
        self.weight_rule = init_weight_rule
        self.weight_stat = init_weight_stat
        self.seed = seed
        random.seed(seed)  # 设置随机种子

    def train(self, train_data, valid_data=None):
        self.rule_corrector.train(train_data)
        rule_corrected_data = []
        for sample in train_data:
            corrected_text = self.rule_corrector.correct(sample['source'])
            new_sample = sample.copy()
            new_sample['source'] = corrected_text
            rule_corrected_data.append(new_sample)

       #  print(rule_corrected_data)

        self.stat_corrector.train(rule_corrected_data)
        self.update_weights(train_data)

    def update_weights(self, valid_data):
        correct_rule = 0
        correct_stat = 0
        total = len(valid_data)

        for sample in valid_data:
            source = sample['source']
            target = sample['target']

            rule_pred = self.rule_corrector.correct(source)
            stat_pred = self.stat_corrector.correct(source)

            if rule_pred == target:
                correct_rule += 1
            if stat_pred == target:
                correct_stat += 1

        acc_rule = correct_rule / total
        acc_stat = correct_stat / total

        total_acc = acc_rule + acc_stat
        if total_acc > 0:
            self.weight_rule = acc_rule / total_acc
            self.weight_stat = acc_stat / total_acc
        else:
            self.weight_rule = self.weight_stat = 0.5

        print(f"[Weight Update] Rule: {self.weight_rule:.3f}, Stat: {self.weight_stat:.3f}")

    def correct(self, sample):
        if isinstance(sample, dict):
            text = sample['source']
            target = sample.get('target', None)
        else:
            text = sample
            target = None

        rule_out = self.rule_corrector.correct(text)
        stat_out = self.stat_corrector.correct(text)

        if rule_out == stat_out:
            return rule_out

        if target:
            import Levenshtein
            dist_rule = Levenshtein.distance(rule_out, target)
            dist_stat = Levenshtein.distance(stat_out, target)

            if dist_rule < dist_stat:
                return rule_out
            elif dist_stat < dist_rule:
                return stat_out
            else:
                return random.choices([rule_out, stat_out], weights=[self.weight_rule, self.weight_stat])[0]

        return random.choices([rule_out, stat_out], weights=[self.weight_rule, self.weight_stat])[0]

    
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