#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rule-based corrector for Chinese Text Correction task.
This module implements rule-based methods for correcting errors in Chinese text.
"""

import re
from typing import Dict, List, Any
from collections import defaultdict
import jieba.posseg as pseg
# Try to import optional dependencies
try:
    import jieba

    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    print("Warning: jieba not available. Some features will be disabled.")


class RuleBasedCorrector:
    """
    A rule-based corrector for Chinese text.
    """

    def __init__(self):
        """
        Initialize the rule-based corrector.
        """
        # Punctuation rules
        self.punctuation_rules = defaultdict(lambda: defaultdict(int))

        # Grammar rules
        self.grammar_rules = defaultdict(lambda: defaultdict(int))

        # or else
        self.confusion_pairs = defaultdict(lambda: defaultdict(int))
        

    def train(self, train_data: List[Dict[str, Any]]) -> None:
        """
        Extract rules from the training data.

        Args:
            train_data: List of dictionaries containing the training data.
        """
        # TODO 完成规则方法的实现，可以参考如下的方法，或者自行设计
        self._extract_confusion_pairs(train_data)
        self._extract_punctuation_rules(train_data)
        self._extract_grammar_rules(train_data)
        self._extract_word_confusion(train_data)

    def _extract_confusion_pairs(self, train_data: List[Dict[str, Any]], min_count: int = 3) -> None:
        """
        Extract character confusion pairs from the training data.

        Args:
            train_data: List of dictionaries containing the training data.
        """
        # Extract character-level confusion pairs from error examples
        for sample in train_data:
            if sample['label'] == 1:  # Only for sentences with errors
                source = sample['source']
                target = sample['target']

                # For character substitution errors (when lengths are equal)
                if len(source) == len(target):
                    for i, (s_char, t_char) in enumerate(zip(source, target)):
                        if s_char != t_char:
                            # Get context (surrounding characters)
                            # left_context = source[max(0, i - 2) : i]
                            # right_context = source[i + 2 : min(len(source), i + 5)]
                            # context = left_context + '_' + right_context
                            # 尝试一个字
                            # context = (source[i - 1] if i > 0 else '', source[i + 1] if i < len(source) - 1 else '')
                            if s_char not in self.confusion_pairs:
                                self.confusion_pairs[s_char] = {}

                            if t_char not in self.confusion_pairs[s_char]:
                                self.confusion_pairs[s_char][t_char] = 0
                            self.confusion_pairs[s_char][t_char] += 1
                            # self.confusion_pairs[s_char][(t_char, context)] += 1

                # elif len(source) > len(target):
                #     # 如果源文本比目标文本长，假定目标文本中有插入错误，我们会记录源文本中的字符
                #     for i in range(len(target), len(source)):
                #         # Record source character as inserted (in target)
                #         s_char = source[i]
                #         self.confusion_pairs[s_char]['<insert>'] += 1
                # else:
                #     # 如果目标文本比源文本长，假定源文本中有删除错误，我们会记录目标文本中的字符
                #     for i in range(len(source), len(target)):
                #         # Record target character as deleted (in source)
                #         t_char = target[i]
                #         self.confusion_pairs['<delete>'][t_char] += 1



        # Filter confusion pairs to keep only the most common ones
        # print(self.confusion_pairs)
        filtered_pairs = {}
        for wrong_char, corrections in self.confusion_pairs.items():
            common_corrections = {correct: count for correct, count in corrections.items() if count >= min_count}
            if common_corrections:
                filtered_pairs[wrong_char] = common_corrections

        self.confusion_pairs = filtered_pairs

    def _extract_punctuation_rules(self, train_data: List[Dict[str, Any]]) -> None:
        """
        Extract punctuation correction rules from the training data.

        Args:
            train_data: List of dictionaries containing the training data.
        """
        punctuation_set = set("，。！？；：“”‘’()（）…’‘”“")
        for sample in train_data:
            if sample['label'] == 1:
                source = sample["source"]
                target = sample["target"]

                # scene 找出每一句的标点符号，看错误的对应是啥。
                source_punct = {i: char for i, char in enumerate(source) if char in punctuation_set}
                target_punct = {i: char for i, char in enumerate(target) if char in punctuation_set}
                # print(source_punct)
                # print(target_punct)
                # print('')
                mismatch = {idx: (source_punct[idx], target_punct[idx]) 
                            for idx in source_punct.keys() & target_punct.keys()  # 只对比共同的 key（索引）
                            if source_punct[idx] != target_punct[idx]}
                for idx, (s_punc, t_punc) in mismatch.items():
                    self.punctuation_rules[s_punc][t_punc] += 1 

        # print(self.punctuation_rules)
                

    def _extract_grammar_rules(self, train_data: List[Dict[str, Any]]) -> None:
        """
        Extract grammar correction rules from the training data.

        Args:
            train_data: List of dictionaries containing the training data.
        """
        total_samples = len(train_data)  # 计算训练集总句子数
        rule_counts = {
            "pos_replace": defaultdict(int),
            "word_replace": defaultdict(int)
        }

        for sample in train_data:
            if sample["label"] == 1:  # 仅处理有错误的句子
                source = sample["source"]
                target = sample["target"]

                # 词性分析
                source_pos = [(word, flag) for word, flag in pseg.cut(source)]
                target_pos = [(word, flag) for word, flag in pseg.cut(target)]

                # 计算词性变化
                temp_changes = []
                min_len = min(len(source_pos), len(target_pos))

                for i in range(min_len):
                    src_word, src_pos = source_pos[i]
                    tgt_word, tgt_pos = target_pos[i]

                    if src_word != tgt_word:
                        temp_changes.append((src_word, src_pos, tgt_word, tgt_pos))

                # 统计错误模式
                for src_word, src_pos, tgt_word, tgt_pos in temp_changes:
                    rule_counts["pos_replace"][(src_pos, tgt_pos)] += 1
                    rule_counts["word_replace"][(src_word, tgt_word)] += 1

        # 计算置信度，并过滤低置信度规则
        confidence_threshold = 0.1  # 置信度阈值, 连0.1都达不到，玩毛线，训练集太小了
        self.grammar_rules = {
            "pos_replace": {
                change: count / total_samples
                for change, count in rule_counts["pos_replace"].items()
                if count / total_samples >= confidence_threshold
            },
            "word_replace": {
                change: count / total_samples
                for change, count in rule_counts["word_replace"].items()
                if count / total_samples >= confidence_threshold
            }
        }

        # print(self.grammar_rules)


    def _extract_word_confusion(self, train_data: List[Dict[str, Any]]) -> None:
        """
        Extract word-level confusion pairs from the training data.

        Args:
            train_data: List of dictionaries containing the training data.
        """
        confusion_pairs = defaultdict(lambda: defaultdict(int))
        src_word_counts = defaultdict(int)
        
        for sample in train_data:
            if sample["label"] != 1:
                continue
                
            # 获取分词和词性标注结果
            src_words = [word for word, _ in pseg.cut(sample["source"])]
            tgt_words = [word for word, _ in pseg.cut(sample["target"])]
            
            # 对齐找出差异点
            for src_word, tgt_word in zip(src_words, tgt_words):
                if src_word != tgt_word:
                    # 记录混淆对
                    confusion_pairs[src_word][tgt_word] += 1
                    src_word_counts[src_word] += 1
        
        # 计算置信度并过滤
        self.word_confusion_rules = {}
        for src_word, targets in confusion_pairs.items():
            total = src_word_counts[src_word]
            confident_targets = {
                tgt_word: count/total
                for tgt_word, count in targets.items()
                if count/total >= 0.9 and count >= 3  
            }
            if confident_targets:
                # 取置信度最高的目标词
                best_target = max(confident_targets.items(), key=lambda x: x[1])
                self.word_confusion_rules[src_word] = best_target
        
        filtered_rules = {
            src: (tgt, conf) 
            for src, (tgt, conf) in self.word_confusion_rules.items()
            if not (
                (len(src) == 2 and len(tgt) == 2 and src[0] != tgt[0])  # 情况1
                or (len(src) != len(tgt))  # 情况2
            )
        }

        # TODO() 可以选择查看
        # print("过滤后的规则（排除长度=2且首字不同的词对）：")
        # for src, (tgt, conf) in list(filtered_rules.items())[:10]:
        #     print(f"'{src}' → '{tgt}' (置信度: {conf:.2%})")

        # 可选：更新原始规则字典
        self.word_confusion_rules = filtered_rules

    


###################################################################################################################################################################################################

    def correct(self, text: str) -> str:
        """
        Apply rule-based correction to the input text.

        Args:
            text: Input text to correct.

        Returns:
            Corrected text.
        """
        # Apply different correction rules in sequence
        # TODO 对应规则方法的实现，完成修正部分（可以参考如下的方法，或者自行设计）
        corrected = self._correct_punctuation(text)
        corrected = self._correct_confusion_chars(corrected)
        corrected = self._correct_grammar(corrected)
        corrected = self._correct_word_confusion(corrected)
        return corrected

    def _correct_punctuation(self, text: str) -> str:
        """
        Correct punctuation errors in the text.

        Args:
            text: Input text.

        Returns:
            Text with corrected punctuation.
        """
        if not self.punctuation_rules:
            return text
        
        # rule 1: 考虑需要成对出现的符号
        paired_punct = {'“': '”', '‘': '’', '（': '）', '【': '】'}
        
        text_list = list(text)
        stack = {}  #
        
        for i, char in enumerate(text_list):
            if char in paired_punct:  # 左符号
                stack[char] = stack.get(char, []) + [i]  # 记录索引
            elif char in paired_punct.values():  # 右符号
                left_symbol = [k for k, v in paired_punct.items() if v == char]
                if left_symbol and left_symbol[0] in stack and stack[left_symbol[0]]:
                    stack[left_symbol[0]].pop()  # 匹配成功，弹出栈
                else:
                    # 右符号没有左符号，可能是误用
                    text_list[i] = ''  
         # 处理缺失的右符号
        for left_symbol, indices in stack.items():
            right_symbol = paired_punct[left_symbol]
            for idx in indices:
                text_list.insert(idx + 1, right_symbol)  # 补全右符号
        
        # 调整引号和句号的顺序，比如 '”。' -> '。”'
        text = "".join(text_list)
        text = re.sub(r'([。！？])([”’])', r'\2\1', text)
        return text

    def _correct_confusion_chars(self, text: str) -> str:
        """
        Correct character confusion errors in the text.

        Args:
            text: Input text.

        Returns:
            Text with corrected characters.
        """
        corrected_text = list(text)  # Convert to list for character-by-character editing

        # Check each character for potential confusion
        for i, char in enumerate(text):
            if char in self.confusion_pairs and self.confusion_pairs[char]:
                # Get the most common correction for this character
                correct_char = max(self.confusion_pairs[char].items(), key=lambda x: x[1])[0]

                # Apply some heuristics to decide whether to correct
                # For example, check if the correction makes sense in this context
                # This is a simplified approach; in a real system, more context would be considered

                # For the common confusion of 的/地/得, apply specific rules
                if char == '的' and correct_char in ['地', '得']:
                    # '地' typically follows an adjective and precedes a verb
                    if i > 0 and i < len(text) - 1 and text[i + 1] not in ',.?!，。？！、；：':
                        # Simple check: if followed by a verb-like character, might be '地'
                        if text[i + 1] in '走跑跳跃飞奔跑跳跃飞奔跑跳跃飞奔':
                            corrected_text[i] = '地'

                    # '得' typically follows a verb and precedes an adjective or adverb
                    if i > 0 and i < len(text) - 1 and text[i - 1] not in ',.?!，。？！、；：':
                        # Simple check: if preceded by a verb-like character, might be '得'
                        if text[i - 1] in '说写跑跳走看听闻感觉':
                            corrected_text[i] = '得'

                # For other confusions, apply a simpler rule
                elif char in ['在', '再'] and correct_char in ['再', '在']:
                    # '在' typically indicates location, '再' typically indicates repetition or future action
                    if i < len(text) - 1 and text[i + 1] in '次遍回':
                        corrected_text[i] = '再'
                    elif i > 0 and text[i - 1] in '正将':
                        corrected_text[i] = '在'

                # For other cases, only correct if we're very confident
                # This is a placeholder for more sophisticated rules
                elif self.confusion_pairs[char][correct_char] > 3:  # Arbitrary threshold
                    corrected_text[i] = correct_char

                # TODO more rules
                # 尝试加了一些规则，不升反降。
                # if char == '<insert>':  # 如果是插入错误，考虑删除字符
                #     corrected_text[i] = ''  # 删除多余的插入字符

                # elif char == '<delete>':  # 如果是删除错误，考虑恢复字符
                #     corrected_text[i] = correct_char  # 恢复为最常见的字符

                # 去除重复词
                if i < len(text) - 1 and text[i] == text[i + 1]:
                    corrected_text[i] = ''  # 删除重复的字符

                if i < len(corrected_text) - 3:
                    current_2gram = ''.join(corrected_text[i:i+2])
                    next_2gram = ''.join(corrected_text[i+2:i+4])
                    if current_2gram == next_2gram:
                        corrected_text[i+2:i+4] = ['', '']
                        i += 2
                        continue

        return ''.join(corrected_text)

    def _correct_grammar(self, text: str) -> str:
        """
        Correct grammar errors in the text based on high-confidence grammar rules.

        Args:
            text: Input text.

        Returns:
            Corrected text.
        """
        # print(self.grammar_rules)
        if not self.grammar_rules:
            return text  # 没有规则就直接返回

        words_pos = [(word, flag) for word, flag in pseg.cut(text)]
        corrected_words = []

        for i, (word, pos) in enumerate(words_pos):
            # 1. 先进行词性修正
            if i > 0:
                _, prev_pos = words_pos[i - 1]
                if (prev_pos, pos) in self.grammar_rules.get("pos_replace", {}):
                    confidence = self.grammar_rules["pos_replace"][(prev_pos, pos)]
                    if confidence >= 0.1:
                        pos = self.grammar_rules["pos_replace"][(prev_pos, pos)]  # 更新词性

            # 2. 进行单词级别修正
            if word in self.grammar_rules.get("word_replace", {}):  # 修正匹配规则
                tgt_word, confidence = self.grammar_rules["word_replace"][word]
                if confidence >= 0.1:
                    word = tgt_word  # 替换错误单词

            corrected_words.append(word)  # 这里才是正常的追加

        # print("".join(corrected_words))
        # print(text)
        # print('')

        return "".join(corrected_words)


    def _correct_word_confusion(self, text: str) -> str:
        """
        Correct word-level confusion errors in the text.

        Args:
            text: Input text.

        Returns:
            Text with corrected words.
        """
        if not hasattr(self, 'word_confusion_rules') or not self.word_confusion_rules:
            return text
            
        words = [word for word, _ in pseg.cut(text)]
        corrected = []
        
        WINDOW_SIZE = 1 
        POS_FILTER = {'的': ['u'], '得': ['v'], '地': ['n']}  
        
        for i, word in enumerate(words):
            if word in self.word_confusion_rules:
                tgt_word, confidence = self.word_confusion_rules[word]
                
                # 检查词性约束（如果为该词定义了约束）
                if word in POS_FILTER:
                    _, pos = next(pseg.cut(word))  
                    if pos not in POS_FILTER[word]:
                        corrected.append(word)
                        continue
                        
                # 检查上下文窗口
                left_context = words[max(0, i-WINDOW_SIZE):i]
                right_context = words[i+1:i+1+WINDOW_SIZE]
                
                # 简单上下文检查（可扩展为更复杂的特征）
                context_valid = True
                if word == "的":
                    if any(w in {"地", "得"} for w in left_context+right_context):
                        context_valid = False
                
                if context_valid and confidence >= 0.9:
                    corrected.append(tgt_word)
                    # print(f"易混淆词替换: '{word}' → '{tgt_word}' (位置: {i})")
                else:
                    corrected.append(word)
            else:
                corrected.append(word)
        
        return ''.join(corrected)