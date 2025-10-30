#!/usr/bin/env python3
"""
ActionPiece 词汇表构建诊断脚本
深度分析为什么嵌入权重没有影响词汇表构建效果
"""

import json
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

class VocabularyBuildingDiagnostic:
    """词汇表构建诊断器"""

    def __init__(self):
        self.results = {}

    def analyze_bpe_nature(self):
        """分析BPE算法的本质特性"""
        print("🔍 BPE算法本质特性分析")
        print("=" * 60)

        analysis = {
            "frequency_dominance": {
                "description": "BPE主要由频次驱动",
                "explanation": [
                    "BPE每次选择出现频次最高的token pair进行合并",
                    "频次差异通常是数量级级别的（如1000 vs 10）",
                    "嵌入权重通常在0.5-2.0范围内变化",
                    "权重影响被高频次主导，难以改变合并顺序"
                ]
            },
            "greedy_selection": {
                "description": "贪心选择策略的局限性",
                "explanation": [
                    "每步只选择当前最优pair，缺乏全局优化",
                    "早期的合并决策影响后续所有选择",
                    "权重需要足够大才能改变贪心路径"
                ]
            },
            "convergence_effect": {
                "description": "词汇表收敛效应",
                "explanation": [
                    "高频patterns会被优先发现，无论权重如何",
                    "低频但高权重的pairs可能永远不会被选中",
                    "最终词汇表趋向于相似的高频结构"
                ]
            }
        }

        for aspect, info in analysis.items():
            print(f"\n📊 {info['description']}")
            for point in info['explanation']:
                print(f"   • {point}")

        return analysis

    def simulate_weight_impact(self, freq_counts, weights, weight_factors):
        """模拟不同权重因子对选择的影响"""
        print(f"\n🧪 模拟权重影响实验")
        print("=" * 60)

        # 创建测试数据
        pairs = [f"pair_{i}" for i in range(len(freq_counts))]

        results = {}
        for factor in weight_factors:
            print(f"\n🔬 权重因子: {factor}")

            # 计算加权分数
            weighted_scores = []
            for i, (freq, weight) in enumerate(zip(freq_counts, weights)):
                if factor == 0:
                    score = freq  # 无权重
                else:
                    score = freq * (weight ** factor)
                weighted_scores.append(score)

            # 排序
            sorted_pairs = sorted(zip(pairs, freq_counts, weights, weighted_scores),
                                key=lambda x: x[3], reverse=True)

            results[factor] = sorted_pairs

            # 显示前5个
            print("   Top-5 pairs:")
            for j, (pair, freq, weight, score) in enumerate(sorted_pairs[:5]):
                print(f"   {j+1}. {pair}: freq={freq}, weight={weight:.3f}, score={score:.1f}")

            # 计算排序变化
            if factor == 0:
                baseline_order = [x[0] for x in sorted_pairs]
            else:
                current_order = [x[0] for x in sorted_pairs]
                # 计算Kendall's tau或简单的重叠度
                overlap = len(set(baseline_order[:10]) & set(current_order[:10]))
                print(f"   与无权重的Top-10重叠度: {overlap}/10 ({overlap*10}%)")

        return results

    def analyze_frequency_distribution(self, cache_dir):
        """分析实际数据中的频次分布"""
        print(f"\n📈 分析实际频次分布")
        print("=" * 60)

        # 尝试加载词汇表文件来分析
        vocab_files = [
            "actionpiece.json",
            "vocab.json"
        ]

        vocab_data = None
        for vocab_file in vocab_files:
            try:
                with open(f"{cache_dir}/{vocab_file}", 'r') as f:
                    vocab_data = json.load(f)
                print(f"✅ 加载词汇表: {vocab_file}")
                break
            except:
                continue

        if vocab_data is None:
            print("❌ 未找到词汇表文件，使用模拟数据")
            return self._simulate_frequency_analysis()

        return self._analyze_real_vocabulary(vocab_data)

    def _simulate_frequency_analysis(self):
        """模拟频次分析"""
        # 创建符合Zipf分布的频次数据
        ranks = np.arange(1, 1001)
        frequencies = 1000 / ranks  # Zipf分布

        print(f"📊 模拟频次统计 (Zipf分布):")
        print(f"   最高频次: {frequencies[0]:.1f}")
        print(f"   中位频次: {np.median(frequencies):.1f}")
        print(f"   最低频次: {frequencies[-1]:.1f}")
        print(f"   频次比 (max/median): {frequencies[0]/np.median(frequencies):.1f}x")

        # 模拟权重
        weights = np.random.normal(1.0, 0.15, len(frequencies))
        weights = np.clip(weights, 0.5, 1.5)

        print(f"\n🎯 权重统计:")
        print(f"   权重均值: {np.mean(weights):.3f}")
        print(f"   权重标准差: {np.std(weights):.3f}")
        print(f"   权重范围: [{np.min(weights):.3f}, {np.max(weights):.3f}]")

        # 测试不同权重因子的影响
        weight_factors = [0, 0.5, 1.0, 2.0, 5.0]
        simulation_results = self.simulate_weight_impact(
            frequencies[:20], weights[:20], weight_factors
        )

        return {
            'frequencies': frequencies,
            'weights': weights,
            'simulation_results': simulation_results
        }

    def _analyze_real_vocabulary(self, vocab_data):
        """分析真实词汇表数据"""
        print(f"📊 真实词汇表分析:")
        print(f"   词汇表大小: {len(vocab_data)}")

        # 这里需要根据实际的vocab_data结构进行分析
        # 通常包含token到ID的映射或规则序列
        return {'vocab_size': len(vocab_data), 'vocab_data': vocab_data}

    def diagnose_weight_integration_issues(self):
        """诊断权重集成的具体问题"""
        print(f"\n🔧 诊断权重集成问题")
        print("=" * 60)

        potential_issues = {
            "numerical_scale": {
                "问题": "数值尺度不匹配",
                "描述": "频次通常是整数(1-1000+)，权重是小数(0.5-2.0)",
                "影响": "权重变化相对于频次变化太小",
                "解决方案": [
                    "使用对数空间: log(freq) + γ*log(weight)",
                    "权重放大: freq * (weight^α), α>1",
                    "归一化处理: 将频次和权重都归一化到相同尺度"
                ]
            },
            "optimization_objective": {
                "问题": "优化目标不匹配",
                "描述": "BPE优化压缩率，权重优化语义相关性",
                "影响": "两个目标可能冲突",
                "解决方案": [
                    "多目标优化: α*freq + β*semantic_score",
                    "分阶段优化: 先语义后频次",
                    "约束优化: 在语义约束下最大化频次"
                ]
            },
            "algorithm_limitation": {
                "问题": "算法本身的局限性",
                "描述": "贪心算法难以考虑全局语义结构",
                "影响": "局部最优解，无法达到全局语义最优",
                "解决方案": [
                    "分层构建: 先构建语义clusters再应用BPE",
                    "后处理优化: 构建后基于权重调整",
                    "替代算法: 使用更适合语义的分词算法"
                ]
            }
        }

        for issue_type, info in potential_issues.items():
            print(f"\n⚠️  {info['问题']}")
            print(f"   描述: {info['描述']}")
            print(f"   影响: {info['影响']}")
            print("   解决方案:")
            for solution in info['解决方案']:
                print(f"   • {solution}")

        return potential_issues

    def recommend_solutions(self):
        """推荐解决方案"""
        print(f"\n💡 推荐解决方案")
        print("=" * 60)

        solutions = {
            "immediate_fixes": {
                "标题": "立即可行的修复",
                "方案": [
                    {
                        "名称": "对数空间混合",
                        "实现": "log_score = log(freq) + γ * (log_w1 + log_w2)",
                        "优势": "避免数值尺度问题，更好的权重影响",
                        "风险": "需要调整γ参数"
                    },
                    {
                        "名称": "权重放大",
                        "实现": "score = freq * (weight^α), α=2-5",
                        "优势": "简单直接，增强权重影响",
                        "风险": "可能过度放大权重差异"
                    },
                    {
                        "名称": "分位数归一化",
                        "实现": "将freq和weight都转为分位数排名",
                        "优势": "消除尺度差异，保持相对关系",
                        "风险": "可能丢失绝对数值信息"
                    }
                ]
            },
            "advanced_approaches": {
                "标题": "高级方法",
                "方案": [
                    {
                        "名称": "语义引导的分层BPE",
                        "实现": "先基于权重聚类，再在cluster内应用BPE",
                        "优势": "结合语义结构和统计特性",
                        "风险": "增加复杂性，需要更多参数"
                    },
                    {
                        "名称": "多目标优化",
                        "实现": "Pareto最优: 同时优化频次和语义得分",
                        "优势": "平衡多个目标",
                        "风险": "计算复杂度高"
                    },
                    {
                        "名称": "后处理重排",
                        "实现": "标准BPE后基于权重调整合并顺序",
                        "优势": "不改变核心算法，风险小",
                        "风险": "可能破坏BPE的一致性"
                    }
                ]
            }
        }

        for category, info in solutions.items():
            print(f"\n🎯 {info['标题']}")
            for i, solution in enumerate(info['方案'], 1):
                print(f"\n   {i}. {solution['名称']}")
                print(f"      实现: {solution['实现']}")
                print(f"      优势: {solution['优势']}")
                print(f"      风险: {solution['风险']}")

        return solutions

    def run_full_diagnostic(self, cache_dir="cache/AmazonReviews2014/Beauty/processed"):
        """运行完整诊断"""
        print("🚀 ActionPiece 词汇表构建诊断")
        print("=" * 80)

        # 1. 分析BPE本质
        bpe_analysis = self.analyze_bpe_nature()

        # 2. 分析频次分布
        freq_analysis = self.analyze_frequency_distribution(cache_dir)

        # 3. 诊断权重集成问题
        integration_issues = self.diagnose_weight_integration_issues()

        # 4. 推荐解决方案
        solutions = self.recommend_solutions()

        # 5. 总结诊断结果
        self._print_diagnostic_summary()

        return {
            'bpe_analysis': bpe_analysis,
            'freq_analysis': freq_analysis,
            'integration_issues': integration_issues,
            'solutions': solutions
        }

    def _print_diagnostic_summary(self):
        """打印诊断总结"""
        print(f"\n📋 诊断总结")
        print("=" * 60)

        conclusions = [
            "🎯 核心问题: BPE算法天然偏向高频patterns，权重影响被稀释",
            "📊 数值问题: 频次(整数1-1000+) vs 权重(小数0.5-2.0)的尺度不匹配",
            "🔀 算法局限: 贪心策略难以整合全局语义信息",
            "💡 解决方向: 需要在数值空间或算法层面做根本性改进"
        ]

        for conclusion in conclusions:
            print(f"   {conclusion}")

        print(f"\n🚀 下一步行动建议:")
        print(f"   1. 实施对数空间混合 (最快见效)")
        print(f"   2. 调整权重放大因子 (简单可控)")
        print(f"   3. 考虑分层构建方法 (根本解决)")

def main():
    """主函数"""
    diagnostic = VocabularyBuildingDiagnostic()
    results = diagnostic.run_full_diagnostic()

    print(f"\n🎉 诊断完成！")
    print(f"建议优先尝试对数空间混合方法来解决权重影响问题。")

if __name__ == "__main__":
    main()
