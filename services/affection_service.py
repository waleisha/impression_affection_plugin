"""
好感度更新服务
"""

from typing import Dict, Any, Tuple
from datetime import datetime

from ..models import UserImpression
from ..clients import LLMClient
from src.common.logger import get_logger
from ..utils.constants import AFFECTION_LEVELS, DIFFICULTY_LEVELS, AFFECTION_INCREMENTS


class AffectionService:
    """好感度更新服务"""

    def __init__(self, llm_client: LLMClient, config: Dict[str, Any]):
        self.llm_client = llm_client
        self.config = config
        self.prompts_config = config.get("prompts", {})
        self.logger = get_logger(__name__)
        self.increment_config = config.get("affection_increment", {})

        # 获取全局难度设置
        self.default_difficulty = config.get("difficulty", {}).get("level", "normal")
        self.allow_difficulty_change = config.get("difficulty", {}).get("allow_user_change", True)

        # 默认增幅配置（Easy/Normal模式）
        self.friendly_increment = self.increment_config.get("friendly_increment", 2.0)
        self.neutral_increment = self.increment_config.get("neutral_increment", 0.5)
        self.negative_increment = self.increment_config.get("negative_increment", -3.0)

    async def update_affection(self, user_id: str, message: str) -> Tuple[bool, str]:
        """
        更新用户好感度

        Args:
            user_id: 用户ID
            message: 消息内容

        Returns:
            (是否成功, 结果描述)
        """
        try:
            # 获取或创建用户印象记录
            impression, created = UserImpression.get_or_create(
                user_id=user_id,
                defaults={
                    "affection_score": 50.0,
                    "affection_level": "一般",
                    "difficulty_level": self.default_difficulty
                }
            )

            # 获取用户的难度等级
            difficulty = impression.difficulty_level

            # 获取难度配置
            difficulty_config = DIFFICULTY_LEVELS.get(difficulty, DIFFICULTY_LEVELS["normal"])
            difficulty_increments = AFFECTION_INCREMENTS.get(difficulty, AFFECTION_INCREMENTS["normal"])
            multiplier = difficulty_config.get("multiplier", 1.0)

            # 评估评论类型
            comment_type, reason = await self._evaluate_comment_type(message)

            # 计算增幅（根据难度等级）
            increment = self._calculate_increment(comment_type, difficulty, difficulty_increments)

            # Nightmare 模式的特殊处理：可能降低好感度
            if difficulty == "nightmare":
                # 在 Nightmare 模式下，即使是友善的话也可能不够有说服力
                nightmare_verdict, nightmare_reason = await self._evaluate_nightmare_mode(message, comment_type)
                if nightmare_verdict == "disagreement":
                    # 用户完全不同意或不感兴趣
                    increment = -difficulty_increments.get("strong_disagreement", -5.0)
                elif nightmare_verdict == "minor_disagreement":
                    # 轻微不同意
                    increment = -difficulty_increments.get("minor_disagreement", -2.0)
                elif nightmare_verdict == "strong_agreement":
                    # 强烈同意
                    increment = difficulty_increments.get("strong_agreement", 2.0)
                else:
                    # 普通聊天，多数情况下会扣分
                    increment = -abs(increment) if increment > 0 else increment

                reason = f"Nightmare模式: {nightmare_reason}"

            # 应用难度倍数
            increment = increment * multiplier

            # 保存原始分数用于显示
            old_score = impression.affection_score

            # 更新好感度分数
            new_score = old_score + increment
            new_score = max(0, min(100, new_score))  # 限制在 0-100

            impression.affection_score = new_score
            impression.affection_level = self._get_affection_level(new_score)
            impression.message_count += 1
            impression.update_timestamps()
            impression.save()

            # 构建结果消息
            if difficulty == "nightmare":
                result = f"Nightmare模式: {reason}\n"
            else:
                result = f"好感度更新: {comment_type} "

            result += f"({old_score:.1f} -> {new_score:.1f})\n"
            result += f"当前好感度: {impression.affection_level} ({new_score:.1f}/100)\n"
            result += f"难度: {difficulty_config['name']}"

            return True, result

        except Exception as e:
            return False, f"更新好感度失败: {str(e)}"

    async def _evaluate_comment_type(self, message: str) -> Tuple[str, str]:
        """评估评论类型"""
        prompt = self._build_affection_prompt(message)

        success, content = await self.llm_client.generate_affection_analysis(prompt)

        if not success:
            raise Exception(f"LLM调用失败: {content}")

        result = self._parse_affection_response(content)

        if not result:
            return "neutral", "解析失败，默认为中性"

        return result.get("type", "neutral"), result.get("reason", "")

    async def _evaluate_nightmare_mode(self, message: str, initial_type: str) -> Tuple[str, str]:
        """
        Nightmare 模式特殊评估 - 判断用户的真实观点是否与AI一致

        Returns:
            (评估结果, 原因), 其中评估结果可以是:
            - "strong_agreement": 强烈同意
            - "normal": 普通聊天
            - "minor_disagreement": 轻微不同意
            - "disagreement": 完全不同意/没有说服力
        """
        prompt = f"""你是一个严厉的评论家。用户的这句话有说服力吗？
        
消息: {message}

请严格评估：
1. 这句话是否表达了真实观点？
2. 是否有说服力和深度？
3. 是否只是虚伪的夸奖或敷衍？

评估标准：
- strong_agreement: 观点深刻、真挚、有说服力
- normal: 普通的对话，但没有特别的说服力
- minor_disagreement: 观点略显肤浅或不真诚
- disagreement: 明显是敷衍、虚伪或没有实际内容

只返回键值对格式：
VERDICT: strong_agreement/normal/minor_disagreement/disagreement;REASON: 原因;消息: {message}"""

        success, content = await self.llm_client.generate_affection_analysis(prompt)

        if not success:
            # 失败时默认为普通评估
            return "normal", "评估失败"

        result = self._parse_affection_response(content)

        if result:
            verdict = result.get("type", "normal")  # 使用 type 字段
            reason = result.get("reason", "")
            # 映射 type 到 verdict
            verdict_map = {
                "strong_agreement": "strong_agreement",
                "normal": "normal",
                "minor_disagreement": "minor_disagreement",
                "disagreement": "disagreement",
                "negative": "disagreement",  # 负面评价视为完全不同意
            }
            verdict = verdict_map.get(verdict, "normal")
            return verdict, reason

        return "normal", "解析失败"

    def _build_affection_prompt(self, message: str) -> str:
        """构建好感度评估提示词 - 原有逻辑"""
        template = self.prompts_config.get("affection_template", "").strip()

        if template:
            return template.format(message=message, context="")

        # 默认提示词
        return f"""你是一个情感分析师。请评估用户消息的情感倾向。

评估标准：
- friendly: 友善的评论（赞美、鼓励、感谢等）
- neutral: 中性的评论（客观陈述、信息性消息等）
- negative: 差劲的评论（批评、讽刺、攻击等）

回复要求：只返回键值对格式，不要包含其他内容

格式：
TYPE: friendly/neutral/negative;REASON: 评估原因;消息: {message}"""

    def _parse_affection_response(self, content: str) -> Dict[str, str]:
        """解析好感度响应 - 原有逻辑"""
        import re

        content = content.strip()

        if len(content) < 10:
            return {}

        # 使用正则表达式提取 TYPE/VERDICT 和 REASON
        type_match = re.search(r'(?:TYPE|VERDICT):\s*(\w+)', content, re.IGNORECASE)
        reason_match = re.search(r'REASON:\s*(.+?)(?:\n|;消息:|$)', content, re.IGNORECASE)

        result = {}
        if type_match:
            result["type"] = type_match.group(1).strip().lower()
        if reason_match:
            result["reason"] = reason_match.group(1).strip()

        return result if result else {}

    def _calculate_increment(self, comment_type: str, difficulty: str, increments: Dict[str, float]) -> float:
        """
        计算好感度增幅

        Args:
            comment_type: 评论类型 (friendly/neutral/negative)
            difficulty: 难度等级
            increments: 难度对应的增幅配置
        """
        # 获取对应难度下的增幅
        increment = increments.get(comment_type, increments.get("neutral", 0))
        return increment

    def _get_affection_level(self, score: float) -> str:
        """根据分数获取好感度等级 - 原有逻辑"""
        for (min_score, max_score), level in AFFECTION_LEVELS.items():
            if min_score <= score <= max_score:
                return level
        return "一般"

    def set_difficulty(self, user_id: str, difficulty_level: str) -> Tuple[bool, str]:
        """设置用户难度等级"""
        if not self.allow_difficulty_change:
            return False, "不允许改变难度等级"

        valid_levels = list(DIFFICULTY_LEVELS.keys())
        if difficulty_level not in valid_levels:
            return False, f"无效的难度等级: {difficulty_level}"

        try:
            impression, _ = UserImpression.get_or_create(user_id=user_id)
            impression.set_difficulty(difficulty_level)
            impression.save()

            config = DIFFICULTY_LEVELS[difficulty_level]
            return True, f"难度已设置为: {config['name']} - {config['description']}"
        except Exception as e:
            return False, f"设置难度失败: {str(e)}"

    def get_affection_summary(self, user_id: str) -> str:
        """获取用户好感度摘要"""
        try:
            impression = UserImpression.get_or_none(UserImpression.user_id == user_id)
            if not impression:
                return "该用户还没有好感度记录"

            difficulty_config = DIFFICULTY_LEVELS.get(impression.difficulty_level, DIFFICULTY_LEVELS["normal"])

            summary = f"当前好感度: {impression.affection_score:.1f}/100 ({impression.affection_level})\n"
            summary += f"难度等级: {difficulty_config['name']}\n"
            summary += f"消息统计: {impression.message_count} 条\n"
            summary += f"印象版本: {impression.impression_version}"

            return summary
        except Exception as e:
            return f"获取失败: {str(e)}"

    def get_difficulty_info(self, user_id: str) -> Dict[str, Any]:
        """获取用户的难度信息"""
        try:
            impression = UserImpression.get_or_none(UserImpression.user_id == user_id)
            if impression:
                difficulty = impression.difficulty_level
                config = DIFFICULTY_LEVELS.get(difficulty, DIFFICULTY_LEVELS["normal"])
                return {
                    "level": difficulty,
                    "name": config["name"],
                    "description": config["description"],
                    "multiplier": config.get("multiplier", 1.0)
                }
            else:
                return {
                    "level": self.default_difficulty,
                    "name": DIFFICULTY_LEVELS[self.default_difficulty]["name"],
                    "description": DIFFICULTY_LEVELS[self.default_difficulty]["description"],
                    "multiplier": DIFFICULTY_LEVELS[self.default_difficulty].get("multiplier", 1.0)
                }
        except Exception as e:
            self.logger.error(f"获取难度信息失败: {str(e)}")
            return {
                "level": self.default_difficulty,
                "name": DIFFICULTY_LEVELS[self.default_difficulty]["name"],
                "description": DIFFICULTY_LEVELS[self.default_difficulty]["description"],
                "multiplier": DIFFICULTY_LEVELS[self.default_difficulty].get("multiplier", 1.0)
            }
    def list_all_difficulties(self) -> Dict[str, Dict[str, str]]:
        """列出所有难度等级"""
        return {
            key: {
                "name": value["name"],
                "description": value["description"]
            }
            for key, value in DIFFICULTY_LEVELS.items()
        }
