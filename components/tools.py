"""
工具组件 - 提供印象查询和搜索功能
"""

from typing import Dict, Any, Optional, List
from src.plugin_system import BaseTool, ToolParamType

from ..models import UserImpression
from ..services import TextImpressionService


class GetUserImpressionTool(BaseTool):
    """获取用户印象和好感度工具"""

    name = "get_user_impression"
    description = "获取用户印象和好感度数据，用于生成个性化回复"
    available_for_llm = True

    parameters = [
        ("user_id", ToolParamType.STRING, "用户QQ号或ID", True, None),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_impression_service = None

    def _get_text_impression_service(self) -> TextImpressionService:
        """获取文本印象服务"""
        if not self.text_impression_service:
            from ..clients import LLMClient
            llm_config = self.plugin_config.get("llm_provider", {})
            llm_client = LLMClient(llm_config)
            self.text_impression_service = TextImpressionService(llm_client, self.plugin_config)

        return self.text_impression_service

    async def execute(self, function_args: dict) -> dict:
        """执行获取印象"""
        try:
            import logging
            logger = logging.getLogger("impression_affection_system")
            
            user_id = function_args.get("user_id")
            if not user_id:
                return {
                    "name": self.name,
                    "content": "错误：缺少user_id参数"
                }

            # 标准化用户ID以确保一致性
            from ..services.database_service import DatabaseService
            normalized_user_id = DatabaseService.normalize_user_id(user_id)
            logger.debug(f"查询用户 {normalized_user_id} 的印象数据 (原始ID: {user_id})")

            # 仅使用精确匹配，禁用模糊匹配以防止错误匹配
            impression = None

            # 直接精确匹配（使用标准化ID）
            try:
                impression = UserImpression.select().where(
                    UserImpression.user_id == normalized_user_id
                ).first()
            except Exception as db_error:
                logger.error(f"精确匹配失败: {str(db_error)}")
                impression = None

            if impression:
                # 记录查询成功日志
                logger.info(f"成功获取用户 {normalized_user_id} 的印象 (消息数: {impression.message_count}, 好感度: {impression.affection_score:.1f})")

                # 获取自然语言印象 - 从正确的字段读取
                natural_impression = impression.personality_traits.strip()  # 修复：从实际存储的字段读取

                # 显示原始查询ID
                display_id = user_id

                # 印象读取成功日志
                if natural_impression:
                    logger.info(f"印象读取成功: 用户 {display_id}, 印象长度: {len(natural_impression)} 字符")
                else:
                    logger.warning(f"印象数据为空: 用户 {display_id}")

                # 如果没有自然印象，显示默认信息
                if not natural_impression:
                    natural_impression = "用户印象正在构建中..."

                result = f"""
用户印象数据 (ID: {display_id})
━━━━━━━━━━━━━━━━━━━━━━
印象描述: {natural_impression}

好感度: {impression.affection_score:.1f}/100 ({impression.affection_level})
累计消息: {impression.message_count} 条
更新时间: {impression.updated_at.strftime('%Y-%m-%d %H:%M:%S')}
━━━━━━━━━━━━━━━━━━━━━━
                """.strip()
            else:
                logger.warning(f"用户 {normalized_user_id} 无印象数据")
                result = f"暂无用户 {user_id} 的印象数据"

            return {
                "name": self.name,
                "content": result
            }

        except Exception as e:
            import logging
            logger = logging.getLogger("impression_affection_system")
            logger.error(f"获取印象数据异常: {str(e)}")
            return {
                "name": self.name,
                "content": f"获取印象数据失败: {str(e)}"
            }


class SearchImpressionsTool(BaseTool):
    """搜索相关印象工具"""

    name = "search_impressions"
    description = "根据关键词搜索用户印象中的相关内容"
    available_for_llm = True

    parameters = [
        ("user_id", ToolParamType.STRING, "用户QQ号或ID", True, None),
        ("keyword", ToolParamType.STRING, "搜索关键词", True, None),
    ]

    async def execute(self, function_args: dict) -> dict:
        """执行印象搜索"""
        try:
            user_id = function_args.get("user_id")
            keyword = function_args.get("keyword")

            if not user_id or not keyword:
                return {
                    "name": self.name,
                    "content": "错误：缺少必要参数"
                }

            # 标准化用户ID以确保一致性
            from ..services.database_service import DatabaseService
            normalized_user_id = DatabaseService.normalize_user_id(user_id)

            # 获取用户的印象数据
            impression = UserImpression.select().where(
                UserImpression.user_id == normalized_user_id
            ).first()

            if not impression:
                return {
                    "name": self.name,
                    "content": f"用户 {user_id} 暂无印象数据"
                }

            # 在自然语言印象中搜索关键词
            natural_impression = impression.personality_traits.strip()  # 修复：从实际存储的字段读取
            
            if natural_impression and keyword.lower() in natural_impression.lower():
                # 提取包含关键词的上下文
                words = natural_impression.split()
                context_parts = []
                
                for i, word in enumerate(words):
                    if keyword.lower() in word.lower():
                        # 获取前后各2个词作为上下文
                        start = max(0, i - 2)
                        end = min(len(words), i + 3)
                        context = " ".join(words[start:end])
                        context_parts.append(f"...{context}...")
                
                result = f"用户 {user_id} 印象描述中找到关键词「{keyword}」:\n\n"
                result += f"完整印象: {natural_impression}\n\n"
                if context_parts:
                    result += "相关上下文:\n" + "\n".join(context_parts)
                result += f"\n\n好感度: {impression.affection_score:.1f}/100 ({impression.affection_level})"
                result += f"\n更新时间: {impression.updated_at.strftime('%Y-%m-%d %H:%M:%S')}"
            else:
                result = f"用户 {user_id} 的印象描述中未找到关键词「{keyword}」"

            return {
                "name": self.name,
                "content": result
            }

        except Exception as e:
            return {
                "name": self.name,
                "content": f"搜索印象失败: {str(e)}"
            }
