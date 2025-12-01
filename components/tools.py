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

            logger.debug(f"查询用户 {user_id} 的印象数据")

            # 尝试多种用户ID匹配方式
            impression = None
            matched_id = None
            
            # 方法1: 直接匹配
            try:
                impression = UserImpression.select().where(
                    UserImpression.user_id == user_id
                ).first()
                if impression:
                    matched_id = user_id
                    logger.debug(f"直接匹配成功: {user_id}")
            except Exception as db_error:
                logger.debug(f"直接匹配失败: {str(db_error)}")
            
            # 方法2: 如果直接匹配失败，尝试从消息状态表中查找
            if not impression:
                try:
                    from ..models import UserMessageState
                    # 查找包含用户名的记录（模糊匹配）
                    state_records = list(UserMessageState.select().limit(10))
                    
                    for record in state_records:
                        # 这里可以添加更复杂的匹配逻辑，比如昵称映射
                        # 目前先尝试查找最近的记录
                        test_impression = UserImpression.select().where(
                            UserImpression.user_id == record.user_id
                        ).first()
                        if test_impression:
                            impression = test_impression
                            matched_id = record.user_id
                            logger.debug(f"通过消息状态表匹配: {user_id} -> {matched_id}")
                            break
                            
                except Exception as db_error:
                    logger.debug(f"消息状态表匹配失败: {str(db_error)}")
            
            # 方法3: 如果还是没有找到，返回最新的印象记录（用于调试）
            if not impression:
                try:
                    impression = UserImpression.select().order_by(UserImpression.updated_at.desc()).first()
                    if impression:
                        matched_id = impression.user_id
                        logger.warning(f"使用最新记录作为备选: {user_id} -> {matched_id}")
                except Exception as db_error:
                    logger.debug(f"备选匹配失败: {str(db_error)}")
            
            logger.debug(f"最终查询结果: {impression}, 匹配ID: {matched_id}")

            if impression:
                # 获取自然语言印象
                natural_impression = impression.interests_hobbies.strip()  # 从复用的字段获取
                
                # 显示原始查询ID和实际匹配的ID
                display_id = user_id
                if matched_id and matched_id != user_id:
                    display_id = f"{user_id} (实际ID: {matched_id})"
                
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
                # 检查是否有其他相关记录
                debug_info = ""
                try:
                    from ..models import UserMessageState, ImpressionMessageRecord
                    
                    # 显示所有用户的记录状态，帮助调试
                    all_states = list(UserMessageState.select().limit(5))
                    state_info = []
                    for state in all_states:
                        state_info.append(f"{state.user_id}({state.total_messages}条消息)")
                    
                    if state_info:
                        debug_info = f"现有用户: {', '.join(state_info)}"
                    else:
                        debug_info = "无用户记录"
                    
                    logger.debug(f"用户 {user_id} 无印象数据 - {debug_info}")
                    
                except Exception as debug_error:
                    logger.error(f"调试信息查询失败: {str(debug_error)}")
                    debug_info = "调试信息查询失败"
                
                result = f"暂无用户 {user_id} 的印象数据 ({debug_info})"

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

            # 获取用户的印象数据
            impression = UserImpression.select().where(
                UserImpression.user_id == user_id
            ).first()

            if not impression:
                return {
                    "name": self.name,
                    "content": f"用户 {user_id} 暂无印象数据"
                }

            # 在自然语言印象中搜索关键词
            natural_impression = impression.interests_hobbies.strip()  # 从复用的字段获取
            
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
