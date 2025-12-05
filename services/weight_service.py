"""
权重评估服务 - 评估消息权重，筛选高价值消息
"""

import os
import hashlib
from typing import Dict, Any, Tuple, Optional, List, Set
from datetime import datetime
from collections import defaultdict

from ..clients import LLMClient
from .database_service import DatabaseService
from src.common.logger import get_logger

logger = get_logger("impression_affection_weight")


class WeightService:
    """权重评估服务"""

    def __init__(self, llm_client: LLMClient, config: Dict[str, Any]):
        self.llm_client = llm_client
        self.config = config
        self.weight_config = config.get("weight_filter", {})
        self.prompts_config = config.get("prompts", {})

        # 阈值配置
        self.high_threshold = self.weight_config.get("high_weight_threshold", 70.0)
        self.medium_threshold = self.weight_config.get("medium_weight_threshold", 40.0)
        self.filter_mode = self.weight_config.get("filter_mode", "selective")
        
        # 自定义权重模型配置
        self.use_custom_weight_model = self.weight_config.get("use_custom_weight_model", False)
        self.weight_llm_client = None
        
        # 内存存储 - 按用户存储消息权重记录
        self.message_weights = defaultdict(list)  # {user_id: [(message_id, score, level, timestamp), ...]}
        
        # 数据库服务
        self.db_service = DatabaseService(config)
        
        # 初始化权重评估客户端
        self._init_weight_llm_client()

    

    def is_message_processed(self, user_id: str, message_id: str = None) -> bool:
        """
        检查消息是否已经处理过（基于message_id）
        
        Args:
            user_id: 用户ID
            message_id: 主程序的实际message_id
            
        Returns:
            是否已处理
        """
        # 如果没有message_id，返回False（让主流程处理）
        if not message_id:
            logger.debug(f"无message_id，无法查重: 用户 {user_id}")
            return False
        
        # 通过MessageService检查（统一查重标准）
        try:
            # 导入MessageService避免循环导入
            from .message_service import MessageService
            message_service = MessageService(self.config)
            return message_service.is_message_processed(user_id, message_id)
        except Exception as e:
            logger.error(f"查重检查失败: {str(e)}")
            return False

    def mark_message_processed(self, user_id: str, message_id: str = None):
        """
        标记消息为已处理（基于message_id）
        
        Args:
            user_id: 用户ID
            message_id: 主程序的实际message_id
        """
        if not message_id:
            logger.debug(f"无message_id，无法标记处理: 用户 {user_id}")
            return
        
        # 通过MessageService标记（统一查重标准）
        try:
            # 导入MessageService避免循环导入
            from .message_service import MessageService
            message_service = MessageService(self.config)
            message_service.mark_message_processed(user_id, message_id)
            logger.debug(f"已标记消息为已处理: 用户 {user_id}, message_id {message_id}")
        except Exception as e:
            logger.error(f"标记处理消息失败: {str(e)}")

    def get_processed_message_ids(self, user_id: str) -> Set[str]:
        """
        获取用户已处理的消息ID列表
        
        Args:
            user_id: 用户ID
            
        Returns:
            已处理的消息ID集合
        """
        # 通过MessageService获取（统一查重标准）
        try:
            # 导入MessageService避免循环导入
            from .message_service import MessageService
            message_service = MessageService(self.config)
            processed_ids = message_service.get_processed_message_ids(user_id)
            
            # 过滤掉无效的ID（保留所有有效格式的message_id，包括临时ID）
            valid_ids = set()
            for msg_id in processed_ids:
                # 只要不是空字符串，就认为是有效ID（包括临时ID格式）
                if msg_id and str(msg_id).strip():
                    valid_ids.add(msg_id)
                else:
                    logger.debug(f"过滤无效message_id: {msg_id}")
            
            logger.debug(f"用户 {user_id} 有效已处理消息ID: {len(valid_ids)} 个")
            return valid_ids
            
        except Exception as e:
            logger.error(f"获取已处理消息ID失败: {str(e)}")
            return set()

    def _init_weight_llm_client(self):
        """初始化权重评估专用LLM客户端"""
        if not self.use_custom_weight_model:
            return
        
        # 避免重复初始化
        if self.weight_llm_client is not None:
            return
        
        try:
            weight_model_config = {
                "provider_type": self.weight_config.get("weight_model_provider", "openai"),
                "api_key": self.weight_config.get("weight_model_api_key", ""),
                "base_url": self.weight_config.get("weight_model_base_url", "https://api.openai.com/v1"),
                "model_id": self.weight_config.get("weight_model_id", "gpt-3.5-turbo")
            }
            
            self.weight_llm_client = LLMClient(weight_model_config)
            logger.debug(f"权重评估模型已初始化: {weight_model_config['model_id']}")
            
        except Exception as e:
            logger.error(f"权重评估模型初始化失败: {str(e)}")
            self.weight_llm_client = None

    async def evaluate_message(self, user_id: str, message_id: str, message: str, context: str = "") -> Tuple[bool, float, str]:
        """
        评估消息权重

        Args:
            user_id: 用户ID
            message_id: 消息ID
            message: 消息内容
            context: 上下文

        Returns:
            (是否成功, 权重分数, 权重等级)
        """
        try:
            # 标准化用户ID
            normalized_user_id = str(user_id).strip()
            
            # 检查消息是否已处理过（基于message_id去重）
            if self.is_message_processed(normalized_user_id, message_id):
                logger.debug(f"消息已处理，跳过权重评估: 用户 {normalized_user_id}, message_id {message_id}")
                # 返回已存在的权重评估结果
                user_messages = self.message_weights.get(normalized_user_id, [])
                for msg_record in user_messages:
                    if msg_record[0] == message_id:  # message_id matches
                        return True, msg_record[1], msg_record[2]  # score, level

            # 生成提示词（包含上下文）
            prompt = self._build_weight_prompt(message, context)

            # 选择LLM客户端进行评估
            if self.use_custom_weight_model and self.weight_llm_client:
                # 使用自定义权重模型
                success, content = await self.weight_llm_client.generate_weight_evaluation(prompt)
            else:
                # 使用默认LLM客户端
                success, content = await self.llm_client.generate_weight_evaluation(prompt)

            if not success:
                # 评估失败，使用默认权重
                return self._save_default_weight(normalized_user_id, message_id, message, context)

            # 解析结果
            result = self._parse_weight_response(content)

            if not result:
                # 解析失败，使用默认权重
                return self._save_default_weight(normalized_user_id, message_id, message, context)

            weight_score = float(result.get("weight_score", 0.0))
            weight_level = result.get("weight_level", "low")

            # 保存到内存
            self._save_weight(normalized_user_id, message_id, message, context, weight_score, weight_level)

            # 不在这里标记消息，等到印象构建成功后由plugin.py统一批量标记
            # self.mark_message_processed(normalized_user_id, message_id)  # v4修复：已删除

            return True, weight_score, weight_level

        except Exception as e:
            return False, 0.0, f"评估权重失败: {str(e)}"

    def _build_weight_prompt(self, message: str, context: str) -> str:
        """构建权重评估提示词"""
        template = self.prompts_config.get("weight_evaluation_prompt", "").strip()

        if template:
            return template.format(message=message, context=context)

        # 默认提示词 - 使用键值对格式
        return f"""基于消息内容和上下文对话，评估消息权重（0-100）。权重评估标准：高权重(70-100): 包含重要个人信息、兴趣爱好、价值观、情感表达、深度思考、独特观点、生活经历分享；中权重(40-69): 一般日常对话、简单提问、客观陈述、基础信息交流；低权重(0-39): 简单问候、客套话、无实质内容的互动、表情符号。特别注意：结合上下文判断，分享个人喜好、询问对方偏好、表达个人观点都应该给予较高权重。只返回键值对格式：WEIGHT_SCORE: 分数;WEIGHT_LEVEL: high/medium/low;REASON: 评估原因;当前消息: {message};历史上下文: {context}"""

    def _parse_weight_response(self, content: str) -> Optional[Dict[str, Any]]:
        """解析权重评估响应 - 支持键值对和JSON两种格式"""
        import logging
        import re
        import json
        
        logger = logging.getLogger("impression_affection_system")
        
        # 清理内容
        content = content.strip()
        
        # 如果内容太短
        if len(content) < 10:
            logger.error(f"LLM响应太短: {repr(content)}")
            return None
        
        # 方法1: 尝试提取键值对格式
        score_match = re.search(r'WEIGHT_SCORE:\s*([\d.]+)', content, re.IGNORECASE)
        level_match = re.search(r'WEIGHT_LEVEL:\s*(\w+)', content, re.IGNORECASE)
        reason_match = re.search(r'REASON:\s*(.+?)(?:\n|;消息:|$)', content, re.IGNORECASE)
        
        result = {}
        if score_match:
            try:
                result["weight_score"] = float(score_match.group(1))
            except:
                result["weight_score"] = 0.0
        if level_match:
            result["weight_level"] = level_match.group(1).strip().lower()
        if reason_match:
            result["reason"] = reason_match.group(1).strip()
        
        if result and "weight_score" in result:
            logger.debug(f"提取到键值对格式权重数据: {result}")
            return result
        
        # 方法2: 尝试解析JSON格式（作为后备）
        try:
            # 查找JSON部分（从第一个{到最后一个}）
            start = content.find('{')
            end = content.rfind('}')
            
            if start >= 0 and end >= 0 and end > start:
                json_str = content[start:end+1]
                json_result = json.loads(json_str)
                
                if isinstance(json_result, dict) and "weight_score" in json_result:
                    logger.debug(f"提取到JSON格式权重数据: {json_result}")
                    return json_result
        except json.JSONDecodeError as e:
            logger.debug(f"JSON解析失败: {e}")
        
        logger.error(f"无法提取权重数据: {repr(content)}")
        return None

    def _save_weight(self, user_id: str, message_id: str, message: str, context: str, weight_score: float, weight_level: str):
        """保存权重评估结果到内存"""
        self.message_weights[user_id].append((
            message_id, 
            weight_score, 
            weight_level, 
            datetime.now(),
            message[:100],  # 保存消息内容的前100字符
            context[:100]   # 保存上下文的前100字符
        ))
        
        # 限制每个用户保存的记录数，从配置读取
        weight_cache_limit = self.config.get("weight_filter", {}).get("max_weight_records", 100)
        if len(self.message_weights[user_id]) > weight_cache_limit:
            self.message_weights[user_id] = self.message_weights[user_id][-weight_cache_limit:]

    def _save_default_weight(self, user_id: str, message_id: str, message: str, context: str) -> Tuple[bool, float, str]:
        """保存默认权重"""
        # 简单规则：消息长度大于20字认为是中权重
        if len(message) > 20:
            weight_score = 50.0
            weight_level = "medium"
        else:
            weight_score = 20.0
            weight_level = "low"

        self._save_weight(user_id, message_id, message, context, weight_score, weight_level)
        return True, weight_score, weight_level

    def get_filtered_messages(self, user_id: str, limit: int = None) -> Tuple[str, list]:
        """
        获取筛选后的消息用于印象构建（包含上下文）

        Args:
            user_id: 用户ID
            limit: 最大消息数，如果为None则从配置读取

        Returns:
            (上下文字符串, 消息ID列表)
        """
        # 如果没有传入limit，从配置读取
        if limit is None:
            history_config = self.config.get("history", {})
            limit = history_config.get("max_messages", 20)

        if self.filter_mode == "disabled":
            return "", []

        # 标准化用户ID
        normalized_user_id = str(user_id).strip()
        
        # 获取已处理的消息ID，用于去重
        processed_message_ids = self.get_processed_message_ids(normalized_user_id)
        
        logger.debug(f"用户 {normalized_user_id} 查重统计: 已处理消息ID {len(processed_message_ids)} 个")
        
        # 首先尝试从数据库获取历史消息（排除已处理的）
        db_messages = self._get_historical_messages(normalized_user_id, limit, exclude_message_ids=list(processed_message_ids))
        
        # 合并内存中的消息权重记录
        user_messages = self.message_weights.get(normalized_user_id, [])
        
        # 根据筛选模式过滤消息
        filtered_messages = []
        
        logger.info(f"用户 {normalized_user_id} 消息去重统计: 已处理消息ID {len(processed_message_ids)} 条，数据库获取 {len(db_messages)} 条，内存权重记录 {len(user_messages)} 条")
        
        # 处理数据库历史消息（默认给予中等权重）
        for db_msg in db_messages:
            filtered_messages.append({
                "message_id": db_msg["message_id"],
                "weight_score": 50.0,  # 历史消息默认中等权重
                "weight_level": "medium",
                "timestamp": db_msg["datetime"],
                "content": db_msg["content"],
                "source": "database",
                "content_hash": db_msg.get("content_hash", "")
            })
        
        # 处理内存中的消息权重记录
        for msg_record in user_messages:
            message_id, weight_score, weight_level, timestamp, message_content, context = msg_record
            
            # 根据筛选模式过滤
            should_include = False
            if self.filter_mode == "selective":
                should_include = weight_score >= self.high_threshold
            elif self.filter_mode == "balanced":
                should_include = weight_score >= self.medium_threshold
            else:  # 默认包含所有消息
                should_include = True
            
            if should_include:
                filtered_messages.append({
                    "message_id": message_id,
                    "weight_score": weight_score,
                    "weight_level": weight_level,
                    "timestamp": timestamp,
                    "content": message_content,
                    "source": "memory",
                    "context": context,
                    "content_hash": hashlib.md5(message_content.encode('utf-8')).hexdigest()
                })

        # 按时间倒序排列，取前limit条
        filtered_messages.sort(key=lambda x: x["timestamp"], reverse=True)
        filtered_messages = filtered_messages[:limit]

        # 统计过滤结果
        total_db_messages = len(db_messages)
        total_memory_messages = len(user_messages)
        total_available = total_db_messages + total_memory_messages
        actually_filtered = len(filtered_messages)
        
        logger.info(f"用户 {normalized_user_id} 消息筛选统计: 数据库消息 {total_db_messages} 条，内存消息 {total_memory_messages} 条，总可用 {total_available} 条，筛选后 {actually_filtered} 条，限制 {limit} 条")

        if not filtered_messages:
            return "", []

        # 构建上下文（保持对话完整性）
        contexts = []
        message_ids = []
        
        # 添加完整的对话上下文，不分离历史和当前消息
        contexts.append(f"用户 {normalized_user_id} 的对话记录:")
        
        for msg in filtered_messages:
            timestamp = msg["timestamp"]
            content = msg["content"]
            weight_score = msg["weight_score"]
            weight_level = msg["weight_level"]
            source = msg["source"]
            
            if isinstance(timestamp, datetime):
                time_str = timestamp.strftime('%m-%d %H:%M')
            else:
                time_str = str(timestamp)
            
            context_str = f"[{time_str}] {content}"
            if weight_score > 0:
                context_str += f" (权重: {weight_score:.1f}, 等级: {weight_level})"
            if source == "database":
                context_str += " [历史]"
            
            contexts.append(context_str)
            message_ids.append(msg["message_id"])

        result = "\n".join(contexts)

        return result, message_ids

    def _get_historical_messages(self, user_id: str, limit: int = 10, exclude_message_ids: List[str] = None) -> list:
        """
        从数据库获取历史消息
        
        Args:
            user_id: 用户ID
            limit: 最大消息数
            exclude_message_ids: 要排除的消息ID列表
            
        Returns:
            历史消息列表
        """
        if not self.db_service or not self.db_service.is_connected():
            return []
        
        try:
            # 从配置获取参数
            history_config = self.config.get("history", {})
            max_messages = history_config.get("max_messages", 20)
            hours_back = history_config.get("hours_back", 72)
            min_length = history_config.get("min_message_length", 5)
            
            # 转换小时数为天数（数据库服务需要天数参数）
            days_back = max(1, hours_back // 24)  # 至少1天
            
            # 使用配置的参数获取历史消息，支持排除已处理的消息
            history_messages = self.db_service.get_user_chat_history(
                user_id=user_id,
                limit=min(limit, max_messages),  # 不超过配置的最大值
                days_back=days_back,
                exclude_message_ids=exclude_message_ids or []
            )
            
            # 过滤掉太短的消息
            filtered_messages = [
                msg for msg in history_messages 
                if msg["content"] and len(msg["content"].strip()) >= min_length
            ]
            
            return filtered_messages[:limit]
            
        except Exception as e:
            # 记录错误但不影响主流程
            logger.error(f"获取历史消息失败: {str(e)}")
            return []

    def get_user_chat_summary(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户聊天摘要
        
        Args:
            user_id: 用户ID
            
        Returns:
            聊天摘要信息
        """
        if not self.db_service or not self.db_service.is_connected():
            return {"error": "数据库未连接"}
        
        try:
            return self.db_service.get_user_chat_summary(user_id, days_back=7)
        except Exception as e:
            return {"error": f"获取摘要失败: {str(e)}"}

    def search_user_messages(self, user_id: str, keyword: str, limit: int = 5) -> list:
        """
        搜索用户消息
        
        Args:
            user_id: 用户ID
            keyword: 搜索关键词
            limit: 最大结果数
            
        Returns:
            匹配的消息列表
        """
        if not self.db_service or not self.db_service.is_connected():
            return []
        
        try:
            return self.db_service.search_user_messages(user_id, keyword, limit)
        except Exception as e:
            print(f"搜索用户消息失败: {str(e)}")
            return []

    def get_recent_interactions(self, user_id: str, hours_back: int = 24) -> list:
        """
        获取用户最近的互动记录
        
        Args:
            user_id: 用户ID
            hours_back: 回溯小时数
            
        Returns:
            最近的互动记录
        """
        if not self.db_service or not self.db_service.is_connected():
            return []
        
        try:
            return self.db_service.get_recent_interactions(user_id, hours_back)
        except Exception as e:
            print(f"获取最近互动失败: {str(e)}")
            return []

    def get_historical_context_for_weight(self, user_id: str) -> str:
        """
        获取用户历史上下文（专门用于权重评估）
        
        Args:
            user_id: 用户ID
            
        Returns:
            历史上下文字符串
        """
        try:
            # 标准化用户ID
            normalized_user_id = str(user_id).strip()
            
            # 从配置获取参数
            history_config = self.config.get("history", {})
            max_messages = history_config.get("max_messages", 20)
            hours_back = history_config.get("hours_back", 72)
            min_length = history_config.get("min_message_length", 5)
            
            # 转换小时数为天数
            days_back = max(1, hours_back // 24)
            
            # 获取历史消息，排除已处理的
            processed_message_ids = self.get_processed_message_ids(normalized_user_id)
            history_messages = self.db_service.get_user_chat_history(
                user_id=normalized_user_id,
                limit=max_messages,
                days_back=days_back,
                exclude_message_ids=list(processed_message_ids)
            )
            
            # 过滤掉太短的消息
            filtered_messages = [
                msg for msg in history_messages 
                if msg["content"] and len(msg["content"].strip()) >= min_length
            ]
            
            if not filtered_messages:
                return ""
            
            # 构建上下文
            contexts = []
            contexts.append(f"用户 {normalized_user_id} 的历史对话:")
            
            # 使用配置中的最大消息数
            for msg in filtered_messages:
                timestamp = msg["datetime"]
                content = msg["content"]  # 不限制长度，让配置控制
                if isinstance(timestamp, datetime):
                    time_str = timestamp.strftime('%m-%d %H:%M')
                else:
                    time_str = str(timestamp)
                contexts.append(f"[{time_str}] {content}")
            
            return "\n".join(contexts)
            
        except Exception as e:
            logger.error(f"获取权重评估历史上下文失败: {str(e)}")
            return ""