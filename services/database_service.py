"""
数据库服务 - 连接主程序数据库获取历史聊天记录
"""

import os
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from peewee import SqliteDatabase, DoesNotExist
from src.common.logger import get_logger

logger = get_logger("impression_affection_database")


class DatabaseService:
    """数据库服务 - 安全地访问主程序数据库"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_config = config.get("database", {})
        self.db = None
        self._init_database()

    @staticmethod
    def normalize_user_id(user_id: Any) -> str:
        """
        标准化用户ID格式，确保一致性
        
        Args:
            user_id: 原始用户ID（可能是数字、字符串等）
            
        Returns:
            标准化后的用户ID字符串
        """
        if user_id is None:
            return ""
        
        # 转换为字符串并去除空白
        user_id_str = str(user_id).strip()
        
        # 如果是纯数字，保持原样
        if user_id_str.isdigit():
            return user_id_str
        
        # 移除可能的前缀和后缀
        # 移除常见的QQ号前缀
        if user_id_str.startswith("qq_"):
            user_id_str = user_id_str[3:]
        elif user_id_str.startswith("QQ:"):
            user_id_str = user_id_str[3:]
        elif user_id_str.startswith("U:"):
            user_id_str = user_id_str[2:]
        
        # 移除花括号、方括号等
        user_id_str = user_id_str.strip("{}[]()")
        
        return user_id_str

    def verify_user_id_match(self, db_user_id: Any, target_user_id: Any) -> bool:
        """
        验证数据库中的用户ID是否与目标用户ID匹配
        
        Args:
            db_user_id: 数据库中的用户ID
            target_user_id: 目标用户ID
            
        Returns:
            是否匹配
        """
        normalized_db_id = self.normalize_user_id(db_user_id)
        normalized_target_id = self.normalize_user_id(target_user_id)
        
        # 严格匹配
        if normalized_db_id == normalized_target_id:
            return True
        
        # 如果都是数字，进行数字比较
        if normalized_db_id.isdigit() and normalized_target_id.isdigit():
            try:
                return int(normalized_db_id) == int(normalized_target_id)
            except ValueError:
                pass
        
        return False

    def _init_database(self):
        """初始化数据库连接"""
        try:
            # 获取主程序数据库路径
            db_path = self._get_main_db_path()
            
            if not os.path.exists(db_path):
                logger.error(f"主程序数据库不存在: {db_path}")
                return

            # 连接数据库（只读模式）
            self.db = SqliteDatabase(
                db_path,
                pragmas={
                    "journal_mode": "wal",
                    "cache_size": -64 * 1000,
                    "foreign_keys": 1,
                    "busy_timeout": 1000,
                    "query_only": 1,  # 只读模式，确保不会修改主程序数据
                },
            )
            
            logger.info(f"成功连接主程序数据库: {db_path}")
            
        except Exception as e:
            logger.error(f"连接主程序数据库失败: {str(e)}")
            self.db = None

    def _get_main_db_path(self) -> str:
        """获取主程序数据库路径"""
        # 默认路径：项目根目录下的 data/MaiBot.db
        default_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "MaiBot.db")
        )
        
        # 支持配置自定义路径
        custom_path = self.db_config.get("main_db_path", "")
        if custom_path and os.path.isabs(custom_path):
            return custom_path
        elif custom_path:
            # 相对路径，相对于插件目录
            return os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", custom_path)
            )
        
        return default_path

    def is_connected(self) -> bool:
        """检查数据库是否已连接"""
        return self.db is not None

    def get_user_chat_history(
        self, 
        user_id: str, 
        limit: int = 20,
        days_back: int = 30,
        exclude_message_ids: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        获取用户历史聊天记录（严格过滤，确保只包含目标用户的消息）
        
        Args:
            user_id: 用户ID
            limit: 最大消息数量
            days_back: 回溯天数
            exclude_message_ids: 要排除的消息ID列表（用于去重）
            
        Returns:
            消息列表，每个消息包含时间、内容等信息
        """
        if not self.is_connected():
            logger.warning("数据库未连接，无法获取历史记录")
            return []

        # 标准化用户ID
        normalized_user_id = self.normalize_user_id(user_id)
        exclude_message_ids = exclude_message_ids or []
        
        try:
            # 计算时间范围
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            start_timestamp = start_time.timestamp()
            end_timestamp = end_time.timestamp()

            # 构建查询，直接使用真实用户ID过滤
            query_conditions = [
                "user_id = ?",
                "user_id IS NOT NULL",
                "user_id != ''",
                "time >= ?",
                "time <= ?",
                "(processed_plain_text IS NOT NULL OR display_message IS NOT NULL)"
            ]
            
            # 如果有要排除的消息ID，添加排除条件
            query_params = [normalized_user_id, start_timestamp, end_timestamp]
            if exclude_message_ids:
                # 确保排除的消息ID有效
                valid_exclude_ids = [msg_id for msg_id in exclude_message_ids if msg_id and msg_id.strip()]
                if valid_exclude_ids:
                    placeholders = ",".join(["?" for _ in valid_exclude_ids])
                    query_conditions.append(f"message_id NOT IN ({placeholders})")
                    query_params.extend(valid_exclude_ids)
                    logger.debug(f"数据库查询排除 {len(valid_exclude_ids)} 个已处理消息ID")
                    # 添加调试日志确认SQL构建正确
                    logger.debug(f"排除条件SQL: message_id NOT IN ({placeholders})")
                    logger.debug(f"排除参数: {valid_exclude_ids}")
                else:
                    logger.warning(f"提供的排除消息ID无效: {exclude_message_ids}")
            else:
                logger.debug(f"无排除消息ID，查询所有消息")
            
            # 从配置获取查询限制
            max_query_limit = self.config.get("history", {}).get("max_messages", 50)
            actual_limit = min(limit, max_query_limit)
            
            query = f"""
            SELECT 
                message_id,
                time,
                processed_plain_text,
                display_message,
                chat_info_user_nickname,
                chat_info_group_name,
                chat_info_platform,
                chat_info_user_id,
                user_id
            FROM messages 
            WHERE {' AND '.join(query_conditions)}
            ORDER BY time DESC 
            LIMIT ?
            """
            
            query_params.append(actual_limit)
            
            # 调试：显示实际执行的SQL和参数
            logger.debug(f"执行SQL查询: {query}")
            logger.debug(f"查询参数: {query_params}")
            
            cursor = self.db.execute_sql(query, query_params)
            rows = cursor.fetchall()
            
            # 调试：显示查询结果统计
            logger.debug(f"数据库查询返回 {len(rows)} 条原始记录")

            # 转换为字典格式，并进行验证
            messages = []
            mismatch_count = 0
            for row in rows:
                db_user_id = row[8]  # user_id (真实发言人QQ号)
                
                # 验证：确保消息确实属于目标用户
                if not self.verify_user_id_match(db_user_id, normalized_user_id):
                    mismatch_count += 1
                    logger.warning(f"跳过不匹配消息: 数据库用户ID {db_user_id} != 目标用户ID {normalized_user_id}")
                    continue
                
                message_time = datetime.fromtimestamp(row[1])
                
                # 优先使用 processed_plain_text，其次使用 display_message
                content = row[2] if row[2] else row[3]
                
                # 验证消息内容不为空且有意义
                if not content or len(content.strip()) < 2:
                    continue
                
                # 生成消息内容的哈希值，用于后续去重（与其他服务保持一致）
                normalized_content = content.strip().lower()
                content_hash = hashlib.md5(normalized_content.encode('utf-8')).hexdigest()
                
                messages.append({
                    "message_id": row[0],
                    "time": row[1],
                    "datetime": message_time,
                    "content": content.strip(),
                    "nickname": row[4],
                    "group_name": row[5],
                    "platform": row[6],
                    "verified_user_id": db_user_id,
                    "normalized_user_id": normalized_user_id,
                    "content_hash": content_hash
                })

            logger.info(f"用户 {normalized_user_id} 历史消息获取统计: 总查询 {len(rows)} 条，有效消息 {len(messages)} 条，排除消息ID {len(exclude_message_ids) if exclude_message_ids else 0} 条，不匹配消息 {mismatch_count} 条")
            
            # 调试：输出前几条消息的用户ID进行验证
            if len(messages) > 0:
                for i, msg in enumerate(messages[:3]):
                    logger.debug(f"消息 {i+1}: 内容={msg.get('content', '')[:50]}...")
            return messages

        except Exception as e:
            logger.error(f"获取用户历史记录失败: {str(e)}")
            return []

    def get_user_chat_summary(
        self, 
        user_id: str, 
        days_back: int = 7
    ) -> Dict[str, Any]:
        """
        获取用户聊天摘要信息
        
        Args:
            user_id: 用户ID
            days_back: 回溯天数
            
        Returns:
            包含聊天统计信息的字典
        """
        if not self.is_connected():
            return {"error": "数据库未连接"}

        # 标准化用户ID
        normalized_user_id = self.normalize_user_id(user_id)
        
        try:
            # 计算时间范围
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            start_timestamp = start_time.timestamp()
            end_timestamp = end_time.timestamp()

            # 获取基本统计信息 - 使用更严格的用户匹配
            query = """
            SELECT 
                COUNT(*) as total_messages,
                MIN(time) as first_message_time,
                MAX(time) as last_message_time,
                AVG(LENGTH(COALESCE(processed_plain_text, display_message))) as avg_message_length
            FROM messages 
            WHERE chat_info_user_id = ? 
                AND chat_info_user_id IS NOT NULL
                AND chat_info_user_id != ''
                AND time >= ? 
                AND time <= ?
                AND (processed_plain_text IS NOT NULL OR display_message IS NOT NULL)
            """

            cursor = self.db.execute_sql(query, (normalized_user_id, start_timestamp, end_timestamp))
            row = cursor.fetchone()

            if not row or row[0] == 0:
                return {"message": "指定时间范围内无聊天记录"}

            # 获取最活跃的群组
            group_query = """
            SELECT 
                chat_info_group_name,
                COUNT(*) as message_count
            FROM messages 
            WHERE chat_info_user_id = ? 
                AND chat_info_user_id IS NOT NULL
                AND chat_info_user_id != ''
                AND time >= ? 
                AND time <= ?
                AND chat_info_group_name IS NOT NULL
                AND chat_info_group_name != ''
            GROUP BY chat_info_group_name
            ORDER BY message_count DESC
            LIMIT 5
            """

            cursor = self.db.execute_sql(group_query, (normalized_user_id, start_timestamp, end_timestamp))
            group_rows = cursor.fetchall()

            active_groups = [
                {"name": row[0], "message_count": row[1]} 
                for row in group_rows
            ]

            return {
                "total_messages": row[0],
                "first_message_time": datetime.fromtimestamp(row[1]) if row[1] else None,
                "last_message_time": datetime.fromtimestamp(row[2]) if row[2] else None,
                "avg_message_length": round(row[3], 1) if row[3] else 0,
                "active_groups": active_groups,
                "days_analyzed": days_back,
                "normalized_user_id": normalized_user_id
            }

        except Exception as e:
            logger.error(f"获取用户聊天摘要失败: {str(e)}")
            return {"error": f"获取摘要失败: {str(e)}"}

    def search_user_messages(
        self, 
        user_id: str, 
        keyword: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        搜索用户包含关键词的消息
        
        Args:
            user_id: 用户ID
            keyword: 搜索关键词
            limit: 最大结果数量
            
        Returns:
            匹配的消息列表
        """
        if not self.is_connected():
            return []

        # 标准化用户ID
        normalized_user_id = self.normalize_user_id(user_id)
        
        try:
            query = """
            SELECT 
                message_id,
                time,
                processed_plain_text,
                display_message,
                chat_info_group_name,
                chat_info_user_id
            FROM messages 
            WHERE chat_info_user_id = ? 
                AND chat_info_user_id IS NOT NULL
                AND chat_info_user_id != ''
                AND (processed_plain_text LIKE ? OR display_message LIKE ?)
            ORDER BY time DESC 
            LIMIT ?
            """

            search_pattern = f"%{keyword}%"
            cursor = self.db.execute_sql(query, (normalized_user_id, search_pattern, search_pattern, limit))
            rows = cursor.fetchall()

            messages = []
            for row in rows:
                db_user_id = row[5]  # chat_info_user_id
                
                # 双重验证消息归属
                if not self.verify_user_id_match(db_user_id, normalized_user_id):
                    continue
                    
                message_time = datetime.fromtimestamp(row[1])
                content = row[2] if row[2] else row[3]
                
                # 验证内容不为空
                if not content or len(content.strip()) < 2:
                    continue
                
                messages.append({
                    "message_id": row[0],
                    "time": row[1],
                    "datetime": message_time,
                    "content": content.strip(),
                    "group_name": row[4],
                    "verified_user_id": db_user_id,
                    "normalized_user_id": normalized_user_id
                })

            logger.info(f"搜索到用户 {normalized_user_id} 包含关键词 '{keyword}' 的 {len(messages)} 条消息")
            return messages

        except Exception as e:
            logger.error(f"搜索用户消息失败: {str(e)}")
            return []

    def get_recent_interactions(
        self, 
        user_id: str, 
        hours_back: int = 24
    ) -> List[Dict[str, Any]]:
        """
        获取用户最近的互动记录
        
        Args:
            user_id: 用户ID
            hours_back: 回溯小时数
            
        Returns:
            最近的互动记录
        """
        if not self.is_connected():
            return []

        # 标准化用户ID
        normalized_user_id = self.normalize_user_id(user_id)
        
        try:
            # 计算时间范围
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours_back)
            start_timestamp = start_time.timestamp()
            end_timestamp = end_time.timestamp()

            query = """
            SELECT 
                message_id,
                time,
                processed_plain_text,
                display_message,
                chat_info_group_name,
                chat_info_platform,
                chat_info_user_id
            FROM messages 
            WHERE chat_info_user_id = ? 
                AND chat_info_user_id IS NOT NULL
                AND chat_info_user_id != ''
                AND time >= ? 
                AND time <= ?
                AND (processed_plain_text IS NOT NULL OR display_message IS NOT NULL)
            ORDER BY time DESC 
            LIMIT 50
            """

            cursor = self.db.execute_sql(query, (normalized_user_id, start_timestamp, end_timestamp))
            rows = cursor.fetchall()

            interactions = []
            for row in rows:
                db_user_id = row[6]  # chat_info_user_id
                
                # 双重验证用户ID匹配
                if not self.verify_user_id_match(db_user_id, normalized_user_id):
                    continue
                    
                message_time = datetime.fromtimestamp(row[1])
                content = row[2] if row[2] else row[3]
                
                # 验证内容不为空
                if not content or len(content.strip()) < 2:
                    continue
                
                interactions.append({
                    "message_id": row[0],
                    "time": row[1],
                    "datetime": message_time,
                    "content": content.strip(),
                    "group_name": row[4],
                    "platform": row[5],
                    "hours_ago": (end_time - message_time).total_seconds() / 3600,
                    "verified_user_id": db_user_id,
                    "normalized_user_id": normalized_user_id
                })

            return interactions

        except Exception as e:
            logger.error(f"获取最近互动失败: {str(e)}")
            return []

    def debug_user_ids(self, user_id: str, limit: int = 10) -> List[str]:
        """
        调试方法：检查数据库中相关的用户ID格式
        
        Args:
            user_id: 用户ID
            limit: 检查的消息数量
            
        Returns:
            找到的用户ID列表
        """
        if not self.is_connected():
            return []
        
        try:
            normalized_user_id = self.normalize_user_id(user_id)
            
            # 查询包含该用户ID的所有消息
            query = """
            SELECT DISTINCT chat_info_user_id
            FROM messages 
            WHERE chat_info_user_id LIKE ? 
            LIMIT ?
            """
            
            cursor = self.db.execute_sql(query, (f"%{normalized_user_id}%", limit))
            rows = cursor.fetchall()
            
            user_ids = [row[0] for row in rows]
            logger.info(f"用户ID调试 - 目标ID: {normalized_user_id}, 找到的ID: {user_ids}")
            
            return user_ids
            
        except Exception as e:
            logger.error(f"调试用户ID失败: {str(e)}")
            return []

    def get_main_message_id(self, user_id: str, timestamp: float, tolerance: float = 1.0) -> Optional[str]:
        """
        从主程序数据库获取实际的message_id
        
        Args:
            user_id: 用户ID
            timestamp: 消息时间戳
            tolerance: 时间容差（秒）
            
        Returns:
            实际的message_id，如果找不到返回None
        """
        if not self.is_connected():
            logger.warning("数据库未连接，无法获取message_id")
            return None
        
        # 检查 timestamp 是否为 None
        if timestamp is None:
            logger.warning(f"时间戳为空，无法查询消息ID (用户: {user_id})")
            return None
        
        try:
            # 标准化用户ID
            normalized_user_id = self.normalize_user_id(user_id)
            
            # 先尝试精确匹配（较小容差）
            for current_tolerance in [1.0, 5.0, 10.0, 30.0, 60.0]:
                query = """
                SELECT message_id, time 
                FROM messages 
                WHERE user_id = ? 
                    AND time >= ? 
                    AND time <= ?
                ORDER BY ABS(time - ?) ASC
                LIMIT 1
                """
                
                cursor = self.db.execute_sql(query, (
                    normalized_user_id, 
                    timestamp - current_tolerance, 
                    timestamp + current_tolerance,
                    timestamp
                ))
                row = cursor.fetchone()
                
                if row:
                    message_id = row[0]
                    actual_time = row[1]
                    time_diff = abs(actual_time - timestamp)
                    logger.info(f"获取到主程序message_id: {message_id} (用户: {normalized_user_id}, 查询时间: {timestamp}, 实际时间: {actual_time}, 时间差: {time_diff:.2f}秒, 容差: {current_tolerance}秒)")
                    return message_id
                else:
                    logger.debug(f"容差 {current_tolerance} 秒内未找到消息记录 (用户: {normalized_user_id}, 时间: {timestamp})")
            
            # 如果所有容差都找不到，尝试获取最新的消息ID
            logger.warning(f"所有容差范围内都未找到匹配消息，尝试获取用户最新消息 (用户: {normalized_user_id})")
            query = """
            SELECT message_id, time 
            FROM messages 
            WHERE user_id = ? 
                AND time <= ?
            ORDER BY time DESC 
            LIMIT 1
            """
            
            cursor = self.db.execute_sql(query, (
                normalized_user_id, 
                timestamp + 300  # 允许未来5分钟内的消息
            ))
            row = cursor.fetchone()
            
            if row:
                message_id = row[0]
                actual_time = row[1]
                time_diff = abs(actual_time - timestamp)
                logger.warning(f"使用用户最新消息ID: {message_id} (用户: {normalized_user_id}, 查询时间: {timestamp}, 实际时间: {actual_time}, 时间差: {time_diff:.2f}秒)")
                return message_id
            else:
                logger.error(f"用户无任何消息记录 (用户: {normalized_user_id})")
                return None
                
        except Exception as e:
            logger.error(f"获取主程序message_id失败: {str(e)}")
            return None
            return None

    

    def close(self):
        """关闭数据库连接"""
        if self.db:
            self.db.close()
            self.db = None
            logger.info("数据库连接已关闭")