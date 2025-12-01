"""
数据库服务 - 连接主程序数据库获取历史聊天记录
"""

import os
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
        days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """
        获取用户历史聊天记录（严格过滤，确保只包含目标用户的消息）
        
        Args:
            user_id: 用户ID
            limit: 最大消息数量
            days_back: 回溯天数
            
        Returns:
            消息列表，每个消息包含时间、内容等信息
        """
        if not self.is_connected():
            logger.warning("数据库未连接，无法获取历史记录")
            return []

        try:
            # 计算时间范围
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            start_timestamp = start_time.timestamp()
            end_timestamp = end_time.timestamp()

            # 构建更严格的查询，确保只获取目标用户的消息
            query = f"""
            SELECT 
                message_id,
                time,
                processed_plain_text,
                display_message,
                chat_info_user_nickname,
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
            LIMIT ?
            """

            cursor = self.db.execute_sql(query, (user_id, start_timestamp, end_timestamp, limit))
            rows = cursor.fetchall()

            # 转换为字典格式，并进行二次验证
            messages = []
            for row in rows:
                # 验证消息归属
                if row[7] != user_id:  # chat_info_user_id
                    logger.debug(f"跳过不匹配消息: 用户ID {row[7]} != {user_id}")
                    continue
                
                message_time = datetime.fromtimestamp(row[1])
                
                # 优先使用 processed_plain_text，其次使用 display_message
                content = row[2] if row[2] else row[3]
                
                # 验证消息内容不为空且有意义
                if not content or len(content.strip()) < 2:
                    continue
                
                messages.append({
                    "message_id": row[0],
                    "time": row[1],
                    "datetime": message_time,
                    "content": content.strip(),
                    "nickname": row[4],
                    "group_name": row[5],
                    "platform": row[6],
                    "verified_user_id": row[7]  # 添加验证字段
                })

            logger.debug(f"用户 {user_id} 历史消息验证完成: {len(messages)} 条有效消息")
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

        try:
            # 计算时间范围
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            start_timestamp = start_time.timestamp()
            end_timestamp = end_time.timestamp()

            # 获取基本统计信息
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

            cursor = self.db.execute_sql(query, (user_id, start_timestamp, end_timestamp))
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

            cursor = self.db.execute_sql(group_query, (user_id, start_timestamp, end_timestamp))
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
                "days_analyzed": days_back
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
            cursor = self.db.execute_sql(query, (user_id, search_pattern, search_pattern, limit))
            rows = cursor.fetchall()

            messages = []
            for row in rows:
                # 验证消息归属
                if row[5] != user_id:  # chat_info_user_id
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
                    "verified_user_id": row[5]
                })

            logger.info(f"搜索到用户 {user_id} 包含关键词 '{keyword}' 的 {len(messages)} 条消息")
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

            cursor = self.db.execute_sql(query, (user_id, start_timestamp, end_timestamp))
            rows = cursor.fetchall()

            interactions = []
            for row in rows:
                # 验证用户ID匹配
                if row[6] != user_id:  # chat_info_user_id
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
                    "verified_user_id": row[6]
                })

            return interactions

        except Exception as e:
            logger.error(f"获取最近互动失败: {str(e)}")
            return []

    def close(self):
        """关闭数据库连接"""
        if self.db:
            self.db.close()
            self.db = None
            logger.info("数据库连接已关闭")