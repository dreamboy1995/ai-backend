"""
业务错误码定义
用于统一前后端错误处理，确保错误信息可追溯、可国际化
"""


class ErrorCode:
    """业务错误码常量类"""
    
    # 认证相关 (10000-10099)
    INVALID_API_KEY = 10001
    TOKEN_EXPIRED = 10002
    TOKEN_INVALID = 10003
    
    # 模型调用相关 (10100-10199)
    MODEL_RATE_LIMIT = 10101      # 模型限流
    MODEL_TOKEN_TOO_LONG = 10102  # Token 超长
    MODEL_TIMEOUT = 10103         # 模型调用超时
    MODEL_NETWORK_ERROR = 10104   # 网络连接错误
    MODEL_SERVICE_ERROR = 10105   # 模型服务异常
    MODEL_INVALID_REQUEST = 10106 # 请求参数错误
    
    # 系统相关 (10200-10299)
    INTERNAL_ERROR = 10201        # 内部错误
    SERVICE_UNAVAILABLE = 10202   # 服务不可用


# 错误码到默认消息的映射
ERROR_MESSAGES = {
    ErrorCode.INVALID_API_KEY: "无效的 API Key",
    ErrorCode.TOKEN_EXPIRED: "Token 已过期",
    ErrorCode.TOKEN_INVALID: "无效的 Token",
    
    ErrorCode.MODEL_RATE_LIMIT: "模型调用频率超限，请稍后重试",
    ErrorCode.MODEL_TOKEN_TOO_LONG: "输入内容过长，请缩短后重试",
    ErrorCode.MODEL_TIMEOUT: "模型响应超时，请稍后重试",
    ErrorCode.MODEL_NETWORK_ERROR: "网络连接失败，请检查网络",
    ErrorCode.MODEL_SERVICE_ERROR: "模型服务异常，请稍后重试",
    ErrorCode.MODEL_INVALID_REQUEST: "请求参数错误",
    
    ErrorCode.INTERNAL_ERROR: "服务器内部错误",
    ErrorCode.SERVICE_UNAVAILABLE: "服务暂时不可用",
}


def get_error_message(code: int, default_msg: str = None) -> str:
    """获取错误码对应的默认消息"""
    return ERROR_MESSAGES.get(code, default_msg or "未知错误")
