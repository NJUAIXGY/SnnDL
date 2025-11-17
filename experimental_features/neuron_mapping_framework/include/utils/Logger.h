#ifndef NEURON_MAPPING_LOGGER_H
#define NEURON_MAPPING_LOGGER_H

#include "../core/Types.h"
#include <string>
#include <memory>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>
#include <mutex>

namespace NeuronMapping {

/**
 * @brief 线程安全的日志系统
 * 
 * 提供分级日志记录功能，支持文件和控制台输出。
 */
class Logger {
public:
    /**
     * @brief 获取全局日志实例
     * @return 日志实例引用
     */
    static Logger& getInstance();
    
    /**
     * @brief 设置日志级别
     * @param level 日志级别
     */
    void setLevel(LogLevel level) { current_level_ = level; }
    
    /**
     * @brief 获取当前日志级别
     * @return 日志级别
     */
    LogLevel getLevel() const { return current_level_; }
    
    /**
     * @brief 设置输出文件
     * @param filename 文件名
     * @return 是否成功设置
     */
    bool setOutputFile(const std::string& filename);
    
    /**
     * @brief 启用/禁用控制台输出
     * @param enable 是否启用
     */
    void enableConsoleOutput(bool enable) { console_output_ = enable; }
    
    /**
     * @brief 启用/禁用时间戳
     * @param enable 是否启用
     */
    void enableTimestamp(bool enable) { timestamp_enabled_ = enable; }
    
    /**
     * @brief 启用/禁用线程ID
     * @param enable 是否启用
     */
    void enableThreadId(bool enable) { thread_id_enabled_ = enable; }
    
    /**
     * @brief 记录调试信息
     * @param message 消息内容
     */
    void debug(const std::string& message);
    
    /**
     * @brief 记录普通信息
     * @param message 消息内容
     */
    void info(const std::string& message);
    
    /**
     * @brief 记录警告信息
     * @param message 消息内容
     */
    void warning(const std::string& message);
    
    /**
     * @brief 记录错误信息
     * @param message 消息内容
     */
    void error(const std::string& message);
    
    /**
     * @brief 记录严重错误信息
     * @param message 消息内容
     */
    void critical(const std::string& message);
    
    /**
     * @brief 记录指定级别的消息
     * @param level 日志级别
     * @param message 消息内容
     */
    void log(LogLevel level, const std::string& message);
    
    /**
     * @brief 刷新输出缓冲区
     */
    void flush();
    
    /**
     * @brief 关闭日志系统
     */
    void close();
    
    /**
     * @brief 格式化日志消息
     * @param level 日志级别
     * @param message 原始消息
     * @return 格式化后的消息
     */
    std::string formatMessage(LogLevel level, const std::string& message) const;

private:
    Logger() = default;
    ~Logger();
    
    // 禁用拷贝和赋值
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    void writeMessage(LogLevel level, const std::string& message);
    std::string levelToString(LogLevel level) const;
    std::string getCurrentTimestamp() const;
    std::string getCurrentThreadId() const;
    
    LogLevel current_level_ = LogLevel::INFO;
    bool console_output_ = true;
    bool timestamp_enabled_ = true;
    bool thread_id_enabled_ = false;
    
    std::unique_ptr<std::ofstream> file_stream_;
    mutable std::mutex log_mutex_;
};

// 便利宏定义
#define LOG_DEBUG(msg) Logger::getInstance().debug(msg)
#define LOG_INFO(msg) Logger::getInstance().info(msg)
#define LOG_WARNING(msg) Logger::getInstance().warning(msg)
#define LOG_ERROR(msg) Logger::getInstance().error(msg)
#define LOG_CRITICAL(msg) Logger::getInstance().critical(msg)

// 格式化日志宏
#define LOG_DEBUG_F(...) do { \
    std::ostringstream oss; \
    oss << __VA_ARGS__; \
    Logger::getInstance().debug(oss.str()); \
} while(0)

#define LOG_INFO_F(...) do { \
    std::ostringstream oss; \
    oss << __VA_ARGS__; \
    Logger::getInstance().info(oss.str()); \
} while(0)

#define LOG_WARNING_F(...) do { \
    std::ostringstream oss; \
    oss << __VA_ARGS__; \
    Logger::getInstance().warning(oss.str()); \
} while(0)

#define LOG_ERROR_F(...) do { \
    std::ostringstream oss; \
    oss << __VA_ARGS__; \
    Logger::getInstance().error(oss.str()); \
} while(0)

#define LOG_CRITICAL_F(...) do { \
    std::ostringstream oss; \
    oss << __VA_ARGS__; \
    Logger::getInstance().critical(oss.str()); \
} while(0)

} // namespace NeuronMapping

#endif // NEURON_MAPPING_LOGGER_H