#include "utils/Logger.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>
#include <chrono>
#include <iomanip>

namespace NeuronMapping {

// === 单例实现 ===

Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}

bool Logger::setOutputFile(const std::string& filename) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    if (file_stream_ && file_stream_->is_open()) {
        file_stream_->close();
    }
    
    file_stream_ = std::make_unique<std::ofstream>(filename, std::ios::app);
    return file_stream_ && file_stream_->is_open();
}

void Logger::log(LogLevel level, const std::string& message) {
    if (level < current_level_) {
        return;
    }
    
    std::string formatted = formatMessage(level, message);
    
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    // 输出到控制台
    if (console_output_) {
        std::cout << formatted << std::endl;
    }
    
    // 输出到文件
    if (file_stream_ && file_stream_->is_open()) {
        *file_stream_ << formatted << std::endl;
        file_stream_->flush();
    }
}

void Logger::debug(const std::string& message) {
    log(LogLevel::DEBUG, message);
}

void Logger::info(const std::string& message) {
    log(LogLevel::INFO, message);
}

void Logger::warning(const std::string& message) {
    log(LogLevel::WARNING, message);
}

void Logger::error(const std::string& message) {
    log(LogLevel::ERROR, message);
}

void Logger::critical(const std::string& message) {
    log(LogLevel::CRITICAL, message);
}

void Logger::flush() {
    std::lock_guard<std::mutex> lock(log_mutex_);
    if (file_stream_ && file_stream_->is_open()) {
        file_stream_->flush();
    }
    std::cout.flush();
}

// === 私有方法 ===

Logger::~Logger() {
    close();
}

void Logger::close() {
    std::lock_guard<std::mutex> lock(log_mutex_);
    if (file_stream_ && file_stream_->is_open()) {
        file_stream_->close();
    }
}

std::string Logger::formatMessage(LogLevel level, const std::string& message) const {
    std::ostringstream oss;
    
    // 时间戳
    if (timestamp_enabled_) {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        oss << "[" << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        oss << "." << std::setfill('0') << std::setw(3) << ms.count() << "] ";
    }
    
    // 线程ID
    if (thread_id_enabled_) {
        oss << "[Thread-" << getCurrentThreadId() << "] ";
    }
    
    // 日志级别
    oss << "[" << levelToString(level) << "] ";
    
    oss << message;
    return oss.str();
}

std::string Logger::levelToString(LogLevel level) const {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO: return "INFO";
        case LogLevel::WARNING: return "WARN";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::CRITICAL: return "CRITICAL";
        default: return "UNKNOWN";
    }
}

std::string Logger::getCurrentTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

std::string Logger::getCurrentThreadId() const {
    std::ostringstream oss;
    oss << std::this_thread::get_id();
    return oss.str();
}

} // namespace NeuronMapping