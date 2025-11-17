#include "utils/Logger.h"
#include <iostream>
#include <cassert>

using namespace NeuronMapping;

void testLoggerBasicFunctionality() {
    std::cout << "=== Testing Logger Basic Functionality ===\n";
    
    // 获取Logger实例
    Logger& logger = Logger::getInstance();
    
    // 测试基本日志输出
    logger.setLevel(LogLevel::DEBUG);
    logger.enableConsoleOutput(true);
    logger.enableTimestamp(true);
    logger.enableThreadId(false);
    
    std::cout << "Testing different log levels:\n";
    logger.debug("This is a debug message");
    logger.info("This is an info message");
    logger.warning("This is a warning message");
    logger.error("This is an error message");
    logger.critical("This is a critical message");
    
    // 测试宏
    std::cout << "\nTesting log macros:\n";
    LOG_DEBUG("Debug macro test");
    LOG_INFO("Info macro test");
    LOG_WARNING("Warning macro test");
    LOG_ERROR("Error macro test");
    LOG_CRITICAL("Critical macro test");
    
    // 测试日志级别过滤
    std::cout << "\nTesting log level filtering (set to WARNING):\n";
    logger.setLevel(LogLevel::WARNING);
    logger.debug("This debug should not appear");
    logger.info("This info should not appear");
    logger.warning("This warning should appear");
    logger.error("This error should appear");
    
    // 测试时间戳和线程ID
    std::cout << "\nTesting timestamp and thread ID:\n";
    logger.enableTimestamp(false);
    logger.enableThreadId(true);
    logger.info("Message with thread ID only");
    
    logger.enableTimestamp(true);
    logger.enableThreadId(true);
    logger.info("Message with both timestamp and thread ID");
    
    logger.flush();
    
    std::cout << "✓ Logger basic functionality test completed\n";
}

void testLoggerFileOutput() {
    std::cout << "\n=== Testing Logger File Output ===\n";
    
    Logger& logger = Logger::getInstance();
    
    // 设置文件输出
    std::string test_log_file = "/tmp/test_logger.log";
    bool file_set = logger.setOutputFile(test_log_file);
    assert(file_set && "Failed to set log file");
    
    logger.setLevel(LogLevel::INFO);
    logger.enableConsoleOutput(false);  // 仅输出到文件
    
    logger.info("This message should go to file only");
    logger.warning("Another message for the file");
    logger.flush();
    
    // 恢复控制台输出
    logger.enableConsoleOutput(true);
    logger.info("File output test completed");
    
    std::cout << "✓ Logger file output test completed\n";
    std::cout << "Check " << test_log_file << " for file output\n";
}

int main() {
    try {
        testLoggerBasicFunctionality();
        testLoggerFileOutput();
        
        std::cout << "\n✅ All Logger tests completed successfully!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n❌ Test failed with unknown exception!\n";
        return 1;
    }
}