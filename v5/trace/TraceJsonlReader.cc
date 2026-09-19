#include "TraceJsonlReader.h"

#include <exception>

namespace SST { namespace SnnDL { namespace v5 {

TraceJsonlReader::Status TraceJsonlReader::next(std::istream& stream, std::uint64_t max_line_bytes,
                                                nlohmann::json& value, std::string& error) {
    error.clear();
    if (stopped_) return Status::End;
    std::string line;
    while (std::getline(stream, line)) {
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
        if (line.size() > max_line_bytes) {
            stopped_ = true;
            return Status::Oversize;
        }
        try {
            value = nlohmann::json::parse(line);
        } catch (const std::exception& parse_error) {
            stopped_ = true;
            error = parse_error.what();
            return Status::ParseError;
        }
        return Status::Record;
    }
    return Status::End;
}

}}}
