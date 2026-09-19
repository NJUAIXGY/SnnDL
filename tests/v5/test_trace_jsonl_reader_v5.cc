// Direct unit test for the bounded JSONL record reader used by the v5 trace
// replay sources.  The reader has no SST dependency, so this target links the
// reader translation unit alone.
#include "v5/trace/TraceJsonlReader.h"

#include <nlohmann/json.hpp>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>

using SST::SnnDL::v5::TraceJsonlReader;
using SST::SnnDL::v5::kTraceMaxLineBytesDefault;
using SST::SnnDL::v5::kTraceMaxLineBytesMinimum;
using SST::SnnDL::v5::normalizeTraceMaxLineBytes;

namespace {

constexpr std::uint64_t kGenerousLimit = 1024;

// One parsed record per line, in file order, followed by End.
static std::string recordLine(std::uint64_t stream_sequence) {
    return nlohmann::json{{"schema_version", "snndl-materialized-request/v1"},
                          {"stream_sequence", stream_sequence}}.dump();
}

// A valid JSON object padded to exactly `length` bytes, so the byte limit can
// be exercised at and just past the boundary.  Trailing whitespace is accepted
// by the parser and is not counted as part of the record.
static std::string exactLengthRecord(std::size_t length) {
    const std::string base = "{\"a\":1}";
    assert(length >= base.size());
    return base + std::string(length - base.size(), ' ');
}

static void test_reads_every_record_in_order() {
    std::istringstream stream(recordLine(1) + "\n" + recordLine(2) + "\n" + recordLine(3) + "\n");
    TraceJsonlReader reader;
    nlohmann::json value;
    std::string error;
    for (std::uint64_t sequence = 1; sequence <= 3; ++sequence) {
        assert(reader.next(stream, kGenerousLimit, value, error) == TraceJsonlReader::Status::Record);
        assert(error.empty());
        assert(value.is_object());
        assert(value.at("schema_version").get<std::string>() == "snndl-materialized-request/v1");
        assert(value.at("stream_sequence").get<std::uint64_t>() == sequence);
    }
    assert(reader.next(stream, kGenerousLimit, value, error) == TraceJsonlReader::Status::End);
    assert(stream.eof());
}

// Blank and whitespace-only lines, including CRLF remnants, never reach a
// caller and never produce a parse diagnostic.
static void test_skips_blank_lines_between_and_after_records() {
    std::istringstream stream("\n" + recordLine(1) + "\n   \t \n\r\n" + recordLine(2) + "\n\n \n");
    TraceJsonlReader reader;
    nlohmann::json value;
    std::string error;
    assert(reader.next(stream, kGenerousLimit, value, error) == TraceJsonlReader::Status::Record);
    assert(value.at("stream_sequence").get<std::uint64_t>() == 1);
    assert(reader.next(stream, kGenerousLimit, value, error) == TraceJsonlReader::Status::Record);
    assert(value.at("stream_sequence").get<std::uint64_t>() == 2);
    assert(reader.next(stream, kGenerousLimit, value, error) == TraceJsonlReader::Status::End);
}

static void test_empty_stream_ends_immediately() {
    std::istringstream stream("");
    TraceJsonlReader reader;
    nlohmann::json value;
    std::string error;
    assert(reader.next(stream, kGenerousLimit, value, error) == TraceJsonlReader::Status::End);
}

// The newline terminator is not part of the record, so a line of exactly the
// limit is accepted and one byte more is rejected.
static void test_size_limit_is_inclusive() {
    {
        std::istringstream stream(exactLengthRecord(24) + "\n");
        TraceJsonlReader reader;
        nlohmann::json value;
        std::string error;
        assert(reader.next(stream, 24, value, error) == TraceJsonlReader::Status::Record);
    }
    {
        std::istringstream stream(exactLengthRecord(25) + "\n" + recordLine(1) + "\n");
        TraceJsonlReader reader;
        nlohmann::json value;
        std::string error;
        assert(reader.next(stream, 24, value, error) == TraceJsonlReader::Status::Oversize);
        // A rejected line is terminal: the reader must not resume and silently
        // drop the record that follows it.
        assert(reader.next(stream, 24, value, error) == TraceJsonlReader::Status::End);
    }
}

// The last line of a shard may have no terminator at all; it is still a record.
static void test_final_line_without_newline_is_still_read() {
    std::istringstream stream(recordLine(1) + "\n" + recordLine(2));
    TraceJsonlReader reader;
    nlohmann::json value;
    std::string error;
    assert(reader.next(stream, kGenerousLimit, value, error) == TraceJsonlReader::Status::Record);
    assert(reader.next(stream, kGenerousLimit, value, error) == TraceJsonlReader::Status::Record);
    assert(value.at("stream_sequence").get<std::uint64_t>() == 2);
    assert(reader.next(stream, kGenerousLimit, value, error) == TraceJsonlReader::Status::End);
}

// A record cut off mid-object, and one with trailing content, are reported as
// parse failures carrying the parser diagnostic and are terminal as well.
static void test_truncated_and_malformed_lines_report_parse_error() {
    {
        std::istringstream stream("{\"a\": 1");
        TraceJsonlReader reader;
        nlohmann::json value;
        std::string error;
        assert(reader.next(stream, kGenerousLimit, value, error) == TraceJsonlReader::Status::ParseError);
        assert(!error.empty());
        assert(reader.next(stream, kGenerousLimit, value, error) == TraceJsonlReader::Status::End);
        // End must not repeat a stale diagnostic in the out parameter.
        assert(error.empty());
    }
    {
        // Same, but with a well-formed record behind the broken line: a parse
        // failure must stop the shard rather than skip ahead.
        std::istringstream stream("{\"a\": 1\n" + recordLine(2) + "\n");
        TraceJsonlReader reader;
        nlohmann::json value;
        std::string error;
        assert(reader.next(stream, kGenerousLimit, value, error) == TraceJsonlReader::Status::ParseError);
        assert(!error.empty());
        assert(reader.next(stream, kGenerousLimit, value, error) == TraceJsonlReader::Status::End);
        assert(!stream.eof());
    }
    {
        std::istringstream stream("{\"a\":1} trailing\n");
        TraceJsonlReader reader;
        nlohmann::json value;
        std::string error;
        assert(reader.next(stream, kGenerousLimit, value, error) == TraceJsonlReader::Status::ParseError);
        assert(!error.empty());
    }
}

// Record shape is the caller's contract: the reader hands over whatever single
// JSON value the line holds and leaves the schema check to the source.
static void test_non_object_document_is_handed_to_the_caller() {
    std::istringstream stream("42\n" + recordLine(1) + "\n");
    TraceJsonlReader reader;
    nlohmann::json value;
    std::string error;
    assert(reader.next(stream, kGenerousLimit, value, error) == TraceJsonlReader::Status::Record);
    assert(!value.is_object());
    assert(reader.next(stream, kGenerousLimit, value, error) == TraceJsonlReader::Status::Record);
    assert(value.is_object());
}

// The record-size bound is one contract-layer value shared by every replay
// source: the absent-parameter default stays pinned to the 1 MiB that both the
// component ELI tables and the Python trace_io.iter_jsonl default state, and an
// absurdly small request is raised to the floor instead of switching the bound
// off.  Sources must not restate either literal.
static void test_shared_line_limit_is_normalized_in_one_place() {
    assert(kTraceMaxLineBytesDefault == 1024 * 1024);
    assert(kTraceMaxLineBytesMinimum == 128);
    assert(normalizeTraceMaxLineBytes(kTraceMaxLineBytesDefault) == kTraceMaxLineBytesDefault);
    assert(normalizeTraceMaxLineBytes(kTraceMaxLineBytesMinimum) == kTraceMaxLineBytesMinimum);
    assert(normalizeTraceMaxLineBytes(1) == kTraceMaxLineBytesMinimum);
    assert(normalizeTraceMaxLineBytes(0) == kTraceMaxLineBytesMinimum);
    assert(normalizeTraceMaxLineBytes(4096) == 4096);
}

}  // namespace

int main() {
    test_reads_every_record_in_order();
    test_skips_blank_lines_between_and_after_records();
    test_empty_stream_ends_immediately();
    test_size_limit_is_inclusive();
    test_final_line_without_newline_is_still_read();
    test_truncated_and_malformed_lines_report_parse_error();
    test_non_object_document_is_handed_to_the_caller();
    test_shared_line_limit_is_normalized_in_one_place();
    return 0;
}
