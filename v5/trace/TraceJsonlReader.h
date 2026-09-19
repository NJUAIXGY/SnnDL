#ifndef SST_SNN_DL_V5_TRACE_JSONL_READER_H
#define SST_SNN_DL_V5_TRACE_JSONL_READER_H

#include <nlohmann/json.hpp>

#include <cstdint>
#include <istream>
#include <string>

namespace SST { namespace SnnDL { namespace v5 {

// Shared bounded-record limit for the trace replay sources.  The value belongs
// to this contract layer, not to a component: when the max_line_bytes parameter
// is absent every source takes this default and no source repeats the literal.
// It mirrors the Python trace_io.iter_jsonl default of 1 MiB, and the floor
// keeps a mistyped parameter from switching the reader off entirely.
constexpr std::uint64_t kTraceMaxLineBytesDefault = 1024 * 1024;
constexpr std::uint64_t kTraceMaxLineBytesMinimum = 128;

constexpr std::uint64_t normalizeTraceMaxLineBytes(std::uint64_t requested) {
    return requested < kTraceMaxLineBytesMinimum ? kTraceMaxLineBytesMinimum : requested;
}

// Trace-runtime failure vocabulary.  Every fatal raised by a v5/trace component
// starts with one of these stable codes, so acceptance tooling can classify a
// failure without substring-guessing prose.  The existing message text is kept.
//   TRACE-CONFIG:    component parameters or links are unusable
//   TRACE-IO:        a declared trace or evidence file cannot be opened
//   TRACE-SCHEMA:    a record is not the declared schema, shape, or size
//   TRACE-IDENTITY:  duplicate, foreign, or empty request/event identity
//   TRACE-ORDER:     stream_sequence is not strictly increasing
//   TRACE-LIMIT:     a bounded record or resource limit was exceeded
//   TRACE-ADMISSION: a request or injection was rejected or answered wrongly
//   TRACE-CONTROL:   the replay-coordinator handshake is out of protocol
//   TRACE-DRAIN:     the component ended before its replay drained
// A reader Status of Oversize maps to TRACE-LIMIT and ParseError to
// TRACE-SCHEMA; the reader itself stays diagnostic-free.

// Bounded JSONL record reader shared by the trace replay sources.  It owns the
// line loop only: blank lines are skipped, a record may not exceed the byte
// limit, and every other line must parse as exactly one JSON object.  Schema
// version checks, field extraction, monotonicity guards, and the component
// fatal messages stay with the callers so their diagnostics are unchanged.
class TraceJsonlReader final {
public:
    enum class Status {
        Record,      // value holds the parsed object
        Oversize,    // line exceeded max_line_bytes; reported as End afterwards
        ParseError,  // error holds the parser diagnostic; reported as End afterwards
        End,         // stream exhausted or a previous line was rejected
    };

    TraceJsonlReader() = default;

    // Advances past blank lines and parses the next record from stream, which
    // must not exceed max_line_bytes.  A rejected line is terminal: the reader
    // never silently skips it and keeps the shard aligned with the caller.
    Status next(std::istream& stream, std::uint64_t max_line_bytes, nlohmann::json& value,
                std::string& error);

private:
    bool stopped_ = false;
};

}}}

#endif
