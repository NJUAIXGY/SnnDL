// -*- c++ -*-
//
// Canonical file-side BCSR source contract.
//
// Routing and diagnostic/fallback weight reads must agree on the same metadata,
// row prefix, layout, and optional block-id validity mask.  This small helper
// keeps that contract in one place; normal weight traffic still goes through
// StandardMem and the simulated memory hierarchy.

#pragma once

#include "BcsrMeta.h"
#include "api/BcsrSourceContract.h"

#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace SST { namespace SnnDL {

struct BcsrFileSource {
    std::string path;
    BcsrMeta meta{};
    uint64_t file_size = 0;
    uint32_t requested_rows = 0;
    uint64_t fingerprint = 0;
    uint64_t content_fingerprint = 0;

    bool blockIdsPresent() const { return meta.blockids_offset != 0; }
    uint32_t blockRows() const { return meta.br ? meta.br : 16u; }
    uint32_t blockCols() const { return meta.bc ? meta.bc : 16u; }
    uint32_t indexBytes() const { return meta.idx_bytes ? meta.idx_bytes : 2u; }
    uint32_t valueBytes() const { return meta.val_bytes ? meta.val_bytes : 4u; }
    bool rowpack() const { return meta.layout_mode == "rowpack_v1"; }

    BcsrSourceIdentity identity() const {
        return BcsrSourceIdentity{fingerprint, content_fingerprint, file_size};
    }
};

inline uint64_t bcsrSourceFnv1a_(const std::string& text) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : text) {
        h ^= static_cast<uint64_t>(c);
        h *= 1099511628211ULL;
    }
    return h;
}

inline uint64_t bcsrSourceFnv1aUpdate_(uint64_t hash,
                                       const uint8_t* bytes,
                                       size_t count) {
    constexpr uint64_t kPrime = 1099511628211ULL;
    for (size_t i = 0; i < count; ++i) {
        hash ^= static_cast<uint64_t>(bytes[i]);
        hash *= kPrime;
    }
    return hash;
}

inline bool loadBcsrFileSource(const std::string& path,
                               uint32_t requested_rows,
                               BcsrFileSource& out,
                               std::string* error = nullptr,
                               bool compute_content_fingerprint = false) {
    auto fail = [&](const char* why) {
        if (error) *error = why ? std::string(why) : std::string();
        return false;
    };

    out = BcsrFileSource{};
    out.path = path;
    out.requested_rows = requested_rows;
    if (path.empty()) return fail("empty BCSR path");

    if (!parseBcsrMetaJsonFile(path + ".meta.json", out.meta)) {
        return fail("BCSR metadata is missing or malformed");
    }
    if (out.meta.rows == 0 || out.meta.cols == 0) return fail("BCSR metadata has no rows/cols");
    const uint32_t rows = requested_rows ? requested_rows : out.meta.rows;
    if (out.meta.rows < rows) return fail("BCSR backing rows are smaller than the requested model");
    if (out.meta.layout_mode.empty()) out.meta.layout_mode = "flat";

    std::ifstream fin(path, std::ios::binary);
    if (!fin.good()) return fail("BCSR data file cannot be opened");
    fin.seekg(0, std::ios::end);
    const std::streamoff size = fin.tellg();
    if (size <= 0) return fail("BCSR data file is empty");
    out.file_size = static_cast<uint64_t>(size);

    // A zero blockids offset means the optional validity mask is absent.  A
    // non-zero offset that lies outside the file is also treated as absent,
    // matching the generator's legacy contract.
    if (out.meta.blockids_offset && out.meta.total_blocks) {
        const uint64_t block_bytes = bcsrBytesPerBlock(out.blockRows(), out.blockCols(), out.valueBytes());
        if (out.meta.blockids_offset + static_cast<uint64_t>(out.meta.total_blocks) * block_bytes > out.file_size) {
            out.meta.blockids_offset = 0;
        }
    }

    std::string validation_error;
    if (!validateBcsrMetaAgainstFile(out.meta, out.file_size, rows, &validation_error)) {
        if (error) *error = validation_error;
        return false;
    }

    std::ifstream meta_file(path + ".meta.json");
    const std::string meta_text((std::istreambuf_iterator<char>(meta_file)),
                                std::istreambuf_iterator<char>());
    // The descriptor identity deliberately excludes the path.  A loader and
    // router may receive equivalent absolute/relative spellings, but they must
    // still agree on the metadata and file bytes.
    out.fingerprint = bcsrSourceFnv1a_(std::to_string(out.file_size) + "|" + meta_text);
    if (compute_content_fingerprint) {
        std::ifstream data(path, std::ios::binary);
        if (!data.good()) return fail("BCSR data file cannot be reopened for fingerprinting");
        uint64_t hash = bcsrSourceFnv1a_(meta_text + "|" + std::to_string(out.file_size));
        std::vector<uint8_t> buffer(64u * 1024u);
        while (data.good()) {
            data.read(reinterpret_cast<char*>(buffer.data()),
                      static_cast<std::streamsize>(buffer.size()));
            const std::streamsize got = data.gcount();
            if (got > 0) {
                hash = bcsrSourceFnv1aUpdate_(hash, buffer.data(), static_cast<size_t>(got));
            }
        }
        if (!data.eof()) return fail("BCSR data file fingerprint read failed");
        out.content_fingerprint = hash;
    }
    return true;
}

inline bool readBcsrRowptr(const BcsrFileSource& source,
                           uint32_t rows,
                           std::vector<uint32_t>& rowptr) {
    const uint32_t effective_rows = rows ? rows : source.meta.rows;
    if (effective_rows == 0 || source.meta.rowptr_offset >= source.file_size) return false;
    const uint32_t n_block_rows = bcsrNumBlockRows(effective_rows, source.blockRows());
    rowptr.assign(static_cast<size_t>(n_block_rows) + 1u, 0u);
    std::ifstream fin(source.path, std::ios::binary);
    if (!fin.good()) return false;
    fin.seekg(static_cast<std::streamoff>(source.meta.rowptr_offset), std::ios::beg);
    fin.read(reinterpret_cast<char*>(rowptr.data()),
             static_cast<std::streamsize>(rowptr.size() * sizeof(uint32_t)));
    return fin.good();
}

inline bool readBcsrWeight(const BcsrFileSource& source,
                           uint32_t post_local,
                           uint32_t pre_global,
                           float& weight_out) {
    weight_out = 0.0f;
    const uint32_t rows = source.requested_rows ? source.requested_rows : source.meta.rows;
    const uint32_t br = source.blockRows();
    const uint32_t bc = source.blockCols();
    const uint32_t idx_bytes = source.indexBytes();
    if (post_local >= rows || pre_global >= source.meta.cols || source.valueBytes() != 4) return false;

    std::vector<uint32_t> rowptr;
    if (!readBcsrRowptr(source, rows, rowptr)) return false;
    const uint32_t block_row = post_local / br;
    const uint32_t intra_row = post_local % br;
    if (block_row + 1u >= rowptr.size()) return false;
    const uint32_t start = rowptr[block_row];
    const uint32_t end = rowptr[block_row + 1u];
    const uint32_t block_col = pre_global / bc;
    const uint32_t intra_col = pre_global % bc;

    std::ifstream fin(source.path, std::ios::binary);
    if (!fin.good()) return false;
    int32_t selected = -1;
    for (uint32_t j = start; j < end; ++j) {
        uint64_t offset = source.meta.colidx_offset;
        if (source.rowpack()) {
            offset += static_cast<uint64_t>(block_row) * source.meta.colidx_row_stride_bytes;
            offset += static_cast<uint64_t>(j - start) * idx_bytes;
        } else {
            offset += static_cast<uint64_t>(j) * idx_bytes;
        }
        if (offset + idx_bytes > source.file_size) return false;
        fin.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        uint32_t col = 0;
        if (idx_bytes == 2) {
            uint16_t v = 0;
            fin.read(reinterpret_cast<char*>(&v), sizeof(v));
            col = v;
        } else {
            fin.read(reinterpret_cast<char*>(&col), sizeof(col));
        }
        if (!fin.good()) return false;
        if (col == block_col) {
            selected = static_cast<int32_t>(j - start);
            break;
        }
    }
    if (selected < 0) return false;

    const uint64_t block_bytes = bcsrBytesPerBlock(br, bc, source.valueBytes());
    const uint64_t global_block = static_cast<uint64_t>(start) + static_cast<uint64_t>(selected);
    uint64_t data_offset = source.meta.blockdata_offset;
    if (source.rowpack()) {
        data_offset += static_cast<uint64_t>(block_row) * source.meta.blockdata_row_stride_bytes;
        data_offset += static_cast<uint64_t>(selected) * block_bytes;
    } else {
        data_offset += global_block * block_bytes;
    }
    const uint64_t cell_offset = data_offset +
        (static_cast<uint64_t>(intra_row) * bc + intra_col) * sizeof(float);
    if (cell_offset + sizeof(float) > source.file_size) return false;

    if (source.blockIdsPresent()) {
        const uint64_t id_offset = source.meta.blockids_offset +
            (global_block * static_cast<uint64_t>(br) * bc +
             static_cast<uint64_t>(intra_row) * bc + intra_col) * sizeof(uint32_t);
        if (id_offset + sizeof(uint32_t) > source.file_size) return false;
        fin.seekg(static_cast<std::streamoff>(id_offset), std::ios::beg);
        uint32_t id = 0xFFFFFFFFu;
        fin.read(reinterpret_cast<char*>(&id), sizeof(id));
        if (!fin.good() || id == 0xFFFFFFFFu) return false;
    }

    fin.seekg(static_cast<std::streamoff>(cell_offset), std::ios::beg);
    fin.read(reinterpret_cast<char*>(&weight_out), sizeof(weight_out));
    return fin.good();
}

}} // namespace SST::SnnDL
