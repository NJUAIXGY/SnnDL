#pragma once

#include <cstddef>
#include <cstdint>

namespace SnnDL {
namespace v5 {

inline constexpr std::uint16_t kArtifactFormatVersion = 1;
inline constexpr std::uint16_t kArtifactLittleEndianFlag = 0x0001;
inline constexpr std::size_t kArtifactHeaderBytes = 72;
inline constexpr std::size_t kArtifactDigestBytes = 32;
inline constexpr char kArtifactMagic[] = "SNNDLV5\0";

const char* artifactMagic();

}  // namespace v5
}  // namespace SnnDL
