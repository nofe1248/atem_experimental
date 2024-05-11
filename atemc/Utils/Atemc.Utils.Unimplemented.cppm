module;

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ErrorHandling.h"

#include <optional>
#include <string>

export module Atemc.Utils.Unimplemented;

export namespace atemc::utils
{
    auto unimplemented(std::optional<std::string_view> feature = std::nullopt) -> void
    {
        if (feature)
        {
            llvm_unreachable(std::string{""}.append("Feature ").append(*feature).append(" is currently unimplemented").c_str());
        }
        else
        {
            llvm_unreachable("Currently unimplemented");
        }
    }
}