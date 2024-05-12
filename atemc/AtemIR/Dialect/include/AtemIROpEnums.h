#ifndef ATEMIROPENUMS_H
#define ATEMIROPENUMS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "Dialect/include/AtemIROpEnums.h.inc"

namespace atemir
{
    static auto isExternalLinkage(GlobalLinkageKind Linkage) -> bool
    {
        return Linkage == GlobalLinkageKind::ExternalLinkage;
    }
    static auto isAvailableExternallyLinkage(GlobalLinkageKind Linkage) -> bool
    {
        return Linkage == GlobalLinkageKind::AvailableExternallyLinkage;
    }
    static auto isLinkOnceAnyLinkage(GlobalLinkageKind Linkage) -> bool
    {
        return Linkage == GlobalLinkageKind::LinkOnceAnyLinkage;
    }
    static auto isLinkOnceODRLinkage(GlobalLinkageKind Linkage) -> bool
    {
        return Linkage == GlobalLinkageKind::LinkOnceODRLinkage;
    }
    static auto isLinkOnceLinkage(GlobalLinkageKind Linkage) -> bool
    {
        return isLinkOnceAnyLinkage(Linkage) or isLinkOnceODRLinkage(Linkage);
    }
    static auto isWeakAnyLinkage(GlobalLinkageKind Linkage) -> bool
    {
        return Linkage == GlobalLinkageKind::WeakAnyLinkage;
    }
    static auto isWeakODRLinkage(GlobalLinkageKind Linkage) -> bool
    {
        return Linkage == GlobalLinkageKind::WeakODRLinkage;
    }
    static auto isWeakLinkage(GlobalLinkageKind Linkage) -> bool
    {
        return isWeakAnyLinkage(Linkage) or isWeakODRLinkage(Linkage);
    }
    static auto isInternalLinkage(GlobalLinkageKind Linkage) -> bool
    {
        return Linkage == GlobalLinkageKind::InternalLinkage;
    }
    static auto isPrivateLinkage(GlobalLinkageKind Linkage) -> bool
    {
        return Linkage == GlobalLinkageKind::PrivateLinkage;
    }
    static auto isLocalLinkage(GlobalLinkageKind Linkage) -> bool
    {
        return isInternalLinkage(Linkage) or isPrivateLinkage(Linkage);
    }
    static auto isExternalWeakLinkage(GlobalLinkageKind Linkage) -> bool
    {
        return Linkage == GlobalLinkageKind::ExternalWeakLinkage;
    }
    LLVM_ATTRIBUTE_UNUSED static auto isCommonLinkage(GlobalLinkageKind Linkage) -> bool
    {
        return Linkage == GlobalLinkageKind::CommonLinkage;
    }
    LLVM_ATTRIBUTE_UNUSED static auto isValidDeclarationLinkage(GlobalLinkageKind Linkage) -> bool
    {
        return isExternalWeakLinkage(Linkage) or isExternalLinkage(Linkage);
    }

    /// Whether the definition of this global may be replaced by something
    /// non-equivalent at link time. For example, if a function has weak linkage
    /// then the code defining it may be replaced by different code.
    LLVM_ATTRIBUTE_UNUSED static auto isInterposableLinkage(GlobalLinkageKind Linkage) -> bool
    {
        switch (Linkage)
        {
        case GlobalLinkageKind::WeakAnyLinkage:
        case GlobalLinkageKind::LinkOnceAnyLinkage:
        case GlobalLinkageKind::CommonLinkage:
        case GlobalLinkageKind::ExternalWeakLinkage:
            return true;

        case GlobalLinkageKind::AvailableExternallyLinkage:
        case GlobalLinkageKind::LinkOnceODRLinkage:
        case GlobalLinkageKind::WeakODRLinkage:
            // The above three cannot be overridden but can be de-refined.

        case GlobalLinkageKind::ExternalLinkage:
        case GlobalLinkageKind::InternalLinkage:
        case GlobalLinkageKind::PrivateLinkage:
            return false;
        default:
            break;
        }
        llvm_unreachable("Fully covered switch above!");
    }

    /// Whether the definition of this global may be discarded if it is not used
    /// in its compilation unit.
    LLVM_ATTRIBUTE_UNUSED static auto isDiscardableIfUnused(GlobalLinkageKind Linkage) -> bool
    {
        return isLinkOnceLinkage(Linkage) or isLocalLinkage(Linkage) or isAvailableExternallyLinkage(Linkage);
    }

    /// Whether the definition of this global may be replaced at link time.  NB:
    /// Using this method outside of the code generators is almost always a
    /// mistake: when working at the IR level use isInterposable instead as it
    /// knows about ODR semantics.
    LLVM_ATTRIBUTE_UNUSED static auto isWeakForLinker(GlobalLinkageKind Linkage) -> bool
    {
        return Linkage == GlobalLinkageKind::WeakAnyLinkage or Linkage == GlobalLinkageKind::WeakODRLinkage or
            Linkage == GlobalLinkageKind::LinkOnceAnyLinkage or Linkage == GlobalLinkageKind::LinkOnceODRLinkage or
            Linkage == GlobalLinkageKind::CommonLinkage or Linkage == GlobalLinkageKind::ExternalWeakLinkage;
    }

    LLVM_ATTRIBUTE_UNUSED static auto isValidLinkage(GlobalLinkageKind L) -> bool
    {
        return isExternalLinkage(L) or isLocalLinkage(L) or isWeakLinkage(L) or isLinkOnceLinkage(L);
    }
} // namespace atemir

#endif // ATEMIROPENUMS_H
