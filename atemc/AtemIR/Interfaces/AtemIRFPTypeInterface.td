#ifndef ATEM_IR_FP_TYPE_INTERFACE_TD
#define ATEM_IR_FP_TYPE_INTERFACE_TD

include "mlir/IR/OpBase.td"

def AtemIRFPTypeInterface
    : TypeInterface<"AtemIRFPTypeInterface"> {
    let description = [{
        Contains helper functions to query properties about a floating-point type
    }];
    let cppNamespace = "::atemir";

    let methods = [
        InterfaceMethod<[{
            Returns the bit width of this floating-point type.
        }],
        /*retTy=*/"unsigned",
        /*methodName=*/"getWidth",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
            return llvm::APFloat::semanticsSizeInBits($_type.getFloatSemantics());
            }]
        >,
        InterfaceMethod<[{
            Return the mantissa width.
        }],
        /*retTy=*/"unsigned",
        /*methodName=*/"getFPMantissaWidth",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
            return llvm::APFloat::semanticsPrecision($_type.getFloatSemantics());
            }]
        >,
        InterfaceMethod<[{
            Return the float semantics of this floating-point type.
        }],
        /*retTy=*/"const llvm::fltSemantics &",
        /*methodName=*/"getFloatSemantics"
        >,
    ];
}

#endif //ATEM_IR_FP_TYPE_INTERFACE_TD