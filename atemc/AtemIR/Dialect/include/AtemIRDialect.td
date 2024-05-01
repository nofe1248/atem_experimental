#ifndef ATEM_IR_DIALECT_TD
#define ATEM_IR_DIALECT_TD

include "mlir/IR/OpBase.td"

def AtemIRDialect : Dialect {
    let name = "atemir";
    let cppNamespace = "::atemir";
    let summary = "A high-level dialect which is suitable for optimization and analytics.";
    let description = [{
        The Atem language is a expression-based, multi-paradigm, general-purpose language.
        This dialect provides a high-level intermediate representation of the language by
        resembling the AST.
    }];

    let useDefaultTypePrinterParser = 0;
    let useDefaultAttributePrinterParser = 0;

    let extraClassDeclaration = [{
        void registerTypes();
        void registerAttributes();
        void registerOperations();

        ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;
        void printType(::mlir::Type type, ::mlir::DialectAsmPrinter &printer) const override;

        ::mlir::Attribute parseAttribute(::mlir::DialectAsmParser &parser,
                             ::mlir::Type type) const override;

        void printAttribute(::mlir::Attribute attr, ::mlir::DialectAsmPrinter &os) const override;
    }];
}

#endif  //ATEM_IR_DIALECT_TD