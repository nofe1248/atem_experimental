module;

#include "AtemParser.h"
#include "AtemParserBaseVisitor.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include "AtemIR/Dialect/include/AtemIRAttrs.h"
#include "AtemIR/Dialect/include/AtemIRDialect.h"
#include "AtemIR/Dialect/include/AtemIROps.h"
#include "AtemIR/Dialect/include/AtemIRTypes.h"

#include <any>
#include <memory>
#include <ranges>
#include <string>
#include <vector>

export module Atemc.Parser.ANTLR4Parser.Visitor;

import Atemc.Utils;

export namespace atemc::parser::antlr4
{
    class AtemIRBuilderVisitor final : public atemc_antlr::AtemParserBaseVisitor
    {
        struct VariableSymbol
        {
            mlir::Value value;
            atemc_antlr::AtemParser::Variable_declaration_expressionContext* ctx;
        };

        using SymbolTableT = llvm::ScopedHashTable<llvm::StringRef, VariableSymbol>;
        SymbolTableT symbol_table_;

        atemc_antlr::AtemParser::ProgramContext *tree_;
        mlir::MLIRContext &context_;
        mlir::OpBuilder builder_;
        std::string filename_;
        mlir::ModuleOp module_;

        auto getFileLineLocation(::antlr4::ParserRuleContext *ctx) -> mlir::FileLineColLoc
        {
            return mlir::FileLineColLoc::get(builder_.getStringAttr(this->filename_), ctx->getStart()->getLine(),
                                             ctx->getStart()->getCharPositionInLine());
        }

    public:
        AtemIRBuilderVisitor(atemc_antlr::AtemParser::ProgramContext *tree, mlir::MLIRContext &context) :
            tree_(tree), context_(context), builder_(&context_),
            filename_(tree_->getStart()->getTokenSource()->getSourceName())
        {
        }

        auto buildModule() -> mlir::ModuleOp { return std::any_cast<mlir::ModuleOp>(this->visitProgram(this->tree_)); }

        auto visitProgram(atemc_antlr::AtemParser::ProgramContext *ctx) -> std::any override
        {
            if (auto ptr = ctx->module_declaration_expression())
            {
                return this->visitModule_declaration_expression(ptr);
            }
            return builder_.create<mlir::ModuleOp>(this->getFileLineLocation(ctx));
        }

        auto visitModule_declaration_expression(atemc_antlr::AtemParser::Module_declaration_expressionContext *ctx)
            -> std::any override
        {
            auto mod = builder_.create<mlir::ModuleOp>(this->getFileLineLocation(ctx), ctx->Identifier()->getText());
            this->module_ = mod;
            for (auto decl : ctx->definition_list_expression()->declaration_expression())
            {
                this->visitDeclaration_expression(decl);
            }
            return mod;
        }

        auto visitFunction_declaration_expression(atemc_antlr::AtemParser::Function_declaration_expressionContext *ctx)
            -> std::any override
        {
            llvm::SmallVector<mlir::Type> param_types{};
            if (auto ptr = ctx->function_argument_list())
            {
                param_types = std::views::transform(ptr->function_argument(),
                                                    [this](atemc_antlr::AtemParser::Function_argumentContext *ctx) {
                                                        return std::any_cast<mlir::Type>(this->visitType_expression(
                                                            ctx->type_annotation()->type_expression()));
                                                    }) |
                    std::ranges::to<llvm::SmallVector<mlir::Type>>();
            }

            llvm::SmallVector<mlir::Type> ret_types{};
            if (auto ptr = ctx->function_return_type_list())
            {
                ret_types = std::views::transform(ptr->type_expression(),
                                                  [this](atemc_antlr::AtemParser::Type_expressionContext *ctx) {
                                                      return std::any_cast<mlir::Type>(this->visitType_expression(ctx));
                                                  }) |
                    std::ranges::to<llvm::SmallVector<mlir::Type>>();
            }
            else
            {
                ret_types.push_back(atemir::UnitType::get(&context_));
            }

            auto func_type = atemir::FunctionType::get(&context_, param_types, ret_types);

            auto func_name = ctx->Identifier()->getText();

            builder_.setInsertionPointToEnd(module_.getBody());

            auto func = builder_.create<atemir::FunctionOp>(this->getFileLineLocation(ctx), func_name, func_type);

            func.addEntryBlock();
            mlir::Block &entry_block = func.front();
            builder_.setInsertionPointToEnd(&entry_block);

            this->visit(ctx->expression_or_block());

            atemir::ReturnOp return_op;
            if (not entry_block.empty())
            {
                return_op = mlir::dyn_cast<atemir::ReturnOp>(entry_block.back());
            }
            if (not return_op)
            {
                builder_.create<atemir::ReturnOp>(this->getFileLineLocation(ctx), mlir::ValueRange{});
            }
            else if (return_op->getNumOperands() > 0)
            {
                func.setType(atemir::FunctionType::get(&context_, func.getFunctionType().getInputs(),
                                                       *return_op->operand_type_begin()));
            }

            if (func_name != "main")
            {
                func.setPrivate();
            }

            return func;
        }

        auto visitSigned_integer_type(atemc_antlr::AtemParser::Signed_integer_typeContext *ctx) -> std::any override
        {
            auto rawtext = ctx->getText();
            rawtext.erase(rawtext.begin(), rawtext.begin() + 3);
            auto int_type = atemir::IntegerType::get(&context_, std::stoi(rawtext), true);
            return mlir::Type{int_type};
        }

        auto visitUnsigned_integer_type(atemc_antlr::AtemParser::Unsigned_integer_typeContext *ctx) -> std::any override
        {
            auto rawtext = ctx->getText();
            rawtext.erase(rawtext.begin(), rawtext.begin() + 4);
            auto int_type = atemir::IntegerType::get(&context_, std::stoi(rawtext), false);
            return mlir::Type{int_type};
        }

        auto visitBoolean_type(atemc_antlr::AtemParser::Boolean_typeContext *ctx) -> std::any override
        {
            return mlir::Type{atemir::BooleanType::get(&context_)};
        }

        auto visitFloating_point_type(atemc_antlr::AtemParser::Floating_point_typeContext *ctx) -> std::any override
        {
            if (ctx->KeywordFloat16())
            {
                return mlir::Type{atemir::Float16Type::get(&context_)};
            }
            if (ctx->KeywordFloat32())
            {
                return mlir::Type{atemir::Float32Type::get(&context_)};
            }
            if (ctx->KeywordFloat64())
            {
                return mlir::Type{atemir::Float64Type::get(&context_)};
            }
            if (ctx->KeywordFloat80())
            {
                return mlir::Type{atemir::Float80Type::get(&context_)};
            }
            if (ctx->KeywordFloat128())
            {
                return mlir::Type{atemir::Float128Type::get(&context_)};
            }
            llvm_unreachable("");
        }

        auto visitFunction_type(atemc_antlr::AtemParser::Function_typeContext *ctx) -> std::any override
        {
            auto params = std::views::transform(ctx->function_type_argument_list()->function_type_argument(),
                                                [this](atemc_antlr::AtemParser::Function_type_argumentContext *ctx) {
                                                    return std::any_cast<mlir::Type>(
                                                        this->visitType_expression(ctx->type_expression()));
                                                }) |
                std::ranges::to<std::vector<mlir::Type>>();
            auto rets = std::views::transform(ctx->function_type_return_type_list()->type_expression(),
                                              [this](atemc_antlr::AtemParser::Type_expressionContext *ctx)
                                              { return std::any_cast<mlir::Type>(this->visitType_expression(ctx)); }) |
                std::ranges::to<std::vector<mlir::Type>>();
            return mlir::Type{atemir::FunctionType::get(&context_, params, rets)};
        }
    };
} // namespace atemc::parser::antlr4
