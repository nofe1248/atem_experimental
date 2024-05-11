parser grammar AtemParser;

options {
	tokenVocab = AtemLexer;
}

program:
	module_declaration_expression? EOF
;

expression:
    primary_expression                                      #primaryExpression |
    expression multiplicative_operators expression          #multiplicativeExpression |
    expression additive_operators expression                #additiveExpression |
    expression comparison_operator expression               #comparisonExpression |
    expression assignment_operators expression              #assignmentExpression |
    expression arithmetic_assignment_operators expression   #arithmeticAssignmentExpression |
    expression function_call_operator                       #functionCallExpression |
    compiler_builtin function_call_operator                 #compilerBuiltinCallExpression |
    control_flow_expression                                 #controlFlowExpression |
    declaration_expression                                  #declarationExpression |
    definition_list_expression                              #definitionListExpression |
    type_expression                                         #typeExpression
;

primary_expression:
    paren_expression |
    identifier_expression |
    literal_expression
;

paren_expression:
    LeftParenthese expression RightParenthese
;

declaration_expression:
    function_declaration_expression |
    module_declaration_expression |
    variable_declaration_expression
;

function_declaration_expression:
    Identifier Colon KeywordFunction function_argument_list? (Arrow function_return_type_list)? Assign expression_or_block
;

function_argument_list:
    LeftParenthese function_argument (Comma function_argument)* RightParenthese
;

function_argument:
    function_argument_name type_annotation
;

function_argument_name:
    Identifier
;

function_return_type_list:
    type_expression (Comma type_expression)?
;

module_declaration_expression:
    Identifier Colon KeywordModule Assign definition_list_expression
;

variable_declaration_expression:
    Identifier Colon KeywordVar type_expression? Assign expression
;

definition_list_expression:
    LeftCurly declaration_expression* RightCurly
;

multiplicative_operators:
    Mul | Divide
;

additive_operators:
    Add | Sub
;

assignment_operators:
    Assign
;

arithmetic_assignment_operators:
    AddAssign |
    SubAssign |
    MulAssign |
    DivideAssign
;

comparison_operator:
	GreaterThan |
	LessThan |
	GreaterThanOrEqual |
	LessThanOrEqual |
	Equal |
	NotEqual |
	ThreeWayComparison
;

function_call_operator:
    LeftParenthese function_call_argument_list? RightParenthese
;

function_call_argument_list:
    expression (Comma expression)*
;

compiler_builtin:
    Builtin Identifier
;

control_flow_expression:
    if_expression |
    while_expression |
    return_expression |
    break_expression |
    continue_expression
;

else_clause:
	KeywordElse expression_or_block
;

if_expression:
    KeywordIf expression then_expression_or_block
    else_clause?
;

while_expression:
    KeywordWhile expression then_expression_or_block
;

return_expression:
    KeywordReturn expression
;

break_expression:
    KeywordBreak
;

continue_expression:
    KeywordContinue
;

then_expression_or_block:
    KeywordThen expression | code_block_expression
;

expression_or_block:
    code_block_expression | expression
;

code_block_expression:
    LeftCurly expression* RightCurly
;

identifier_expression:
    Identifier
;

type_expression:
    simple_type |
    function_type
;

function_type:
    LeftParenthese function_type_argument_list RightParenthese (Arrow function_type_return_type_list)?
;

function_type_argument_list:
    function_type_argument (Comma function_type_argument)*
;

function_type_argument:
    type_expression
;

function_type_return_type_list:
    type_expression (Comma type_expression)?
;

simple_type:
    integer_type |
    floating_point_type |
    boolean_type
;

integer_type:
    unsigned_integer_type |
    signed_integer_type
;

signed_integer_type:
    KeywordInt
;

unsigned_integer_type:
    KeywordUInt
;

floating_point_type:
    KeywordFloat16 |
    KeywordFloat32 |
    KeywordFloat64 |
    KeywordFloat80 |
    KeywordFloat128
;

boolean_type:
    KeywordBool
;

literal_expression:
    integer_literal |
    boolean_literal
;

integer_literal:
    DecimalDigits |
    DecimalLiteral |
    BinaryLiteral |
    OctalLiteral |
    HexadecimalLiteral
;

boolean_literal:
    KeywordTrue |
    KeywordFalse
;

type_annotation:
    Colon type_expression
;