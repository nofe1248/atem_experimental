module {
    atemir.function @main() -> !atemir.int<s, 64> {
        %lhs = atemir.constant #atemir.int<42>: !atemir.int<s, 64>
        %rhs = atemir.constant #atemir.int<24>: !atemir.int<s, 64>
        %0 = atemir.unary neg %lhs: !atemir.int<s, 64> -> !atemir.int<s, 64>
        %1 = atemir.binary add %lhs %rhs: !atemir.int<s, 64> -> !atemir.int<s, 64>
        %2 = atemir.binary sub %lhs %rhs nsw: !atemir.int<s, 64> -> !atemir.int<s, 64>
        %3 = atemir.binary mul %lhs %rhs nuw: !atemir.int<s, 64> -> !atemir.int<s, 64>
        %4 = atemir.compare lt %lhs %rhs: !atemir.int<s, 64>
        atemir.return %0: !atemir.int<s, 64>
    }
}