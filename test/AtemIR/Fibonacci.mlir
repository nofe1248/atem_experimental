module {
    atemir.func @fibonacci(%n: i32) -> i32 {
        %pred1 = atemir.constant 1: i32
        %pred2 = atemir.constant 1: i32
        %cond = atemir.cmpi ult n, 2
        %i = atemir.constant 2: i32
        %ret = atemir.if %cond {
            atemir.yield 1: i32
        } else {
            %res = atemir.while (%arg1 = %i) : (i32) -> (i32) {
                %condition = atemir.call @evaluate_condition(%arg1) : (i32) -> i1
                atemir.condition(%condition) %arg1 : i32
            } do {
                ^bb0(%arg2: i32):
                    %next = atemir.call @payload(%arg2) : (i32) -> i32
                    atemir.yield %next : i32
            }
            atemir.yield %ret: i32
        }
        atemir.return %ret: i32
    }
    atemir.func @main() -> i32 {
        %c = atemir.constant 10: i32
        %fib = atemir.call @fibonacci(%c) : (i32) -> i32
        atemir.intrinsics("__builtin_print", %fib)
        atemir.return %c: i32
    }
}