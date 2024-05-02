module {
    atemir.function @main() -> !atemir.int<s, 64> {
        %0 = atemir.constant #atemir.int<42>: !atemir.int<s, 64>

        atemir.while {
            atemir.yield
        } do {
            %while_loop_cond = atemir.constant #atemir.bool<true>: !atemir.bool
            atemir.condition(%while_loop_cond)
        }

        atemir.do {
            %do_while_loop_cond = atemir.constant #atemir.bool<true>: !atemir.bool
            atemir.condition(%do_while_loop_cond)
        } while {
            atemir.yield
        }

        %if_cond = atemir.constant #atemir.bool<true>: !atemir.bool
        atemir.if %if_cond {
            atemir.yield
        } else {
            atemir.yield
        }

        atemir.cfor : cond {
            %cfor_loop_cond = atemir.constant #atemir.bool<true>: !atemir.bool
            atemir.condition(%cfor_loop_cond)
        } body {
            atemir.yield
        } step {
            atemir.yield
        }

        atemir.return %0: !atemir.int<s, 64>
    }
}