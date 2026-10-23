"""Acceptance tests for the experimental 'callee_frame' JIT parameter.

The toy is fib over a virtualizable Frame; the tests check the results
with the flag off and on, the virtual entry and the virtual body of the
callee, and that callers do not allocate the Frame they pass to
call_assembler.

    python2 pytest.py rpython/jit/metainterp/test/test_callee_frame.py
"""
import py
from rpython.rlib.jit import JitDriver, hint, dont_look_inside
from rpython.jit.metainterp.test.support import LLJitMixin
from rpython.jit.metainterp.resoperation import rop
from rpython.jit.metainterp.warmspot import get_stats



# ---------- the toy interpreter: fib(n) with one virtualizable Frame ----------

class Frame(object):
    _virtualizable_ = ['n', 'acc']

    def __init__(self, n):
        self = hint(self, access_directly=True, fresh_virtualizable=True)
        self.n = n
        self.acc = 0

driver = JitDriver(greens=['code'], reds=['frame'], virtualizables=['frame'])

@dont_look_inside
def peek(frame):
    # residual call that receives the frame => forces it via vable_token;
    # the result shows whether the forced Frame has the right fields
    return frame.n * 100 + frame.acc

class Boom(Exception):
    pass

def portal(code, frame):
    # code: 0 = plain fib, 1 = also peek() at n == 3,
    #       2 = also raise Boom at n == 0 (caught one level up),
    #       3 = rare branch at n == 3 whose frame state is used later
    driver.jit_merge_point(code=code, frame=frame)
    n = frame.n
    if n < 2:
        if code == 2 and n == 0:
            raise Boom
        return n
    extra = 0
    if code == 1 and n == 3:
        frame.acc = 7
        extra = peek(frame)
    if code == 3 and n == 3:
        frame.acc = 5
    try:
        a = portal(code, Frame(n - 1))
    except Boom:
        a = 0
    try:
        b = portal(code, Frame(n - 2))
    except Boom:
        b = 0
    return a + b + frame.acc + extra

def main(code, n):
    return portal(code, Frame(n))


class CalleeFrameTests:

    def run(self, code, n, callee_frame):
        return self.meta_interp(main, [code, n], inline=True,
                                max_unroll_recursion=1,
                                callee_frame=callee_frame)

    def count_call_assembler(self):
        return sum(1 for loop in get_stats().get_all_loops()
                     for op in loop.operations
                     if rop.is_call_assembler(op.getopnum()))

    def is_frame_descr(self, descr):
        jd = get_stats().metainterp_sd.jitdrivers_sd[0]
        return descr is jd.virtualizable_info.vable_size_descr

    def frames_allocated_for_call_assembler(self):
        """Number of call_assembler or keepalive ops that get an argument
        produced by a new_with_vtable in the same trace (= caller-side
        Frame allocation; the keepalive after the call would force it too)."""
        count = 0
        for loop in get_stats().get_all_loops():
            news = set()
            for op in loop.operations:
                if op.getopnum() == rop.NEW_WITH_VTABLE:
                    news.add(op)
                elif (rop.is_call_assembler(op.getopnum()) or
                        op.getopnum() == rop.KEEPALIVE):
                    if [arg for arg in op.getarglist() if arg in news]:
                        count += 1
        return count

    def frame_allocs_except_escapes(self):
        """Number of new_with_vtable(Frame) in the compiled loops (bridges
        are not recorded by get_stats()) whose result is not passed to a
        residual call, i.e. allocations that are not an escape point."""
        count = 0
        loops = get_stats().get_all_loops()
        # with a virtual body, the virtual entry is redirected to it and
        # never runs (the toy has a single procedure)
        redirected = [loop for loop in loops if loop.name == 'virtual body']
        for loop in loops:
            if redirected and loop.name == 'virtual entry':
                continue
            frames = [op for op in loop.operations
                      if op.getopnum() == rop.NEW_WITH_VTABLE and
                         self.is_frame_descr(op.getdescr())]
            escaped = set()
            for op in loop.operations:
                if (rop.is_call(op.getopnum()) and
                        not rop.is_call_assembler(op.getopnum())):
                    escaped.update(op.getarglist())
            count += len([op for op in frames if op not in escaped])
        return count

    # ---- baseline ----

    def test_toy_uses_call_assembler(self):
        assert self.run(0, 15, 0) == main(0, 15)
        assert self.count_call_assembler() > 0
        assert self.frames_allocated_for_call_assembler() > 0

    def test_results_flag_off(self):
        for code in [0, 1, 2, 3]:
            assert self.run(code, 15, 0) == main(code, 15)

    def test_results_flag_on(self):
        for code in [0, 1, 2, 3]:
            assert self.run(code, 15, 1) == main(code, 15)

    def test_caller_reads_callee_frame_after_call(self):
        # like test_recursive.test_directly_call_assembler_virtualizable:
        # the caller reads the callee frame after call_assembler returns,
        # so callee_frame must fall back to the materialized Frame here.
        def portal2(code, frame):
            # the result is only passed back through the callee's Frame
            driver.jit_merge_point(code=code, frame=frame)
            n = frame.n
            if n < 2:
                frame.acc = n
                return 0
            sub1 = Frame(n - 1)
            portal2(code, sub1)
            sub2 = Frame(n - 2)
            portal2(code, sub2)
            frame.acc = sub1.acc + sub2.acc
            return 0
        def main2(code, n):
            f = Frame(n)
            portal2(code, f)
            return f.acc
        res = self.meta_interp(main2, [0, 12], inline=True,
                               max_unroll_recursion=1, callee_frame=1)
        assert res == main2(0, 12) == 144
        assert self.count_call_assembler() > 0
        assert self.frames_allocated_for_call_assembler() > 0

    def test_caller_stores_callee_frame(self):
        # the Frame is stored into another (virtual) object before the
        # call and read through it afterwards: the caller must keep
        # passing the Frame itself
        class Holder(object):
            pass
        def portal3(code, frame):
            driver.jit_merge_point(code=code, frame=frame)
            n = frame.n
            frame.acc = n
            if n < 2:
                return n
            holder = Holder()
            holder.frame = Frame(n - 1)
            a = portal3(code, holder.frame)
            b = portal3(code, Frame(n - 2))
            return a + b + holder.frame.acc
        def main3(code, n):
            return portal3(code, Frame(n))
        res = self.meta_interp(main3, [0, 12], inline=True,
                               max_unroll_recursion=1, callee_frame=2)
        assert res == main3(0, 12)
        assert self.count_call_assembler() > 0
        assert self.frames_allocated_for_call_assembler() > 0

    # ---- the virtual entry itself (callee side) ----

    def virtual_entries(self):
        return [token.virtual_entry
                for token in get_stats().get_all_jitcell_tokens()
                if token.virtual_entry is not None]

    def call_virtual_entry(self, vtoken, *args):
        # what CALL_ASSEMBLER does, for an INT result
        from rpython.rtyper.lltypesystem import lltype, llmemory
        metainterp_sd = get_stats().metainterp_sd
        cpu = metainterp_sd.cpu
        deadframe = cpu.execute_token(vtoken, *args)
        if cpu.get_latest_descr(deadframe) is cpu.done_with_this_frame_descr_int:
            return cpu.get_int_value(deadframe, 0)
        jd = vtoken.outermost_jitdriver_sd
        return jd._assembler_call_helper(deadframe,
                                         lltype.nullptr(llmemory.GCREF.TO))

    def test_no_virtual_entry_flag_off(self):
        self.run(0, 15, 0)
        assert self.virtual_entries() == []

    def test_virtual_entry_compiled(self):
        self.run(0, 15, 1)
        [vtoken] = self.virtual_entries()
        assert vtoken.is_virtual_entry
        # inputargs: [n, acc] (the vable fields); the Frame is allocated
        # at the head of the virtual entry
        for n in [0, 1, 2, 5, 10, 15]:
            assert self.call_virtual_entry(vtoken, n, 0) == main(0, n)

    def test_virtual_entry_slow_paths(self):
        self.check_slow_paths(1)

    def check_slow_paths(self, callee_frame):
        for code in [1, 2, 3]:
            self.run(code, 15, callee_frame)
            for vtoken in self.virtual_entries():
                # several times each, so that bridges get compiled too
                for n in [3, 4, 9, 3, 4, 9, 0, 1, 0, 1]:
                    try:
                        got = self.call_virtual_entry(vtoken, n, 0)
                    except Exception:
                        got = 'raised'
                    try:
                        expected = portal(code, Frame(n))
                    except Boom:
                        expected = 'raised'
                    assert got == expected

    def test_virtual_entry_extra_fields(self):
        # a field outside _virtualizable_ becomes an extra argument
        class Frame2(object):
            _virtualizable_ = ['n']
            def __init__(self, n, base):
                self = hint(self, access_directly=True,
                            fresh_virtualizable=True)
                self.n = n
                self.base = base
        driver2 = JitDriver(greens=['code'], reds=['frame'],
                            virtualizables=['frame'])
        def portal2(code, frame):
            driver2.jit_merge_point(code=code, frame=frame)
            n = frame.n
            if n < 2:
                return frame.base + n
            return (portal2(code, Frame2(n - 1, frame.base)) +
                    portal2(code, Frame2(n - 2, frame.base)))
        def main2(code, n, base):
            return portal2(code, Frame2(n, base))
        res = self.meta_interp(main2, [0, 15, 1000], inline=True,
                               max_unroll_recursion=1, callee_frame=1)
        assert res == main2(0, 15, 1000)
        [vtoken] = self.virtual_entries()
        for n in [0, 1, 5, 12]:
            assert self.call_virtual_entry(vtoken, n, 7) == portal2(
                0, Frame2(n, 7))

    # ---- the virtual body (callee_frame=2, callee side) ----

    def own_frame_allocs(self, operations):
        """new_with_vtable(Frame) whose result is not passed to any call
        or keepalive, i.e. the procedure's own Frame being allocated."""
        used = set()
        for op in operations:
            if (rop.is_call(op.getopnum()) or
                    rop.is_call_assembler(op.getopnum()) or
                    op.getopnum() == rop.KEEPALIVE):
                used.update(op.getarglist())
        return len([op for op in operations
                    if op.getopnum() == rop.NEW_WITH_VTABLE and
                       self.is_frame_descr(op.getdescr()) and
                       op not in used])

    def collect_bridges(self):
        from rpython.jit.metainterp import compile
        bridges = []
        orig = compile.send_bridge_to_backend
        def send(jitdriver_sd, metainterp_sd, faildescr, inputargs,
                 operations, original_loop_token, memo):
            bridges.append((original_loop_token, list(operations)))
            return orig(jitdriver_sd, metainterp_sd, faildescr, inputargs,
                        operations, original_loop_token, memo)
        compile.send_bridge_to_backend = send
        def undo():
            compile.send_bridge_to_backend = orig
        return bridges, undo

    def test_no_virtual_body_flag_1(self):
        self.run(0, 15, 1)
        [vtoken] = self.virtual_entries()
        assert vtoken.virtual_body is None

    def test_virtual_body_compiled(self):
        self.run(0, 15, 2)
        [vtoken] = self.virtual_entries()
        assert vtoken.virtual_body is not None
        [body] = [loop for loop in get_stats().get_all_loops()
                  if loop.name == 'virtual body']
        assert self.own_frame_allocs(body.operations) == 0
        # the virtual entry is redirected to the virtual body
        for n in [0, 1, 2, 5, 10, 15]:
            assert self.call_virtual_entry(vtoken, n, 0) == main(0, n)

    def test_virtual_body_slow_paths(self):
        self.check_slow_paths(2)

    def test_virtual_body_bridge_keeps_frame_virtual(self):
        bridges, undo = self.collect_bridges()
        try:
            self.run(0, 15, 2)
            [vtoken] = self.virtual_entries()
            body = vtoken.virtual_body
            # n < 2 fails the first guard of the body: blackhole first,
            # then a bridge out of the virtual body (unless the run made
            # one already)
            for i in range(10):
                assert self.call_virtual_entry(vtoken, 1, 0) == 1
        finally:
            undo()
        mine = [ops for token, ops in bridges if token is body]
        assert mine
        for ops in mine:
            assert self.own_frame_allocs(ops) == 0

    # ---- no caller-side Frame allocation for call_assembler ----

    def test_no_frame_alloc_before_call_assembler(self):
        assert self.run(0, 15, 1) == main(0, 15)
        assert self.count_call_assembler() > 0
        assert self.frames_allocated_for_call_assembler() == 0

    # ---- the Frame stays virtual in the callee; slow paths correct ----

    def test_frame_virtual_in_callee(self):
        assert self.run(0, 15, 2) == main(0, 15)
        assert self.count_call_assembler() > 0
        assert self.frame_allocs_except_escapes() == 0

    def test_guard_failure_in_callee(self):
        # code 3 takes a rare branch at n == 3 and later reads frame.acc:
        # the guard failure must rebuild the Frame with the right fields
        assert self.run(3, 15, 2) == main(3, 15)
        assert self.frame_allocs_except_escapes() == 0

    def test_force_materializes_frame(self):
        # peek() reads the forced Frame; its result is part of the result
        assert self.run(1, 15, 2) == main(1, 15)
        assert self.frame_allocs_except_escapes() == 0

    def test_exception_in_callee(self):
        assert self.run(2, 15, 2) == main(2, 15)
        assert self.frame_allocs_except_escapes() == 0


class TestLLtype(CalleeFrameTests, LLJitMixin):
    pass
