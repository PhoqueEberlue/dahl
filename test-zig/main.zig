const std = @import("std");

const c = @cImport({
    @cInclude("starpu.h");
});

export fn scal(buffers: [*c]?*anyopaque, cl_arg: ?*anyopaque) callconv(.c) void {
    std.debug.print("Inside codelet task! {*} {*} \n", .{buffers, cl_arg});
}

const cl_scal = c.starpu_codelet {
    .cpu_funcs = scal,
    .nbuffers = 1,
    // .modes = { c.STARPU_RW; },
};

pub fn main() void {
    if (c.starpu_init(null) != 0) {
        std.debug.print("Failed to initialize StarPU\n", .{});
        return;
    }

    const handle: [*c]c.starpu_data_handle_t = undefined;

    const data = @Vector(4, i32){ 5, 6, 7, 8 };

    c.starpu_vector_data_register(handle, c.STARPU_MAIN_RAM, @intFromPtr(&data), 4, @sizeOf(i32));

    const ret = c.starpu_task_insert(&cl_scal, handle, c.STARPU_RW, 0);

    if (ret != 0) {
        std.debug.print("Failed to submit StarPU task (as expected, no codelet)\n", .{});
    }

    _ = c.starpu_shutdown();
    std.debug.print("StarPU shutdown complete.\n", .{});
}
