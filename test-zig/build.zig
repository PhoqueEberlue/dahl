const std = @import("std");

pub fn build(b: *std.Build) void {
    const exe = b.addExecutable(.{
        .name = "starpu_zig_example",
        .root_source_file = .{ .src_path = .{ .owner = b, .sub_path = "main.zig" } },
        .target = b.standardTargetOptions(.{}),
        .optimize = b.standardOptimizeOption(.{}),
    });

    // Add StarPU include and library
    exe.addIncludePath(.{ .src_path = .{ .owner = b, .sub_path = "/nix/store/nqjg5y51ga3dg4cj2y7y8yc78qng6xbj-starpu-master/include/starpu/1.4/" } }); // Adjust if necessary
    exe.linkSystemLibrary("starpu-1.4");

    b.installArtifact(exe);

    const run_exe = b.addRunArtifact(exe);

    const run_step = b.step("run", "Run the application");
    run_step.dependOn(&run_exe.step);
}
