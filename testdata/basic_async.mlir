// Minimal Torch-dialect entry points that emit coarse-fences `$async`
// wrappers alongside the synchronous entry points.
//
// Compile with `just testdata`.
module @basic_async {
  func.func @abs(%arg0: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
    %0 = torch.aten.abs %arg0 : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
    return %0 : !torch.vtensor<[4],f32>
  }

  func.func @sum_and_product(%arg0: !torch.vtensor<[4],f32>, %arg1: !torch.vtensor<[4],f32>) -> (!torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>) {
    %int1 = torch.constant.int 1
    %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>, !torch.int -> !torch.vtensor<[4],f32>
    %1 = torch.aten.mul.Tensor %arg0, %arg1 : !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
    return %0, %1 : !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>
  }
}
