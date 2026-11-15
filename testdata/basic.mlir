// Minimal entry points that exercise the FFI paths the real models need:
//   - f32 tensors in/out                (SigLIP / BioCLIP / ArcFace / OCR)
//   - i64 tensors                       (SigLIP text input_ids)
//   - multiple results                  (SCRFD: 9 outputs; SigLIP: hidden + pooled)
//   - dynamic dims                      (PP-OCR detection H x W)
//   - f16 result                        (precision variants)
//
// Compile with just testdata.
module @basic {
  func.func @abs(%x: tensor<4xf32>) -> tensor<4xf32> {
    %0 = math.absf %x : tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  func.func @add_i64(%a: tensor<2x3xi64>, %b: tensor<2x3xi64>) -> tensor<2x3xi64> {
    %0 = arith.addi %a, %b : tensor<2x3xi64>
    return %0 : tensor<2x3xi64>
  }

  func.func @sum_and_product(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
    %0 = arith.addf %a, %b : tensor<4xf32>
    %1 = arith.mulf %a, %b : tensor<4xf32>
    return %0, %1 : tensor<4xf32>, tensor<4xf32>
  }

  func.func @negate_dynamic(%x: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = arith.negf %x : tensor<?x?xf32>
    return %0 : tensor<?x?xf32>
  }

  func.func @to_f16(%x: tensor<4xf32>) -> tensor<4xf16> {
    %0 = arith.truncf %x : tensor<4xf32> to tensor<4xf16>
    return %0 : tensor<4xf16>
  }
}
