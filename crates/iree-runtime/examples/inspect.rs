//! Inspect a `.vmfb` and optionally benchmark one function with zero-filled inputs
//! synthesized from its `iree.abi.declaration` (static shapes only).
//!
//! ```text
//! cargo run -p iree-runtime --release --example inspect -- model.vmfb
//! cargo run -p iree-runtime --release --example inspect -- model.vmfb \
//!     --driver metal --run module.main --iters 50
//! ```
//!
//! Handy for comparing precision variants of one model (fp32 / fp16 / w8 vmfb):
//! the declaration shows the I/O dtypes each variant really exposes.

#[cfg(not(iree_linked))]
fn main() {
    eprintln!("iree-runtime was built without the IREE runtime library; run just iree");
    std::process::exit(1);
}

#[cfg(iree_linked)]
fn main() {
    if let Err(err) = real_main() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

#[cfg(iree_linked)]
fn real_main() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    use iree_runtime::{BufferView, Instance, Session};

    let mut args = std::env::args().skip(1);
    let mut vmfb = None;
    let mut driver = "local-task".to_owned();
    let mut run = None;
    let mut iters = 10usize;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--driver" => driver = args.next().ok_or("--driver needs a value")?,
            "--run" => run = Some(args.next().ok_or("--run needs a function name")?),
            "--iters" => iters = args.next().ok_or("--iters needs a value")?.parse()?,
            "-h" | "--help" => {
                println!(
                    "usage: inspect <vmfb> [--driver local-task|metal|...] [--run <module.fn>] [--iters N]"
                );
                return Ok(());
            }
            other if vmfb.is_none() => vmfb = Some(other.to_owned()),
            other => return Err(format!("unexpected argument `{other}`").into()),
        }
    }
    let vmfb = vmfb.ok_or("missing <vmfb>")?;

    let instance = Instance::new()?;
    let device = instance.create_device(&driver)?;
    println!("device: {}", device.id());
    let session = Session::from_file(&device, &vmfb)?;

    for f in session.exported_functions() {
        if f.full_name.starts_with("hal.") {
            continue; // runtime built-ins
        }
        println!("{}  [{}]", f.full_name, f.calling_convention);
        if let Some(decl) = &f.declaration {
            println!("    {decl}");
        }
    }

    let Some(name) = run else { return Ok(()) };
    let function = session.lookup_function(&name)?;
    let decl = function
        .declaration()
        .ok_or("function has no iree.abi.declaration")?;
    let inputs = parse_input_tensors(&decl)?
        .into_iter()
        .map(|(shape, elem)| {
            let numel: usize = shape.iter().product();
            let bytes = vec![0u8; numel * elem.byte_size().expect("byte-aligned element type")];
            BufferView::from_bytes(&device, &shape, elem, &bytes)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let input_refs: Vec<&BufferView> = inputs.iter().collect();

    let mut call = function.call()?;
    let warm = Instant::now();
    let outputs = call.invoke_with(&input_refs)?;
    println!("first call: {:.2} ms", warm.elapsed().as_secs_f64() * 1e3);
    for (i, out) in outputs.iter().enumerate() {
        println!("  output{i}: {:?} {}", out.shape(), out.element_type());
    }

    let start = Instant::now();
    for _ in 0..iters {
        let outputs = call.invoke_with(&input_refs)?;
        // Include readback, as a real service would.
        for out in &outputs {
            out.to_bytes()?;
        }
    }
    let per_call = start.elapsed().as_secs_f64() * 1e3 / iters.max(1) as f64;
    println!("{iters} iters: {per_call:.3} ms/call (incl. d2h readback)");
    Ok(())
}

/// Extracts `(shape, element type)` of each argument from a declaration like
/// `sync func @main(%input0: tensor<1x3x224x224xf32>, %input1: tensor<1x64xi64>) -> (...)`.
#[cfg(iree_linked)]
type TensorSpec = (Vec<usize>, iree_runtime::ElementType);

#[cfg(iree_linked)]
fn parse_input_tensors(decl: &str) -> Result<Vec<TensorSpec>, Box<dyn std::error::Error>> {
    use iree_runtime::ElementType;

    let args = decl.split("->").next().unwrap_or(decl);
    let mut out = Vec::new();
    for chunk in args.split("tensor<").skip(1) {
        let body = chunk.split('>').next().ok_or("unterminated tensor type")?;
        let mut parts: Vec<&str> = body.split('x').collect();
        let elem = parts.pop().ok_or("empty tensor type")?;
        let elem = match elem {
            "f32" => ElementType::FLOAT32,
            "f16" => ElementType::FLOAT16,
            "bf16" => ElementType::BFLOAT16,
            "f64" => ElementType::FLOAT64,
            "i64" => ElementType::INT64,
            "i32" => ElementType::INT32,
            "i16" => ElementType::INT16,
            "i8" => ElementType::INT8,
            "i1" => ElementType::BOOL8,
            other => return Err(format!("unsupported element type `{other}`").into()),
        };
        let shape = parts
            .iter()
            .map(|d| {
                d.parse::<usize>().map_err(|_| {
                    format!("dynamic or invalid dim `{d}` in `{body}`; pass explicit shapes")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        out.push((shape, elem));
    }
    Ok(out)
}
