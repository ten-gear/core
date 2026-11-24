# tsgrad

A small PyTorch-like ML framework built from scratch in TypeScript, with no external ML libraries, all math implemented by hand.

Inspired by [@srush_nlp](https://x.com/srush_nlp)'s [minitorch](https://github.com/minitorch/minitorch).

## Install

```bash
npm install @tsgrad/core
```

`@tsgrad/core` ships the compiled library, TypeScript declarations, the pure TypeScript CPU backend, and the WebGPU backend. npm also installs the optional native addon whose `os` and `cpu` match the machine. Windows x64 and Linux x64 install two addons, CPU and CUDA 12, because both declare the same platform. The loader uses the CUDA addon when `cudaGetDeviceCount` reports a device, and otherwise uses the CPU addon. The CUDA package bundles the CUDA runtime, so a user needs an NVIDIA driver and does not need the CUDA toolkit.

Alpine and other platforms without a published addon still install. `auto` continues to WebGPU, then the TypeScript CPU backend.

```typescript
import { initBackend, Tensor } from "@tsgrad/core";

await initBackend({ backend: "auto" });
```

## Tensor backends

Initialize a backend before creating tensors. Inside this repo the same functions are imported from `src/`:

```typescript
import { initBackend } from "./src/backend/index.js";

await initBackend({ backend: "auto" });
```

`auto` prefers a locally built `backend/build/Release/tsgrad_native.node`, then the installed CUDA addon when a GPU is present, then the installed CPU addon, then WebGPU, then the TypeScript CPU backend. You can explicitly request `"cuda"`, `"native"`, `"webgpu"`, or `"cpu"`; unavailable explicit choices throw instead of silently changing backend.

WebGPU works in browsers through `navigator.gpu` and in Node through the
optional `webgpu` package. Tensor operations only enqueue GPU work. Reading
data back is asynchronous:

```typescript
const values = await tensor.toFloat32();
const scalar = await tensor.itemAsync();
```

Set `TSGRAD_BACKEND=webgpu` (or another backend name) to override `auto` in
Node. `project/benchmark_backend.ts` times a synchronized matmul workload.

---

## What's implemented

- **Scalar** - scalar values that track their own computation history
- **Reverse-mode autograd** - automatic differentiation via a dynamically built computation graph
- **Backpropagation** - gradient computation via reverse traversal of the graph
- **Parameters & Modules** - base classes for building trainable components
- **Neural network layers** - `Linear` layers with configurable activation functions (`relu`, `sigmoid`)
- **SGD / SGD + momentum** - gradient descent optimizers

## Example

```typescript
export class SimpleNetwork extends Network {
  constructor() {
    super();
    this.addLayer(Linear, "relu", 2, 3);
    this.addLayer(Linear, "sigmoid", 3, 1);
  }
}
```
  
# To download and test on the mnist dataset run the following from inside framework
```bash
mkdir project\data\mnist

curl.exe -L https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz -o project\data\mnist\train-images-idx3-ubyte.gz
curl.exe -L https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz -o project\data\mnist\train-labels-idx1-ubyte.gz
curl.exe -L https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz -o project\data\mnist\t10k-images-idx3-ubyte.gz
curl.exe -L https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz -o project\data\mnist\t10k-labels-idx1-ubyte.gz
```  
  
## Roadmap  
Working on tensors now, then CUDA next.  
