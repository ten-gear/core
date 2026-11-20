export function mul(a: number, b: number): number{
    return a * b;
}

export function id(a: number): number{
    return a;
}

export function add(a: number, b: number): number{
    return a + b;
}

export function sub(a: number, b: number): number{
    return a - b;
}

export function div(a: number, b: number): number{
    return a / b;
}

// Just means flip the sign
export function neg(a: number): number{
    return -a;
}

export function lt(a: number, b: number): number{
    return a < b ? 1 : 0;
}

export function eq(a: number, b: number): number{
    return a === b ? 1 : 0;
}

export function max(a : number, b: number): number{
    return a > b ? a : b;
}

export function isClose(a: number, b: number): number{
    return Math.abs(a - b) < 1e-2 ? 1 : 0;
}

export function sigmoid(a: number): number{
    if (a >= 0)
        return 1.0/(1.0 + Math.exp(-a));  // e^a
    else
        return Math.exp(a) / (1.0 + Math.exp(a));
}

export function relu(a: number): number{
    return Math.max(0, a);
}

// Abramowitz and Stegun 7.1.26, matching erf closely.
function erf(a: number): number{
    const sign = a < 0 ? -1 : 1;
    const absZ = Math.abs(a);
    const t = 1 / (1 + 0.3275911 * absZ);
    return sign * (1 - (((((1.061405429 * t - 1.453152027) * t)
        + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-absZ * absZ));
}

export function gelu(a: number): number{
    return 0.5 * a * (1 + erf(a * Math.SQRT1_2));
}

export function geluBackward(grad: number, input: number): number{
    const cdf = 0.5 * (1 + erf(input * Math.SQRT1_2));
    const pdf = Math.exp(-0.5 * input * input) / Math.sqrt(2 * Math.PI);
    return grad * (cdf + input * pdf);
}

export function leakyrelu(a: number, alpha: number = 0.01): number{
    if (a > 0)
        return a;
    return alpha * a;
}

export function log(a: number): number{
    return Math.log(a); // ln(a)
}

export function exp(a: number): number{
    return Math.exp(a); // e^a
}

export function sqrt(a: number): number{
    return Math.sqrt(a);
}

// reciprocal
export function inv(a: number): number{
    return 1.0 / a;
}

export function logBack(a: number, b: number): number{
    return b / a;
}

export function invBack(a: number, b: number): number{
    return -b / (a * a);
}

export function reluBack(a: number, b: number): number{
    return a > 0 ? b : 0;
}

// For this task 0.3 section, for now I'm going to group everything under array
// This might need to be changed to a general class of iterable later
export function map(func: (x: number) => number): (it: number[]) => number[]{
    return (it: number[]) => it.map(func);
}

export function zipWith(func: (it1: number, it2: number) => number): (it1: number[], it2: number[]) => number[]{
    return (it1: number[], it2: number[]) => it1.map((_, i) => func(it1[i]!, it2[i]!));
}

export function reduce(func: (x: number, y: number) => number, base: number): (it: number[]) => number{
    return (it: number[]) => {let res: number = base; it.forEach(val => res = func(res, val)); return res;};
}

export function negList(it: number[]): number[]{
    return map(neg)(it);
}

export function addLists(it1: number[], it2: number[]): number[]{
    return zipWith(add)(it1, it2);
}

export function sum(it: number[]): number{
    return reduce(add, 0)(it);
}

export function prod(it: number[]): number{
    return reduce(mul, 1)(it);
}