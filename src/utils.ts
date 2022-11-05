import { multiply, sum, zipWith } from "lodash";

/** Сумма квадратов отклонений */
function estimateSquaredError(
  targets: number[],
  outputs: number[]
): number {
  /** .5 для получения более простой производной */
  return sum(zipWith(targets, outputs, (t, o) => 0.5 * (t - o) ** 2));
}

/** 
 * Частная производная функции `estimateSquaredError` по конкретному выходу.
 * `target` берется за константу.
 */
export function squaredErrorDerivate(target: number, output: number) {
  return output - target;
}

/** Скалярное произведение 2-х векторов */
export function dotProduct(arr1: number[], arr2: number[]) {
  return sum(zipWith(arr1, arr2, multiply));
}

/** Обойти массив попарно */
export function pairwise<T>(arr: T[], f: (a: T, b: T, index: number) => void) {
  arr.reduce((a, b, i) => (f(a, b, i), b));
}
