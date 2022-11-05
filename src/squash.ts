import { constant, Dictionary, identity } from "lodash";

/** Функция активации */
export interface ISquashFunction {
  /** Возвращает нормированное число */
  original(a: number): number;
  derivate(a: number): number;
}

export const squashFunctions: Dictionary<ISquashFunction> = {
  identity: {
    original: identity,
    derivate: constant(1),
  },
  sigmoid: {
    original(x: number) {
      return 1 / (1 + Math.exp(-x));
    },
    derivate(x: number) {
      const a = this.original(x);
      return a * (1 - a);
    },
  },
  relu: {
    original(x: number) {
      return x > 0 ? x : 0;
    },
    derivate(x: number) {
      return x > 0 ? 1 : 0;
    },
  },
};
