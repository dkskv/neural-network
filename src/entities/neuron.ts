import { zipWith, map, size, random, times } from "lodash";
import { squashFunctions } from "../squash";
import { dotProduct } from "../utils";

export class Neuron {
  debugKey: string | null = null;

  // Входы
  private inputNeurons: Neuron[] = [];
  private weights: number[] = [];

  /** Производная функции общей ошибки по взвешенной сумме нейрона */
  private delta = 0;
  output = 0;

  constructor(private squashFunction = squashFunctions.identity) {}

  setInputConnections(inputNeurons: Neuron[], initialWeights?: number[]) {
    this.inputNeurons = inputNeurons;
    this.weights =
      initialWeights ?? times(this.inputNeurons.length, this.generateWeight);
  }

  private generateWeight() {
    /** todo: уточнить, какой диапазон будет более правильным */
    return random(-10, 10);
  }

  get inputValues() {
    return map(this.inputNeurons, "output");
  }

  get weightedSum() {
    return dotProduct(this.inputValues, this.weights);
  }

  get hasWeights() {
    return size(this.weights) > 0;
  }

  calcOutput() {
    this.output = this.squashFunction.original(this.weightedSum);
  }

  propagateDelta() {
    this.inputNeurons.forEach((inputNeuron, i) => {
      if (!inputNeuron.hasWeights) {
        // Нет смысла считать градиент для нейрона без весов
        return;
      }

      /** Производная функции суммы по выходу конкретного входного нейрона */
      const gradient = this.weights[i] * this.delta;
      return inputNeuron.addToDelta(gradient);
    });
  }

  adjustWeights(rate: number) {
    this.weights = zipWith(
      this.weights,
      this.inputValues,
      (weight, inputValue) => {
        /** Производная функции ошибки по конкретному весу */
        const weightGradient = inputValue * this.delta;
        return weight - weightGradient * rate;
      }
    );
  }

  updateDelta() {
    /** Производная функции активации по взвешенной сумме */
    const gradient = this.squashFunction.derivate(this.output);
    this.delta *= gradient;
  }

  addToDelta(a: number) {
    this.delta += a;
  }

  // todo: pushDeltaInBatch() для пакетного градиента

  clearDelta() {
    this.delta = 0;
  }

  setDebugKey(key: string) {
    this.debugKey = key;
  }
}

export class BiasNeuron extends Neuron {
  output = 1;
}
