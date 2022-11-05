import { times } from "lodash";
import { Neuron } from "./neuron";
import { squashFunctions } from "../squash";

export class Layer {
  private neurons: Neuron[];

  constructor(n: number, squashFunction = squashFunctions.identity) {
    this.neurons = times(n, () => new Neuron(squashFunction));
  }

  get size() {
    return this.neurons.length;
  }

  getNeurons() {
    return this.neurons;
  }

  forEachNeuron(f: (n: Neuron, i: number) => void) {
    this.neurons.forEach(f);
  }

  setDebugKeys(layerKey: string) {
    this.neurons.forEach((neuron, i) => neuron.setDebugKey(`${layerKey}_${i}`));
  }
}
