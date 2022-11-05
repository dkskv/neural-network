import { drop, forEach, forEachRight, map, size } from "lodash";
import { Layer } from "./layer";
import { BiasNeuron, Neuron } from "./neuron";
import { pairwise, squaredErrorDerivate } from "../utils";

interface ILayersConfig {
  input: Layer;
  hidden: Layer[];
  output: Layer;
}

export class Perceptron {
  constructor(private layersConfig: ILayersConfig) {
    this.bindLayers();
    this.setDebugKeys();
  }

  private get layers() {
    return [this.inputLayer, ...this.hiddenLayers, this.outputLayer];
  }

  private get inputLayer() {
    return this.layersConfig.input;
  }

  private get hiddenLayers() {
    return this.layersConfig.hidden;
  }

  private get outputLayer() {
    return this.layersConfig.output;
  }

  get outputs() {
    return map(this.outputLayer.getNeurons(), "output");
  }

  private bindLayers() {
    const bias = new BiasNeuron();
    bias.setDebugKey("b");

    pairwise(this.layers, (prevLayer, nextLayer) => {
      const inputs = [...prevLayer.getNeurons(), bias];
      nextLayer.forEachNeuron((neuron) => neuron.setInputConnections(inputs));
    });
  }

  activate(inputs: number[]) {
    if (size(inputs) !== this.inputLayer.size) {
      throw new Error("Передано неверное количество входов");
    }

    this.inputLayer.forEachNeuron((neuron, i) => {
      // todo: устанавливать вход, чтобы пропустить через функцию активации
      neuron.output = inputs[i];
    });

    forEach(drop(this.layers), (layer) => {
      layer.forEachNeuron((neuron) => neuron.calcOutput());
    });
  }

  train(targets: number[], learningRate: number) {
    this.propagateBackward(targets);
    this.applyStochasticGradient(learningRate);
    this.clearDeltas();
  }

  private propagateBackward(targets: number[]) {
    if (size(targets) !== this.outputLayer.size) {
      throw new Error("Передано неверное количество ожидаемых выходов");
    }

    this.outputLayer.forEachNeuron((neuron, i) => {
      /** Производная общей ошибки по конкретному выходу */
      const gradient = squaredErrorDerivate(targets[i], neuron.output);
      neuron.addToDelta(gradient);
    });

    forEachRight(drop(this.layers), (layer, layerIndex) => {
      layer.forEachNeuron((neuron) => {
        neuron.updateDelta();

        if (layerIndex !== 0) {
          neuron.propagateDelta();
        }
      });
    });
  }

  private applyStochasticGradient(learningRate: number) {
    this.forEachNeuron((neuron) => neuron.adjustWeights(learningRate));
  }

  private clearDeltas() {
    this.forEachNeuron((neuron) => neuron.clearDelta());
  }

  forEachNeuron(f: (neuron: Neuron, i: number) => void) {
    forEach(this.layers, (layer) => layer.forEachNeuron(f));
  }

  private setDebugKeys() {
    this.inputLayer.setDebugKeys("in");
    forEach(this.hiddenLayers, (layer, i) => layer.setDebugKeys(`h${i}`));
    this.outputLayer.setDebugKeys("out");
  }
}
