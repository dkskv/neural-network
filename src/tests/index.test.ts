import { round } from "lodash";
import { Layer } from "../entities/layer";
import { Perceptron } from "../entities/perceptron";
import { squashFunctions } from "../squash";

/** todo: Написать тесты для активации и обратного распространения ошибки */

/**
 * todo: часто застревает и дает выход ~0.5 для всех входов.
 * Частота застревания зависит от диапазона начальных весов.
 */
test("Обучение работе функции XOR", () => {
  const network = new Perceptron({
    input: new Layer(2, squashFunctions.identity),
    hidden: [new Layer(3, squashFunctions.sigmoid)],
    output: new Layer(1, squashFunctions.sigmoid),
  });

  const trainingSet = [
    { input: [1, 1], target: [0] },
    { input: [0, 1], target: [1] },
    { input: [1, 0], target: [1] },
    { input: [0, 0], target: [0] },
  ];

  const epochCount = 20000;
  const learningRate = 0.3;

  for (let i = 0; i < epochCount; i++) {
    trainingSet.forEach(({ input, target }) => {
      network.activate(input);
      network.train(target, learningRate);
    });
  }

  trainingSet.forEach(function testCase({ input, target }) {
    network.activate(input);
    expect(network.outputs.map((a) => round(a, 1))).toEqual(target);
  });
});
