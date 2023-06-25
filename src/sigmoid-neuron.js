import random from 'random';
import seedrandom from 'seedrandom';
import { Matrix } from 'ml-matrix';

export class SigmoidNeuron {
    synapticWeights;

    constructor() {
        random.use(seedrandom('1'));
        this.synapticWeights = new Matrix([[random.float(-1, 1)], [random.float(-1, 1)], [random.float(-1, 1)]]);
    }

    train(trainingSetInputs, trainingSetOutputs, iterations) {
        while (iterations--) {
            const output = this.think(trainingSetInputs);
            const error = Matrix.sub(trainingSetOutputs, output);
            const adjustment = trainingSetInputs.transpose().mmul(error).mmul(this.sigmoidDerivative(output));

            this.synapticWeights = Matrix.add(this.synapticWeights, adjustment);
        }
    }

    think(inputs) {
        return this.sigmoid(inputs.mmul(this.synapticWeights));
    }

    sigmoid(mat) {
        const normalizedMat = mat.to2DArray().map((x) => [1 / (1 + Math.exp(-x))]);
        return new Matrix(normalizedMat);
    }

    sigmoidDerivative(mat) {
        const sigmoidCurveGradient = mat.to2DArray().map((x) => [x * (1 - x)]);
        return new Matrix(sigmoidCurveGradient);
    }
}
