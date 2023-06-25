import { Matrix } from 'ml-matrix';
import { SigmoidNeuron } from './sigmoid-neuron.js';

const sigmaNeuron = new SigmoidNeuron();

console.log('Initial synaptic weights (random): \n', sigmaNeuron.synapticWeights.to2DArray(), '\n');

const trainingSetInputs = new Matrix([
    [0, 0, 1],
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1],
]);
const trainingSetOutputs = new Matrix([[0], [1], [1], [0]]);
sigmaNeuron.train(trainingSetInputs, trainingSetOutputs, 10000);

console.log('Synaptic weights after training: \n', sigmaNeuron.synapticWeights.to2DArray(), '\n');

console.log('Given a new situation [1, 0, 0]: ');
console.log(sigmaNeuron.think(new Matrix([[1, 0, 0]])).to1DArray());
