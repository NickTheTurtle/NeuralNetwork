class NeuralNetwork {
  constructor(layers, {activation, activationDerivative} = {activation: true, activationDerivative: true}) {
    // Check if layers is a valid argument
    // Initialize neural network
    if (!Array.isArray(layers) || layers.length < 2) {
      throw Error("Layers must be specified as an array of length at least 2");
    }
    this.weights = [];
    this.biases = [];
    for (let i = 0, l = layers.length; i < l; ++i) {
      let currentLayer = layers[i];
      if (typeof currentLayer === "number" && Number.isInteger(currentLayer) && currentLayer > 0) {
        let numWeights = layers[i + 1];
        if (i < l - 1) {
          this.weights.push([]);
        }
        if (i) {
          this.biases.push([]);
        }

        // Seed weights and biases
        for (let j = 0; j < currentLayer; ++j) {
          if (i < l - 1) {
            let weights = [];
            for (let k = 0; k < numWeights; ++k) {
              weights.push(Math.random() * 2 - 1);
            }
          this.weights[i].push(weights);
          }
          if (i) {
            this.biases[i - 1].push(Math.random() * 2 - 1);
          }
        }
      } else {
        throw Error("Array used to specify NeuralNetwork layers must consist solely of positive integers");
      }
    }

    // Set activation function
    if (typeof activation === "function") {
      this.activation = activation;
      if (typeof activationDerivative === "function") {
        this.activationDerivative = activationDerivative;
      } else {
        this.activationDerivative = (x) => {
          let h = 0.001;
          return (this.activation(x + h) - this.activation(x - h)) / (2 * h);
        };
      }
    } else if (activation === "tanh") {
      this.activation = (x) => Math.tanh(x);
      this.activationDerivative = (x) => 1 - Math.tanh(x) ** 2;
    } else if (activation === "arctan" || activation === "atan") {
      this.activation = (x) => Math.atan(x);
      this.activationDerivative = (x) => 1 / (1 + x ** 2);
    } else {
      this.activation = (x) => 1 / (1 + Math.exp(-x));
      this.activationDerivative = (x) => Math.exp(-x) / (1 + Math.exp(-x)) ** 2;
    }
    console.log("Successfully initialized NeuralNetwork");
    return this;
  }
  run(input, training) {
    // Check if input is valid
    if (!training && (!Array.isArray(input) || input.length !== this.weights[0].length || input.find((a) => typeof a !== "number"))) {
      throw Error(`Input for this NeuralNetwork must be an array of ${this.weights[0].length} numbers`);
    }

    // Forward propagation
    let currentInput;
    if (training) {
      currentInput = [input.map((a) => {return {before: a, after: a}})];
    } else {
      currentInput = [...input];
    }
    for (let i = 0, l = this.weights.length; i < l; ++i) {
      let newInput = [];
      for (let j = 0, m = this.weights[i][0].length, n = (training ? currentInput[i] : currentInput).length; j < m; ++j) {
        let sum = this.biases[i][j];
        for (let k = 0; k < n; ++k) {
          sum += (training ? currentInput[i][k].after : currentInput[k]) * this.weights[i][k][j];
        }
        if (training) {
          newInput.push({
            before: sum,
            after: this.activation(sum)
          });
        } else {
          newInput.push(this.activation(sum));
        }
      }
      if (training) {
        currentInput.push(newInput);
      } else {
        currentInput = newInput;
      }
    }
    return currentInput;
  }
  train(data, learningRate = 0.1, iterations = 10000, maxTime) {
    // Check if data, iterations, and maxTime is valid
    let inputLength = this.weights[0].length,
        outputLength = this.weights[this.weights.length - 1][0].length,
        startTime = (new Date).getTime();
    if (!Array.isArray(data) || data.length < 1
      || data.find((a) => !Array.isArray(a.input) || a.input.length !== inputLength || a.input.find((b) => typeof b !== "number") || !Array.isArray(a.output) || a.output.length !== outputLength || a.output.find((b) => typeof b !== "number"))
    ) {
      throw Error(`Training data not formatted correctly`);
    }
    if (typeof iterations !== "number" && !Number.isInteger(iterations) && iterations < 1) {
      throw Error("Iterations must be a positive integer");
    }
    if (!(maxTime === undefined || maxTime === null) && (typeof iterations !== "number" && iterations <= 0)) {
      throw Error("Max time must be one of undefined, null, or a positive number");
    }
    // Backward propagation
    for (let i = 0; i < iterations; ++i) {
      if (maxTime && maxTime < (new Date).getTime() - startTime) {
        console.log(`Training ended due to time limit reached\nTime spent: ${(new Date).getTime() - startTime} ms`);
        return this;
      }
      for (let j = 0, l = data.length; j < l; ++j) {
        let currentData = data[j],
            result = this.run(currentData.input, true),
            outputLayer = result[result.length - 1],
            outputLayerError = [],
            errors = [];
        for (let k = 0, m = outputLayer.length; k < m; ++k) {
          let currentOutputNeuron = outputLayer[k];
          outputLayerError.push((currentOutputNeuron.after - currentData.output[k]) * this.activationDerivative(currentOutputNeuron.before));
        }
        errors.push(outputLayerError);
        for (let k = result.length - 1; k > 1; --k) {
          let previousErrors = errors[0],
              newErrors = [],
              currentLayerWeights = this.weights[k - 1],
              previousResult = result[k - 1];
          for (let i = 0, n = currentLayerWeights.length; i < n; ++i) {
            let sum = 0,
                currentNeuronWeights = currentLayerWeights[i];
            for (let j = 0, o = currentNeuronWeights.length; j < o; ++j) {
              sum += currentNeuronWeights[j] * previousErrors[j];
            }
            newErrors.push(sum * this.activationDerivative(previousResult[i].before));
          }
          errors.unshift(newErrors);
        }
        for (let k = 0, n = this.biases.length; k < n; ++k) {
          let currentLayerWeights = this.weights[k],
              currentLayerBiases = this.biases[k],
              currentLayerErrors = errors[k],
              currentLayerResults = result[k];
          for (let i = 0, o = currentLayerBiases.length; i < o; ++i) {
            let change = learningRate * currentLayerErrors[i];
            for (let j = 0, p = currentLayerWeights.length; j < p; ++j) {
              currentLayerWeights[j][i] -= change * currentLayerResults[j].after;
            }
            currentLayerBiases[i] -= change;
          }
        }
      }
    }
    console.log(`Training ended due to iterations reached\nTime spent: ${(new Date).getTime() - startTime} ms`);
    return this;
  }
  test(data) {
    let inputLength = this.weights[0].length,
        outputLength = this.weights[this.weights.length - 1][0].length;
    if (!Array.isArray(data) || data.length < 1
      || data.find((a) => !Array.isArray(a.input) || a.input.length !== inputLength || a.input.find((b) => typeof b !== "number") || !Array.isArray(a.output) || a.output.length !== outputLength || a.output.find((b) => typeof b !== "number"))
    ) {
      throw Error(`Training data not formatted correctly`);
    }
    let num = 0;
    for (let i = 0, l = data.length; i < l; ++i) {
      let currentData = data[i],
          result = this.run(currentData.input),
          correct = true;
      for (let j = 0, m = result.length; j < m; ++j) {
        if (Math.round(result[j]) !== Math.round(currentData.output)) {
          correct = false;
          break;
        }
      }
      if (correct) {
        ++num;
      }
    }
    console.log(`Testing ended\nAccuracy rate: ${Math.round(num / data.length * 10000) / 100}%`);
    return this;
  }
  toFunction() {
    return `function(input) {
  let weights = ${JSON.stringify(this.weights)},
      biases = ${JSON.stringify(this.biases)},
      activation = ${this.activation.toString()};
  if (!Array.isArray(input) || input.length !== weights[0].length || input.find((a) => typeof a !== "number")) {
    throw Error("Input must be an array of weights[0].length numbers");
  }
  let currentInput = [...input];
  for (let i = 0, l = weights.length; i < l; ++i) {
    let newInput = [];
    for (let j = 0, m = weights[i][0].length, n = currentInput.length; j < m; ++j) {
      let sum = biases[i][j];
      for (let k = 0; k < n; ++k) {
        sum += currentInput[k] * weights[i][k][j];
      }
      newInput.push(activation(sum));
    }
    currentInput = newInput;
  }
  return currentInput;
};`;
  }
}
