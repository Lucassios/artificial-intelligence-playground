package com.lucasmarques.artificialintelligence.backprop;

import java.util.Arrays;
import java.util.Random;

public class Layer {

    private float[] output;
    private float[] input;
    private float[] weights;
    private float[] dWeights;
    private Random random;

    public Layer(int inputSize, int outputSize) {
        
        inputSize += 1; // bias node
        this.input = new float[inputSize];
        this.output = new float[outputSize];
        this.weights = new float[inputSize * outputSize]; // represents all nodes connections
        this.dWeights = new float[this.weights.length];
        this.random = new Random();

        initWeights();
        
    }

    private void initWeights() {
        for (int i = 0; i < this.weights.length; i++) {
            this.weights[i] = (random.nextFloat() - 0.5f) * 4f; // [-2,2]
        }
    }

    public float[] run(float[] inputArray) {

        System.arraycopy(inputArray, 0, this.input, 0, inputArray.length);

        this.input[this.input.length - 1] = 1;
        int offset = 0; // aux variable to get the correct neuron connection index

        for (int i = 0; i < this.output.length; i++) {
            for (int j = 0; j < this.input.length; j++) {
                this.output[i] += this.weights[offset + j] * this.input[j];
            }
            this.output[i] = ActivationFunction.sigmoid(this.output[i]);
            offset += this.input.length;
        }

        return Arrays.copyOf(this.output, this.output.length);

    }

    public float[] train(float[] error, float learningRate, float momentum) {

        int offset = 0; // aux variable to get the correct neuron connection index
        float[] nextError = new float[this.input.length];

        for (int i = 0; i < this.output.length; i++) {

            float delta = error[i] * ActivationFunction.dSigmoid(this.output[i]);

            for (int j = 0; j < this.input.length; j++) {
                int weightIndex = offset + j;
                nextError[j] = nextError[j] + this.weights[weightIndex] * delta;
                float dw = this.input[j] * delta * learningRate;
            }

        }

    }

}
